import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from common.abstract_recommender import GeneralRecommender
from common.loss import EmbLoss

class MMGAT(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MMGAT, self).__init__(config, dataset)
        
        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.n_layers = config['n_layers']
        self.dropout = config['dropout']
        self.reg_weight = config['reg_weight']
        self.n_nodes = self.n_users + self.n_items
        
        # Load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.norm_adj = self.get_norm_adj_mat().to(self.device)
        
        # Embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        # Feature transformation layers
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_transform = nn.Sequential(
                nn.Linear(self.v_feat.shape[1], self.feat_embed_dim),
                nn.LayerNorm(self.feat_embed_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout)
            )
            
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_transform = nn.Sequential(
                nn.Linear(self.t_feat.shape[1], self.feat_embed_dim),
                nn.LayerNorm(self.feat_embed_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout)
            )

        # Cross-modal fusion
        self.modal_fusion = nn.Sequential(
            nn.Linear(self.feat_embed_dim * 2, self.feat_embed_dim),
            nn.LayerNorm(self.feat_embed_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )

        # Feature aggregation
        self.feature_aggregation = nn.Sequential(
            nn.Linear(self.embedding_dim + self.feat_embed_dim, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )

        self.reg_loss = EmbLoss()
        self.temperature = nn.Parameter(torch.FloatTensor([0.2]))
        
        # Initialize weight matrices for attention
        self.W_q = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.W_k = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.W_v = nn.Linear(self.embedding_dim, self.embedding_dim)

    def get_norm_adj_mat(self):
        A = sp.dok_matrix((self.n_nodes, self.n_nodes), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        for key, value in data_dict.items():
            A[key] = value
        
        # Compute normalized adjacency matrix
        sumArr = (A > 0).sum(axis=1)
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        
        # Convert to tensor
        L = sp.coo_matrix(L)
        indices = torch.LongTensor(np.array([L.row, L.col]))
        values = torch.FloatTensor(L.data)
        
        return torch.sparse.FloatTensor(indices, values, torch.Size([self.n_nodes, self.n_nodes]))

    def self_attention(self, x):
        # Multi-head self-attention
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        
        attention_weights = F.softmax(torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.embedding_dim), dim=-1)
        attention_weights = F.dropout(attention_weights, self.dropout, training=self.training)
        
        return torch.matmul(attention_weights, V)

    def forward(self):
        # Process multimodal features
        if self.v_feat is not None and self.t_feat is not None:
            image_feats = self.image_transform(self.image_embedding.weight)
            text_feats = self.text_transform(self.text_embedding.weight)
            
            # Cross-modal fusion
            fused_features = self.modal_fusion(torch.cat([image_feats, text_feats], dim=1))
        elif self.v_feat is not None:
            fused_features = self.image_transform(self.image_embedding.weight)
        elif self.t_feat is not None:
            fused_features = self.text_transform(self.text_embedding.weight)
        else:
            fused_features = None

        # Graph convolution
        ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
        all_embeddings = [ego_embeddings]
        
        for _ in range(self.n_layers):
            # Apply dropout to adjacency matrix during training
            if self.training:
                adj = F.dropout(self.norm_adj.to_dense(), self.dropout, training=True)
                adj = adj.to_sparse()
            else:
                adj = self.norm_adj
                
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
            
            # Apply self-attention
            side_embeddings = self.self_attention(side_embeddings)
            
            # Add residual connection and normalization
            ego_embeddings = F.layer_norm(ego_embeddings + side_embeddings, normalized_shape=[self.embedding_dim])
            all_embeddings.append(ego_embeddings)

        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)

        # Integrate multimodal features with item embeddings
        if fused_features is not None:
            i_g_embeddings = self.feature_aggregation(
                torch.cat([i_g_embeddings, fused_features], dim=1)
            )

        return u_g_embeddings, i_g_embeddings

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        u_embeddings, i_embeddings = self.forward()
        
        u_embeddings = u_embeddings[users]
        pos_embeddings = i_embeddings[pos_items]
        neg_embeddings = i_embeddings[neg_items]

        # InfoNCE loss for positive samples
        pos_scores = torch.sum(u_embeddings * pos_embeddings, dim=1)
        pos_scores = torch.exp(pos_scores / self.temperature)

        # Negative sampling loss
        neg_scores = torch.sum(u_embeddings * neg_embeddings, dim=1)
        neg_scores = torch.exp(neg_scores / self.temperature)
        
        # Combine losses with adaptive temperature
        loss = -torch.log(pos_scores / (pos_scores + neg_scores))
        loss = loss.mean()

        # L2 regularization
        reg_loss = self.reg_weight * self.reg_loss(u_embeddings, pos_embeddings, neg_embeddings)
        
        # Add regularization for multimodal features
        if self.v_feat is not None:
            reg_loss += self.reg_weight * torch.norm(self.image_transform(self.image_embedding.weight))
        if self.t_feat is not None:
            reg_loss += self.reg_weight * torch.norm(self.text_transform(self.text_embedding.weight))

        return loss + reg_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        u_embeddings, i_embeddings = self.forward()
        u_embeddings = u_embeddings[user]
        scores = torch.matmul(u_embeddings, i_embeddings.transpose(0, 1))
        return scores