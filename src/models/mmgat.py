import torch
import torch.nn as nn
import torch.nn.functional as F
from common.abstract_recommender import GeneralRecommender
import scipy.sparse as sp
import numpy as np

class MMGAT(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MMGAT, self).__init__(config, dataset)
        
        # Model hyperparameters
        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.num_heads = 4
        self.dropout = 0.1
        self.n_layers = config['n_layers']
        self.temperature = 0.2
        self.lambda_coeff = config['lambda_coeff']
        self.knn_k = config['knn_k']
        self.reg_weight = config['reg_weight']
        self.n_nodes = self.n_users + self.n_items
        
        # Load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        
        # Initialize embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)
        
        # Initialize feature transformations
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)
            self.image_proj = nn.Linear(self.feat_embed_dim, self.embedding_dim)
        
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)
            self.text_proj = nn.Linear(self.feat_embed_dim, self.embedding_dim)

        # Multimodal fusion weights
        if self.v_feat is not None and self.t_feat is not None:
            self.modal_weight = nn.Parameter(torch.Tensor([0.5, 0.5]))
            self.softmax = nn.Softmax(dim=0)
            
        # Get normalized adjacency matrix
        self.norm_adj = self.get_norm_adj_mat().to(self.device)
        self.mm_adj = None
        self.build_item_graph = True
        
        # Initialize attention weights
        self.attention = nn.Parameter(torch.ones(self.num_heads, self.embedding_dim, 1))
        nn.init.xavier_uniform_(self.attention)

    def get_norm_adj_mat(self):
        # Following DRAGON's implementation for robust adjacency matrix computation
        A = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        for key, value in data_dict.items():
            A[key] = value
        
        # Compute normalized Laplacian
        sumArr = (A > 0).sum(axis=1)
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        
        # Convert to sparse tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor([row, col])
        data = torch.FloatTensor(L.data)
        
        return torch.sparse.FloatTensor(i, data, torch.Size((self.n_nodes, self.n_nodes)))

    def compute_attention_scores(self, features):
        # Multi-head attention implementation
        attention_heads = []
        for head in range(self.num_heads):
            scores = torch.mm(features, self.attention[head])
            attention_weights = F.softmax(scores, dim=0)
            attended_features = features * attention_weights
            attention_heads.append(attended_features)
        
        return torch.mean(torch.stack(attention_heads), dim=0)

    def forward(self, build_item_graph=True):
        # Process multimodal features
        modal_features = None
        if self.v_feat is not None:
            image_feats = self.image_trs(self.image_embedding.weight)
            image_feats = F.dropout(image_feats, self.dropout, training=self.training)
            image_feats = self.image_proj(image_feats)
            modal_features = image_feats
        
        if self.t_feat is not None:
            text_feats = self.text_trs(self.text_embedding.weight)
            text_feats = F.dropout(text_feats, self.dropout, training=self.training)
            text_feats = self.text_proj(text_feats)
            if modal_features is None:
                modal_features = text_feats
            else:
                weight = self.softmax(self.modal_weight)
                modal_features = weight[0] * image_feats + weight[1] * text_feats

        # Multi-layer propagation
        ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
        all_embeddings = [ego_embeddings]
        
        for _ in range(self.n_layers):
            # Graph convolution
            side_embeddings = torch.sparse.mm(self.norm_adj, ego_embeddings)
            
            # Apply attention if we have modal features
            if modal_features is not None:
                item_embeddings = side_embeddings[self.n_users:]
                attended_features = self.compute_attention_scores(item_embeddings)
                side_embeddings = torch.cat((side_embeddings[:self.n_users], attended_features), dim=0)
            
            # Dropout and normalization
            side_embeddings = F.dropout(side_embeddings, self.dropout, training=self.training)
            norm_embeddings = F.normalize(side_embeddings, p=2, dim=1)
            all_embeddings.append(norm_embeddings)
        
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)

        # Combine with modal features if available
        if modal_features is not None:
            i_g_embeddings = i_g_embeddings + modal_features

        return u_g_embeddings, i_g_embeddings

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        u_embeddings, i_embeddings = self.forward(build_item_graph=self.build_item_graph)
        self.build_item_graph = False

        u_embeddings = u_embeddings[users]
        pos_embeddings = i_embeddings[pos_items]
        neg_embeddings = i_embeddings[neg_items]

        # BPR Loss with improved numerical stability
        pos_scores = torch.sum(u_embeddings * pos_embeddings, dim=1)
        neg_scores = torch.sum(u_embeddings * neg_embeddings, dim=1)
        bpr_loss = torch.mean(F.softplus(neg_scores - pos_scores))

        # Contrastive Loss for multimodal features
        contrastive_loss = torch.tensor(0.0, device=self.device)
        if self.v_feat is not None and self.t_feat is not None:
            image_feats = F.normalize(self.image_trs(self.image_embedding.weight[pos_items]), dim=1)
            text_feats = F.normalize(self.text_trs(self.text_embedding.weight[pos_items]), dim=1)
            
            sim = torch.matmul(image_feats, text_feats.t()) / self.temperature
            labels = torch.arange(sim.size(0), device=self.device)
            contrastive_loss = F.cross_entropy(sim, labels)

        # L2 regularization
        reg_loss = self.reg_weight * (
            torch.norm(u_embeddings) +
            torch.norm(pos_embeddings) +
            torch.norm(neg_embeddings)
        )

        return bpr_loss + 0.1 * contrastive_loss + reg_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        u_embeddings, i_embeddings = self.forward(build_item_graph=True)
        
        u_embeddings = u_embeddings[user]
        scores = torch.matmul(u_embeddings, i_embeddings.transpose(0, 1))
        return scores