import os
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
        self.knn_k = config['knn_k']
        self.reg_weight = config['reg_weight']
        self.mm_fusion_mode = config['mm_fusion_mode']
        self.dropout = config['dropout']
        self.device = config['device']
        
        # Initialize embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        # Load and process features
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_proj = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)
            self.image_transform = nn.Sequential(
                nn.Linear(self.feat_embed_dim, self.feat_embed_dim),
                nn.LayerNorm(self.feat_embed_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout)
            )
            
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_proj = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)
            self.text_transform = nn.Sequential(
                nn.Linear(self.feat_embed_dim, self.feat_embed_dim),
                nn.LayerNorm(self.feat_embed_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout)
            )

        # Prepare adjacency matrices
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        indices, norm_adj = self.get_adj_mat()
        self.norm_adj = self._convert_sp_mat_to_sp_tensor(norm_adj).to(self.device)
        
        # Initialize modality weights for fusion
        if self.v_feat is not None and self.t_feat is not None:
            self.modal_weights = nn.Parameter(torch.ones(2) / 2)
        
        # Feature fusion layers
        fusion_dim = self.feat_embed_dim * 2 if self.mm_fusion_mode == 'concat' else self.feat_embed_dim
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_dim, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
        self.predictor = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, self.embedding_dim)
        )

        # Loss functions
        self.criterion = nn.BCEWithLogitsLoss()
        self.reg_loss = EmbLoss()

    def get_adj_mat(self):
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.interaction_matrix.tolil()
        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()

        row_sum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(row_sum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        norm_adj = d_mat.dot(adj_mat).dot(d_mat)
        
        # Build indices for the sparse matrix
        norm_adj = norm_adj.tocoo()
        indices = np.vstack((norm_adj.row, norm_adj.col))
        
        return indices, norm_adj

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        indices = torch.LongTensor([coo.row, coo.col])
        values = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(indices, values, torch.Size(coo.shape))

    def get_knn_graph(self, embeddings, k):
        sim = F.normalize(embeddings, p=2, dim=1) @ F.normalize(embeddings, p=2, dim=1).t()
        topk_indices = torch.topk(sim, k=k, dim=-1)[1]
        mask = torch.zeros_like(sim)
        mask.scatter_(1, topk_indices, 1)
        mask = (mask + mask.t() > 0).float()
        return mask * sim

    def process_modalities(self, dropout=False):
        image_emb = text_emb = None
        
        if self.v_feat is not None:
            image_emb = self.image_transform(self.image_proj(self.image_embedding.weight))
            if dropout:
                image_emb = F.dropout(image_emb, p=self.dropout, training=self.training)
                
        if self.t_feat is not None:
            text_emb = self.text_transform(self.text_proj(self.text_embedding.weight))
            if dropout:
                text_emb = F.dropout(text_emb, p=self.dropout, training=self.training)
                
        if self.v_feat is not None and self.t_feat is not None:
            weights = F.softmax(self.modal_weights, dim=0)
            if self.mm_fusion_mode == 'concat':
                fused_emb = torch.cat([image_emb, text_emb], dim=1)
            else:  # weighted sum
                fused_emb = weights[0] * image_emb + weights[1] * text_emb
        elif self.v_feat is not None:
            fused_emb = image_emb
        else:
            fused_emb = text_emb
            
        return self.fusion_layer(fused_emb)

    def forward(self, adj_matrix=None):
        if adj_matrix is None:
            adj_matrix = self.norm_adj
            
        # Process modalities and get fused embeddings
        fused_features = self.process_modalities(dropout=True)
        
        # Combine ID embeddings with fused features for items
        item_embeddings = self.item_id_embedding.weight + fused_features
        
        # Initialize embeddings for message passing
        all_embeddings = torch.cat([self.user_embedding.weight, item_embeddings])
        embeddings_list = [all_embeddings]
        
        # Multi-layer message passing
        for _ in range(self.n_layers):
            all_embeddings = torch.sparse.mm(adj_matrix, all_embeddings)
            all_embeddings = F.normalize(all_embeddings, p=2, dim=1)
            embeddings_list.append(all_embeddings)
        
        # Aggregate embeddings from all layers
        final_embeddings = torch.stack(embeddings_list, dim=1).mean(dim=1)
        user_embeddings, item_embeddings = torch.split(final_embeddings, [self.n_users, self.n_items])
        
        # Apply predictor MLP
        user_embeddings = self.predictor(user_embeddings)
        item_embeddings = self.predictor(item_embeddings)
        
        return user_embeddings, item_embeddings

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        user_embeddings, item_embeddings = self.forward()
        
        u_embeddings = user_embeddings[users]
        pos_embeddings = item_embeddings[pos_items]
        neg_embeddings = item_embeddings[neg_items]

        # Compute user-item similarities
        pos_scores = (u_embeddings * pos_embeddings).sum(dim=1)
        neg_scores = (u_embeddings * neg_embeddings).sum(dim=1)
        
        # BPR loss
        loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8))

        # Regularization
        reg_loss = self.reg_weight * self.reg_loss(u_embeddings, pos_embeddings, neg_embeddings)
        
        # Modality alignment loss if both modalities are present
        modal_loss = 0
        if self.v_feat is not None and self.t_feat is not None:
            image_emb = self.image_transform(self.image_proj(self.image_embedding.weight))
            text_emb = self.text_transform(self.text_proj(self.text_embedding.weight))
            modal_loss = 1 - F.cosine_similarity(image_emb[pos_items], text_emb[pos_items]).mean()
            
        return loss + reg_loss + 0.1 * modal_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        
        user_embeddings, item_embeddings = self.forward()
        u_embeddings = user_embeddings[user]
        
        scores = torch.matmul(u_embeddings, item_embeddings.t())
        return scores

    def predict(self, interaction):
        user = interaction[0]
        item = interaction[1]
        
        user_embeddings, item_embeddings = self.forward()
        
        u_embeddings = user_embeddings[user]
        i_embeddings = item_embeddings[item]
        
        scores = torch.sum(u_embeddings * i_embeddings, dim=1)
        return scores