import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp

from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss

class MMGAT(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MMGAT, self).__init__(config, dataset)
        
        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.n_layers = config['n_layers']
        self.knn_k = config['knn_k']
        self.dropout = config['dropout']
        self.reg_weight = config['reg_weight']
        self.temperature = 0.2
        
        self.n_nodes = self.n_users + self.n_items
        
        # Load interaction matrix
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.norm_adj = self.get_norm_adj_mat()
        
        # Initialize embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)
        
        # Move embeddings to device
        self.user_embedding = self.user_embedding.to(self.device)
        self.item_id_embedding = self.item_id_embedding.to(self.device)
        
        # Multimodal components
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_projector = nn.Sequential(
                nn.Linear(self.v_feat.shape[1], self.feat_embed_dim),
                nn.LayerNorm(self.feat_embed_dim),
                nn.ReLU()
            )
            self.image_attention = nn.Sequential(
                nn.Linear(self.feat_embed_dim, self.feat_embed_dim // 2),
                nn.ReLU(),
                nn.Linear(self.feat_embed_dim // 2, 1)
            )
            # Move to device
            self.image_embedding = self.image_embedding.to(self.device)
            self.image_projector = self.image_projector.to(self.device)
            self.image_attention = self.image_attention.to(self.device)
            
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_projector = nn.Sequential(
                nn.Linear(self.t_feat.shape[1], self.feat_embed_dim),
                nn.LayerNorm(self.feat_embed_dim),
                nn.ReLU()
            )
            self.text_attention = nn.Sequential(
                nn.Linear(self.feat_embed_dim, self.feat_embed_dim // 2),
                nn.ReLU(),
                nn.Linear(self.feat_embed_dim // 2, 1)
            )
            # Move to device
            self.text_embedding = self.text_embedding.to(self.device)
            self.text_projector = self.text_projector.to(self.device)
            self.text_attention = self.text_attention.to(self.device)

        # Cross-modal fusion
        self.modal_fusion = nn.Sequential(
            nn.Linear(self.feat_embed_dim * 2, self.feat_embed_dim),
            nn.LayerNorm(self.feat_embed_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        ).to(self.device)
        
        # Dual channel aggregation
        self.channel_weight = nn.Parameter(torch.FloatTensor([0.5, 0.5])).to(self.device)
        self.softmax = nn.Softmax(dim=0)
        
        # Initialize multimodal graphs
        self.mm_adj = None
        self.norm_adj = self.norm_adj.to(self.device)
        with torch.no_grad():
            self.mm_adj = self._init_mm_graph().to(self.device)
        
    def get_norm_adj_mat(self):
        adj_mat = sp.dok_matrix((self.n_nodes, self.n_nodes), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.interaction_matrix.tolil()
        
        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()
        
        row_sum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(row_sum + 1e-10, -0.5).flatten()  # Add epsilon to avoid division by zero
        d_mat_inv = sp.diags(d_inv)
        
        norm_adj = d_mat_inv.dot(adj_mat).dot(d_mat_inv)
        return self._sparse_mx_to_torch_sparse_tensor(norm_adj.tocoo())
        
    def _sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col))).long()
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse_coo_tensor(indices, values, shape)
        
    def _init_mm_graph(self):
        v_adj, t_adj = None, None
        
        if self.v_feat is not None:
            v_emb = self.image_projector(self.image_embedding.weight)
            v_adj = self._build_knn_graph(v_emb)
            
        if self.t_feat is not None:
            t_emb = self.text_projector(self.text_embedding.weight)
            t_adj = self._build_knn_graph(t_emb)
            
        if v_adj is not None and t_adj is not None:
            weights = self.softmax(self.channel_weight)
            return weights[0] * v_adj + weights[1] * t_adj
        return v_adj if v_adj is not None else t_adj
        
    def _build_knn_graph(self, embeddings):
        sim_matrix = F.normalize(embeddings, p=2, dim=1) @ F.normalize(embeddings, p=2, dim=1).t()
        topk_values, topk_indices = torch.topk(sim_matrix, k=self.knn_k, dim=1)
        
        row_indices = torch.arange(embeddings.size(0), device=self.device).view(-1, 1).expand(-1, self.knn_k)
        adj_indices = torch.stack([row_indices.flatten(), topk_indices.flatten()], dim=0)
        adj_values = topk_values.flatten()
        
        adj_tensor = torch.sparse_coo_tensor(
            adj_indices, adj_values, 
            torch.Size([embeddings.size(0), embeddings.size(0)]),
            device=self.device
        )
        return self._normalize_adj(adj_tensor)
        
    def _normalize_adj(self, adj):
        row_sum = torch.sparse.sum(adj, dim=1).to_dense()
        d_inv_sqrt = torch.pow(row_sum + 1e-7, -0.5)
        d_mat = torch.diag(d_inv_sqrt)
        return torch.sparse.mm(torch.sparse.mm(d_mat, adj), d_mat)
        
    def forward(self):
        # Multimodal feature extraction
        image_features = text_features = None
        if self.v_feat is not None:
            image_features = self.image_projector(self.image_embedding.weight)
            image_att = self.image_attention(image_features)
            image_features = image_features * image_att
            
        if self.t_feat is not None:
            text_features = self.text_projector(self.text_embedding.weight)
            text_att = self.text_attention(text_features)
            text_features = text_features * text_att
            
        # Cross-modal fusion
        if image_features is not None and text_features is not None:
            item_features = self.modal_fusion(torch.cat([image_features, text_features], dim=1))
        else:
            item_features = image_features if image_features is not None else text_features
            
        # Graph convolution on items
        item_embeddings = item_features
        for _ in range(self.n_layers):
            item_embeddings = torch.sparse.mm(self.mm_adj, item_embeddings)
            
        # User-item graph convolution
        ego_embeddings = torch.cat([self.user_embedding.weight, self.item_id_embedding.weight], dim=0)
        all_embeddings = [ego_embeddings]
        
        for _ in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.norm_adj, ego_embeddings)
            all_embeddings.append(ego_embeddings)
            
        all_embeddings = torch.stack(all_embeddings, dim=1).mean(dim=1)
        user_embeddings, base_item_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items])
        
        # Combine base embeddings with multimodal features
        final_item_embeddings = base_item_embeddings + F.normalize(item_embeddings, p=2, dim=1)
        return user_embeddings, final_item_embeddings
        
    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        
        user_embeddings, item_embeddings = self.forward()
        
        u_embeddings = user_embeddings[users]
        pos_embeddings = item_embeddings[pos_items]
        neg_embeddings = item_embeddings[neg_items]
        
        # BPR Loss
        pos_scores = torch.sum(u_embeddings * pos_embeddings, dim=1)
        neg_scores = torch.sum(u_embeddings * neg_embeddings, dim=1)
        mf_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        
        # Regularization Loss
        reg_loss = self.reg_weight * (
            torch.norm(u_embeddings) +
            torch.norm(pos_embeddings) +
            torch.norm(neg_embeddings)
        )
        
        return mf_loss + reg_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_embeddings, item_embeddings = self.forward()
        u_embeddings = user_embeddings[user]
        scores = torch.matmul(u_embeddings, item_embeddings.t())
        return scores