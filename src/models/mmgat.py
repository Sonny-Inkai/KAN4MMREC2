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
        self.n_ui_layers = config['n_ui_layers']
        self.n_mm_layers = config['n_mm_layers']
        self.knn_k = config['knn_k']
        self.dropout = config['dropout']
        self.reg_weight = config['reg_weight']
        self.mm_fusion_mode = config['mm_fusion_mode']
        self.n_nodes = self.n_users + self.n_items
        
        # Load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.norm_adj = self.get_norm_adj_mat().to(self.device)
        
        # User-item embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        self.init_weights()
        
        # Feature embeddings and transformations
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_encoder = ModalityEncoder(
                input_dim=self.v_feat.shape[1],
                hidden_dim=self.feat_embed_dim,
                output_dim=self.embedding_dim,
                dropout=self.dropout
            )
            
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_encoder = ModalityEncoder(
                input_dim=self.t_feat.shape[1],
                hidden_dim=self.feat_embed_dim,
                output_dim=self.embedding_dim,
                dropout=self.dropout
            )
            
        # Multimodal fusion
        if self.v_feat is not None and self.t_feat is not None:
            self.modal_attention = MultiModalAttention(self.embedding_dim)
        
        # Initialize modality-specific graph adjacency
        self.mm_adj = self.init_modal_graph()
        
        # Loss components
        self.bpr_loss = BPRLoss()
        self.emb_loss = EmbLoss()
        
        # Gradient reversal coefficient
        self.grl_lambda = 0.1
        
    def init_weights(self):
        """Initialize weights with Xavier uniform"""
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        
    def get_norm_adj_mat(self):
        """Construct normalized adjacency matrix"""
        adj_mat = sp.dok_matrix((self.n_nodes, self.n_nodes), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.interaction_matrix.tolil()
        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()
        
        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum + 1e-9, -0.5).flatten()
        d_mat_inv = sp.diags(d_inv)
        norm_adj = d_mat_inv.dot(adj_mat).dot(d_mat_inv)
        
        norm_adj = norm_adj.tocoo()
        indices = np.vstack((norm_adj.row, norm_adj.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(norm_adj.data)
        shape = torch.Size(norm_adj.shape)
        return torch.sparse_coo_tensor(i, v, shape)
        
    def init_modal_graph(self):
        """Initialize multimodal similarity graph"""
        if self.v_feat is not None:
            image_sim = self.compute_modal_similarity(self.image_embedding.weight)
            mm_adj = image_sim
            
        if self.t_feat is not None:
            text_sim = self.compute_modal_similarity(self.text_embedding.weight)
            mm_adj = text_sim if mm_adj is None else (mm_adj + text_sim) / 2
            
        return mm_adj.to(self.device)
        
    def compute_modal_similarity(self, features):
        """Compute KNN-based similarity graph for modality features"""
        norm_features = F.normalize(features, p=2, dim=1)
        sim_matrix = torch.mm(norm_features, norm_features.t())
        
        # KNN graph construction
        topk_values, topk_indices = torch.topk(sim_matrix, k=self.knn_k, dim=1)
        mask = torch.zeros_like(sim_matrix)
        mask.scatter_(1, topk_indices, 1)
        sim_matrix = sim_matrix * mask
        
        # Symmetric normalization
        D = torch.sum(sim_matrix, dim=1)
        D_sqrt_inv = torch.pow(D + 1e-9, -0.5)
        D1 = torch.diag(D_sqrt_inv)
        D2 = torch.diag(D_sqrt_inv)
        return D1 @ sim_matrix @ D2
    
    def forward(self):
        # Process modality features
        item_v_emb = item_t_emb = None
        if self.v_feat is not None:
            item_v_emb = self.image_encoder(self.image_embedding.weight)
        if self.t_feat is not None:
            item_t_emb = self.text_encoder(self.text_embedding.weight)
            
        # Multimodal feature fusion
        if self.v_feat is not None and self.t_feat is not None:
            item_mm_emb = self.modal_attention(item_v_emb, item_t_emb)
        else:
            item_mm_emb = item_v_emb if item_v_emb is not None else item_t_emb
            
        # Graph convolution on user-item graph
        ego_embeddings = torch.cat((self.user_embedding.weight, self.item_embedding.weight), dim=0)
        all_embeddings = []
        
        for layer in range(self.n_ui_layers):
            # Message passing
            side_embeddings = torch.sparse.mm(self.norm_adj, ego_embeddings)
            
            # Residual connection and normalization
            ego_embeddings = F.normalize(side_embeddings + ego_embeddings, p=2, dim=1)
            
            # Dropout for training
            if self.training:
                ego_embeddings = F.dropout(ego_embeddings, p=self.dropout)
                
            all_embeddings.append(ego_embeddings)
            
        # Multi-scale fusion
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        
        # Process item embeddings through modal graph
        item_emb = item_mm_emb + i_g_embeddings
        for _ in range(self.n_mm_layers):
            item_emb = F.normalize(torch.mm(self.mm_adj, item_emb), p=2, dim=1)
            
        return u_g_embeddings, item_emb
        
    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        
        u_embeddings, i_embeddings = self.forward()
        
        # Positive and negative item embeddings
        u_e = u_embeddings[users]
        pos_e = i_embeddings[pos_items]
        neg_e = i_embeddings[neg_items]
        
        # BPR loss with gradient reversal
        pos_scores = torch.sum(u_e * pos_e, dim=1)
        neg_scores = torch.sum(u_e * neg_e, dim=1)
        mf_loss = self.bpr_loss(pos_scores, neg_scores)
        
        # Diversity-promoting regularization
        batch_size = users.shape[0]
        pos_sim = torch.matmul(pos_e, pos_e.t()) / self.embedding_dim
        identity = torch.eye(batch_size, device=self.device)
        diversity_loss = torch.mean(torch.abs(pos_sim - identity))
        
        # L2 regularization
        reg_loss = self.reg_weight * self.emb_loss(u_e, pos_e, neg_e)
        
        # Total loss with gradient reversal for better convergence
        loss = mf_loss + reg_loss + self.grl_lambda * diversity_loss
        
        return loss
        
    def full_sort_predict(self, interaction):
        user = interaction[0]
        
        u_embeddings, i_embeddings = self.forward()
        u_embeddings = u_embeddings[user]
        
        scores = torch.matmul(u_embeddings, i_embeddings.t())
        return scores

class ModalityEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
    def forward(self, x):
        return self.net(x)

class MultiModalAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, 2),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x1, x2):
        combined = torch.cat([x1, x2], dim=1)
        weights = self.attention(combined)
        return weights[:, 0:1] * x1 + weights[:, 1:2] * x2