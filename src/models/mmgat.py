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
        self.cl_weight = 0.1
        self.temperature = 0.2
        self.lambda_coeff = 0.5
        
        self.n_nodes = self.n_users + self.n_items
        self.norm_adj = None
        self.mm_adj = None
        self.edge_index = None
        self.edge_values = None
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)

        self._init_graph()
        self._init_modules()
        
    def _init_modules(self):
        # Basic embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim).to(self.device)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim).to(self.device)
        
        # Feature transformations
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False).to(self.device)
            self.image_encoder = ImageEncoder(self.v_feat.shape[1], self.feat_embed_dim).to(self.device)
            
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False).to(self.device)
            self.text_encoder = TextEncoder(self.t_feat.shape[1], self.feat_embed_dim).to(self.device)
        
        # Fusion modules
        self.modal_fusion = ModalFusion(self.feat_embed_dim, self.dropout).to(self.device)
        self.modal_gate = DynamicGate(self.feat_embed_dim).to(self.device)
        
        # Graph modules
        self.graph_encoder = GraphEncoder(self.feat_embed_dim, self.n_layers).to(self.device)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            nn.init.xavier_normal_(module.weight)
            
    def _init_graph(self):
        # Build interaction graph
        indices = self._build_edges()
        self.edge_index = torch.LongTensor(indices).to(self.device)
        self.edge_values = self._compute_edge_values(indices)
        self.norm_adj = self._build_norm_adj()
        self.mm_adj = self._init_mm_graph()
        
    def _build_edges(self):
        rows = np.concatenate([self.interaction_matrix.row, self.interaction_matrix.col + self.n_users])
        cols = np.concatenate([self.interaction_matrix.col + self.n_users, self.interaction_matrix.row])
        return np.stack([rows, cols])
        
    def _compute_edge_values(self, indices):
        row_sum = np.bincount(indices[0], minlength=self.n_nodes)
        col_sum = np.bincount(indices[1], minlength=self.n_nodes)
        
        row_sqrt = np.sqrt(row_sum[indices[0]]) + 1e-8
        col_sqrt = np.sqrt(col_sum[indices[1]]) + 1e-8
        values = 1 / (row_sqrt * col_sqrt)
        
        return torch.FloatTensor(values).to(self.device)
        
    def _build_norm_adj(self):
        values = self.edge_values
        indices = self.edge_index
        adj = torch.sparse_coo_tensor(
            indices, values,
            size=(self.n_nodes, self.n_nodes),
            device=self.device
        )
        return adj
        
    def _init_mm_graph(self):
        v_adj = t_adj = None
        
        if self.v_feat is not None:
            v_emb = self.image_encoder(self.image_embedding.weight)
            v_adj = self._build_knn_graph(v_emb)
            
        if self.t_feat is not None:
            t_emb = self.text_encoder(self.text_embedding.weight)
            t_adj = self._build_knn_graph(t_emb)
            
        if v_adj is not None and t_adj is not None:
            return self.lambda_coeff * v_adj + (1 - self.lambda_coeff) * t_adj
        return v_adj if v_adj is not None else t_adj
        
    def _build_knn_graph(self, embeddings):
        sim = self._cosine_similarity(embeddings)
        topk_values, topk_indices = torch.topk(sim, self.knn_k, dim=1)
        mask = torch.zeros_like(sim, device=self.device)
        mask.scatter_(1, topk_indices, 1)
        adj = mask * sim
        return self._symmetric_normalize(adj)
        
    def _cosine_similarity(self, embeddings):
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return torch.mm(embeddings, embeddings.t())
        
    def _symmetric_normalize(self, adj):
        rowsum = adj.sum(1)
        d_inv_sqrt = torch.pow(rowsum + 1e-8, -0.5)
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        return d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt
        
    def forward(self):
        # Extract features
        v_features = t_features = None
        if self.v_feat is not None:
            v_features = self.image_encoder(self.image_embedding.weight)
        if self.t_feat is not None:
            t_features = self.text_encoder(self.text_embedding.weight)
            
        # Fuse modalities
        if v_features is not None and t_features is not None:
            alpha = self.modal_gate(v_features, t_features)
            item_features = self.modal_fusion(v_features, t_features, alpha)
        else:
            item_features = v_features if v_features is not None else t_features
            
        # Graph convolution
        item_enhanced = self.graph_encoder(item_features, self.mm_adj)
        
        # User-item interactions
        ego_embeddings = torch.cat([self.user_embedding.weight, self.item_id_embedding.weight], dim=0)
        embeddings = [ego_embeddings]
        
        for _ in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.norm_adj, ego_embeddings)
            embeddings.append(ego_embeddings)
            
        embeddings = torch.stack(embeddings, dim=1)
        embeddings = torch.mean(embeddings, dim=1)
        
        user_embeddings, item_embeddings = torch.split(embeddings, [self.n_users, self.n_items])
        item_embeddings = item_embeddings + F.normalize(item_enhanced, p=2, dim=1)
        
        return user_embeddings, item_embeddings
        
    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        
        user_embeddings, item_embeddings = self.forward()
        
        u_embeddings = user_embeddings[users]
        pos_embeddings = item_embeddings[pos_items]
        neg_embeddings = item_embeddings[neg_items]
        
        # BPR loss
        pos_scores = torch.sum(u_embeddings * pos_embeddings, dim=1)
        neg_scores = torch.sum(u_embeddings * neg_embeddings, dim=1)
        bpr_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        
        # Contrastive loss
        if self.v_feat is not None and self.t_feat is not None:
            v_emb = self.image_encoder(self.image_embedding.weight[pos_items])
            t_emb = self.text_encoder(self.text_embedding.weight[pos_items])
            
            v_emb = F.normalize(v_emb, dim=1)
            t_emb = F.normalize(t_emb, dim=1)
            
            logits = torch.mm(v_emb, t_emb.t()) / self.temperature
            labels = torch.arange(logits.size(0), device=self.device)
            cl_loss = F.cross_entropy(logits, labels)
        else:
            cl_loss = 0.0
            
        # L2 regularization
        l2_loss = self.reg_weight * (
            torch.norm(u_embeddings) +
            torch.norm(pos_embeddings) +
            torch.norm(neg_embeddings)
        )
        
        return bpr_loss + self.cl_weight * cl_loss + l2_loss
        
    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_embeddings, item_embeddings = self.forward()
        u_embeddings = user_embeddings[user]
        scores = torch.matmul(u_embeddings, item_embeddings.t())
        return scores

class ImageEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        hidden_dim = (input_dim + output_dim) // 2
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        return self.net(x)

class TextEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        hidden_dim = (input_dim + output_dim) // 2
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        return self.net(x)

class ModalFusion(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, v_feat, t_feat, alpha):
        combined = torch.cat([v_feat * alpha, t_feat * (1-alpha)], dim=1)
        return self.fusion(combined)

class DynamicGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, v_feat, t_feat):
        combined = torch.cat([v_feat, t_feat], dim=1)
        return self.gate(combined)

class GraphEncoder(nn.Module):
    def __init__(self, dim, n_layers):
        super().__init__()
        self.n_layers = n_layers
        self.gru = nn.GRU(dim, dim, batch_first=True)
        
    def forward(self, x, adj):
        embeddings = [x]
        for _ in range(self.n_layers):
            x = torch.sparse.mm(adj, x)
            embeddings.append(x)
        stacked = torch.stack(embeddings, dim=1)
        output, _ = self.gru(stacked)
        return output[:, -1]