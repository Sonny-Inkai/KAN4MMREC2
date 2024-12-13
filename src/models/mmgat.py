import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from common.abstract_recommender import GeneralRecommender

class LaplacianNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x, edge_index):
        row, col = edge_index
        deg = torch.bincount(row).float().clamp(min=1)
        deg_inv_sqrt = deg.pow(-0.5)
        
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        x = x - x.mean(dim=0, keepdim=True)
        x = x / (x.std(dim=0, keepdim=True) + self.eps)
        return x * self.scale + self.bias

class StableGATLayer(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.dim = dim
        self.head_dim = dim // heads
        
        self.W = nn.Linear(dim, dim)
        self.a = nn.Parameter(torch.zeros(heads, 2 * self.head_dim))
        nn.init.xavier_uniform_(self.a)
        
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(dim)

    def forward(self, x, edge_index):
        N = x.size(0)
        h = self.dropout(self.W(x))
        h = h.view(N, self.heads, self.head_dim)
        
        edge_h = torch.cat([h[edge_index[0]], h[edge_index[1]]], dim=-1)
        alpha = self.leakyrelu((self.a * edge_h).sum(-1))
        alpha = F.softmax(alpha, dim=1)
        alpha = self.dropout(alpha)
        
        out = torch.zeros_like(h)
        for i in range(self.heads):
            out[:, i] = self._aggregate(h[:, i], edge_index, alpha[:, i])
        
        out = out.view(N, -1)
        out = self.layernorm(out + x)
        return F.normalize(out, p=2, dim=-1)
    
    def _aggregate(self, h, edge_index, alpha):
        out = torch.zeros_like(h)
        out.index_add_(0, edge_index[0], alpha.unsqueeze(-1) * h[edge_index[1]])
        return out

class MMGAT(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MMGAT, self).__init__(config, dataset)
        
        self.embedding_dim = config["embedding_size"]
        self.feat_embed_dim = config["feat_embed_dim"]
        self.n_layers = config["n_mm_layers"]
        self.n_ui_layers = config["n_ui_layers"]
        self.dropout = config["dropout"]
        self.reg_weight = config["reg_weight"]
        self.knn_k = config["knn_k"]
        
        # User-Item embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        
        nn.init.xavier_normal_(self.user_embedding.weight, gain=1.414)
        nn.init.xavier_normal_(self.item_embedding.weight, gain=1.414)
        
        # Load and process features
        if self.v_feat is not None:
            self.v_feat = F.normalize(self.v_feat, p=2, dim=1)
            self.image_proj = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)
            
        if self.t_feat is not None:
            self.t_feat = F.normalize(self.t_feat, p=2, dim=1)
            self.text_proj = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)
            
        # Graph layers
        self.gat_layers = nn.ModuleList([
            StableGATLayer(self.embedding_dim)
            for _ in range(self.n_layers)
        ])
        
        # Modal fusion
        if self.v_feat is not None and self.t_feat is not None:
            self.fusion = nn.Sequential(
                nn.Linear(self.feat_embed_dim * 2, self.feat_embed_dim),
                nn.LayerNorm(self.feat_embed_dim),
                nn.GELU(),
                nn.Dropout(self.dropout)
            )
            self.modal_weight = nn.Parameter(torch.ones(2))
        
        # Load interaction data
        self.edge_index = self.build_edges()
        self.mm_edge_index = None
        self.build_modal_graph()
        
        self.to(self.device)

    def build_edges(self):
        rows = torch.from_numpy(self.interaction_matrix.row)
        cols = torch.from_numpy(self.interaction_matrix.col)
        cols = cols + self.n_users
        
        edge_index = torch.stack([
            torch.cat([rows, cols]),
            torch.cat([cols, rows])
        ]).to(self.device)
        
        return edge_index

    def compute_similarity_batched(self, features, batch_size):
        n_items = features.size(0)
        edge_index_list = []
        edge_weight_list = []
        
        for i in range(0, n_items, batch_size):
            batch_features = features[i:i+batch_size]
            sim = torch.mm(batch_features, features.t())
            topk_values, topk_indices = sim.topk(k=self.knn_k, dim=1)
            
            rows = torch.arange(i, min(i+batch_size, n_items), device=self.device)
            rows = rows.view(-1, 1).expand_as(topk_indices)
            
            edge_index_list.append(torch.stack([rows.reshape(-1), topk_indices.reshape(-1)]))
            edge_weight_list.append(topk_values.reshape(-1))
            
            del sim
            torch.cuda.empty_cache()
        
        edge_index = torch.cat(edge_index_list, dim=1)
        edge_weight = torch.cat(edge_weight_list)
        
        return edge_index, edge_weight

    def build_modal_graph(self):
        if not (self.v_feat is not None or self.t_feat is not None):
            return
        
        if self.v_feat is not None and self.t_feat is not None:
            v_feat = self.image_proj(self.v_feat)
            t_feat = self.text_proj(self.t_feat)
            weights = F.softmax(self.modal_weight, dim=0)
            features = weights[0] * v_feat + weights[1] * t_feat
        else:
            features = self.v_feat if self.v_feat is not None else self.t_feat
            features = F.normalize(features, p=2, dim=1)
        
        edge_index, edge_weight = self.compute_similarity_batched(features, self.batch_size)
        self.mm_edge_index = edge_index
        self.mm_edge_weight = edge_weight

    def forward(self):
        # Process modalities
        modal_emb = None
        
        if self.v_feat is not None or self.t_feat is not None:
            if self.v_feat is not None and self.t_feat is not None:
                v_emb = self.image_proj(self.v_feat)
                t_emb = self.text_proj(self.t_feat)
                weights = F.softmax(self.modal_weight, dim=0)
                modal_emb = self.fusion(torch.cat([v_emb, t_emb], dim=1))
            else:
                features = self.v_feat if self.v_feat is not None else self.t_feat
                modal_emb = F.normalize(features, p=2, dim=1)
        
        # Process user-item interactions
        x = torch.cat([self.user_embedding.weight, self.item_embedding.weight])
        h = x
        
        for layer in self.gat_layers:
            h = layer(h, self.edge_index)
        
        user_emb, item_emb = torch.split(h, [self.n_users, self.n_items])
        
        if modal_emb is not None:
            item_emb = item_emb + F.normalize(modal_emb, p=2, dim=1)
        
        return user_emb, item_emb

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        
        user_emb, item_emb = self.forward()
        
        u_emb = user_emb[users]
        pos_emb = item_emb[pos_items]
        neg_emb = item_emb[neg_items]
        
        # BPR loss with temperature scaling
        pos_scores = torch.sum(u_emb * pos_emb, dim=1)
        neg_scores = torch.sum(u_emb * neg_emb, dim=1)
        
        bpr_loss = -F.logsigmoid(pos_scores - neg_scores).mean()
        
        # L2 regularization
        l2_loss = self.reg_weight * (
            torch.norm(u_emb) + 
            torch.norm(pos_emb) + 
            torch.norm(neg_emb)
        )
        
        return bpr_loss + l2_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_emb, item_emb = self.forward()
        scores = torch.matmul(user_emb[user], item_emb.transpose(0, 1))
        return scores