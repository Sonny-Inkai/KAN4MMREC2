# coding: utf-8
import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from common.abstract_recommender import GeneralRecommender

class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-12):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = (x - mean).pow(2).mean(-1, keepdim=True)
        norm = (x - mean) / (var + self.eps).sqrt()
        return norm * self.weight + self.bias

class ModalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.norm1 = LayerNorm(hidden_dim)
        self.norm2 = LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.norm1(x)
        x = self.dropout(self.gelu(x))
        x = self.fc2(x)
        x = self.norm2(x)
        return F.normalize(x, p=2, dim=1)

class GraphAttentionLayer(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.1):
        super().__init__()
        assert dim % heads == 0
        self.gat = GATConv(dim, dim // heads, heads=heads, dropout=dropout)
        self.norm = LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index):
        return x + self.dropout(self.norm(self.gat(x, edge_index)))

class MMGAT(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MMGAT, self).__init__(config, dataset)
        
        self.embedding_dim = config["embedding_size"]
        self.feat_embed_dim = config["feat_embed_dim"]
        self.hidden_dim = self.feat_embed_dim * 2
        self.n_layers = config["n_mm_layers"]
        self.dropout = config["dropout"]
        self.reg_weight = config["reg_weight"]
        self.knn_k = config["knn_k"]
        self.temp = 0.2
        
        # User-Item embeddings
        self.user_embedding = nn.Parameter(torch.zeros(self.n_users, self.embedding_dim))
        self.item_embedding = nn.Parameter(torch.zeros(self.n_items, self.embedding_dim))
        nn.init.xavier_uniform_(self.user_embedding)
        nn.init.xavier_uniform_(self.item_embedding)
        
        # Modal encoders
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_encoder = ModalEncoder(self.v_feat.shape[1], self.hidden_dim, self.feat_embed_dim, self.dropout)
            
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_encoder = ModalEncoder(self.t_feat.shape[1], self.hidden_dim, self.feat_embed_dim, self.dropout)
        
        # Graph layers
        self.mm_layers = nn.ModuleList([
            GraphAttentionLayer(self.feat_embed_dim, heads=4, dropout=self.dropout)
            for _ in range(self.n_layers)
        ])
        
        self.ui_layers = nn.ModuleList([
            GraphAttentionLayer(self.embedding_dim, heads=4, dropout=self.dropout)
            for _ in range(self.n_layers)
        ])
        
        # Modal fusion with gating
        if self.v_feat is not None and self.t_feat is not None:
            self.fusion = nn.Sequential(
                nn.Linear(self.feat_embed_dim * 2, self.feat_embed_dim * 2),
                LayerNorm(self.feat_embed_dim * 2),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.feat_embed_dim * 2, self.feat_embed_dim * 2)
            )
            self.gate = nn.Linear(self.feat_embed_dim * 2, self.feat_embed_dim)
        
        # Load data
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.edge_index = self.build_edges()
        self.mm_edge_index = None
        self.build_modal_graph()
        
        self.to(self.device)

    def build_edges(self):
        rows = torch.from_numpy(self.interaction_matrix.row)
        cols = torch.from_numpy(self.interaction_matrix.col) + self.n_users
        
        edge_index = torch.stack([
            torch.cat([rows, cols]),
            torch.cat([cols, rows])
        ]).to(self.device)
        
        # Compute normalized edge weights
        edge_weight = self._compute_normalized_weights(edge_index)
        self.edge_weight = edge_weight.to(self.device)
        
        return edge_index

    def _compute_normalized_weights(self, edge_index):
        num_nodes = self.n_users + self.n_items
        row, col = edge_index
        deg = torch.bincount(row, minlength=num_nodes)
        deg_inv_sqrt = torch.pow(deg + 1e-12, -0.5)
        
        # Symmetric normalization
        weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        return weight

    def build_modal_graph(self):
        if self.v_feat is None and self.t_feat is None:
            return
            
        if self.v_feat is not None:
            feats = F.normalize(self.v_feat, p=2, dim=1)
            sim = torch.mm(feats, feats.t())
            
        if self.t_feat is not None:
            feats = F.normalize(self.t_feat, p=2, dim=1)
            sim = torch.mm(feats, feats.t())
        
        # KNN graph construction with temperature scaling
        sim = sim / self.temp
        values, indices = sim.topk(k=self.knn_k, dim=1)
        rows = torch.arange(feats.size(0), device=self.device).view(-1, 1).expand_as(indices)
        
        # Compute similarity-based weights
        weights = F.softmax(values, dim=1).reshape(-1)
        
        self.mm_edge_index = torch.stack([
            torch.cat([rows.reshape(-1), indices.reshape(-1)]),
            torch.cat([indices.reshape(-1), rows.reshape(-1)])
        ]).to(self.device)
        
        # Symmetric normalization for modal graph
        self.mm_weights = torch.cat([weights, weights]).to(self.device)

    def forward(self):
        # Process modalities with residual connections
        img_emb = txt_emb = None
        
        if self.v_feat is not None:
            img_emb = self.image_encoder(self.image_embedding.weight)
            img_res = img_emb
            for layer in self.mm_layers:
                img_emb = layer(img_emb, self.mm_edge_index)
            img_emb = img_emb + img_res
                
        if self.t_feat is not None:
            txt_emb = self.text_encoder(self.text_embedding.weight)
            txt_res = txt_emb
            for layer in self.mm_layers:
                txt_emb = layer(txt_emb, self.mm_edge_index)
            txt_emb = txt_emb + txt_res
        
        # Adaptive modal fusion with gating
        if img_emb is not None and txt_emb is not None:
            concat = torch.cat([img_emb, txt_emb], dim=1)
            gates = torch.sigmoid(self.gate(self.fusion(concat)))
            modal_emb = gates * img_emb + (1 - gates) * txt_emb
        else:
            modal_emb = img_emb if img_emb is not None else txt_emb
        
        # Process user-item graph with skip connections
        x = torch.cat([self.user_embedding, self.item_embedding])
        x_list = [x]
        
        for i, layer in enumerate(self.ui_layers):
            if self.training:
                x = F.dropout(x, p=self.dropout, training=True)
            x = layer(x, self.edge_index)
            x = x + x_list[0] if i == self.n_layers - 1 else x
            x_list.append(x)
            
        x = torch.stack(x_list, dim=1)
        x = torch.mean(x, dim=1)
        
        user_emb, item_emb = torch.split(x, [self.n_users, self.n_items])
        
        # Combine with modal embeddings using residual connection
        if modal_emb is not None:
            item_emb = item_emb + F.dropout(modal_emb, p=self.dropout, training=self.training)
            
        return user_emb, item_emb, img_emb, txt_emb

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        
        user_emb, item_emb, img_emb, txt_emb = self.forward()
        
        u_emb = user_emb[users]
        pos_emb = item_emb[pos_items]
        neg_emb = item_emb[neg_items]
        
        # InfoNCE loss for user-item interactions
        pos_scores = (u_emb * pos_emb).sum(dim=1)
        neg_scores = (u_emb * neg_emb).sum(dim=1)
        
        pos_scores = pos_scores / self.temp
        neg_scores = neg_scores / self.temp
        
        mf_loss = -torch.mean(pos_scores - torch.logsumexp(torch.stack([pos_scores, neg_scores], dim=1), dim=1))
        
        # Modal alignment loss with temperature scaling
        modal_loss = 0.0
        if img_emb is not None and txt_emb is not None:
            img_pos = img_emb[pos_items]
            txt_pos = txt_emb[pos_items]
            img_neg = img_emb[neg_items]
            txt_neg = txt_emb[neg_items]
            
            pos_modal = (img_pos * txt_pos).sum(dim=1) / self.temp
            neg_modal = (img_pos * txt_neg).sum(dim=1) / self.temp
            
            modal_loss = -torch.mean(pos_modal - torch.logsumexp(torch.stack([pos_modal, neg_modal], dim=1), dim=1))
        
        # L2 regularization with weight decay
        l2_loss = self.reg_weight * (
            torch.norm(u_emb, p=2) +
            torch.norm(pos_emb, p=2) +
            torch.norm(neg_emb, p=2)
        )
        
        # Combined loss with dynamic weighting
        total_loss = mf_loss + 0.1 * modal_loss + l2_loss
        return total_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_emb, item_emb, _, _ = self.forward()
        
        # Compute scores with temperature scaling
        scores = torch.matmul(user_emb[user], item_emb.transpose(0, 1)) / self.temp
        return scores