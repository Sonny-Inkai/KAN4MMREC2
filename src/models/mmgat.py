# coding: utf-8
import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch.optim.lr_scheduler import CosineAnnealingLR
from common.abstract_recommender import GeneralRecommender

class ModalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
    def forward(self, x):
        return F.normalize(self.encoder(x), p=2, dim=1)

class OptimizedGraphAttentionLayer(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.gat = GATConv(dim, dim // 4, heads=4, dropout=dropout)
        self.residual = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index):
        h = self.gat(x, edge_index)
        h = self.dropout(h)
        return self.norm(h + self.residual(x))  # Add residual connection

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
        
        # User-Item embeddings
        self.user_embedding = nn.Embedding(dataset.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(dataset.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        
        # Modal encoders
        if dataset.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(dataset.v_feat, freeze=False)
            self.image_encoder = ModalEncoder(dataset.v_feat.shape[1], self.hidden_dim, self.feat_embed_dim, self.dropout)
            
        if dataset.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(dataset.t_feat, freeze=False)
            self.text_encoder = ModalEncoder(dataset.t_feat.shape[1], self.hidden_dim, self.feat_embed_dim, self.dropout)
            
        # Graph layers
        self.ui_layers = nn.ModuleList([
            OptimizedGraphAttentionLayer(self.embedding_dim, self.dropout)
            for _ in range(self.n_layers)
        ])
        
        # Modal fusion
        if dataset.v_feat is not None and dataset.t_feat is not None:
            self.fusion = nn.Sequential(
                nn.Linear(self.feat_embed_dim * 2, self.feat_embed_dim),
                nn.LayerNorm(self.feat_embed_dim),
                nn.GELU(),
                nn.Dropout(self.dropout)
            )
        
        # Load data
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.edge_index = self.build_edges(dataset.n_users, dataset.n_items)
        self.mm_edge_index = None
        self.build_modal_graph(dataset)
        
        self.to(config["device"])

    def build_edges(self, n_users, n_items):
        rows = self.interaction_matrix.row
        cols = self.interaction_matrix.col + n_users
        
        edge_index = torch.tensor(np.vstack([
            np.concatenate([rows, cols]),
            np.concatenate([cols, rows])
        ]), dtype=torch.long).to(self.device)
        
        return edge_index

    def build_modal_graph(self, dataset):
        if dataset.v_feat is None and dataset.t_feat is None:
            return
            
        if dataset.v_feat is not None:
            feats = F.normalize(dataset.v_feat, p=2, dim=1)
            sim = torch.mm(feats, feats.t())
            
        if dataset.t_feat is not None:
            feats = F.normalize(dataset.t_feat, p=2, dim=1)
            sim = torch.mm(feats, feats.t())
        
        values, indices = sim.topk(k=self.knn_k, dim=1)
        rows = torch.arange(feats.size(0), device=self.device).view(-1, 1).expand_as(indices)
        
        self.mm_edge_index = torch.stack([
            torch.cat([rows.reshape(-1), indices.reshape(-1)]),
            torch.cat([indices.reshape(-1), rows.reshape(-1)])
        ]).to(self.device)

    def forward(self):
        # Process user-item graph
        x = torch.cat([self.user_embedding.weight, self.item_embedding.weight])
        for layer in self.ui_layers:
            x = layer(x, self.edge_index)
            
        user_emb, item_emb = torch.split(x, [self.user_embedding.num_embeddings, self.item_embedding.num_embeddings])
        return user_emb, item_emb

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        
        user_emb, item_emb = self.forward()
        
        u_emb = user_emb[users]
        pos_emb = item_emb[pos_items]
        neg_emb = item_emb[neg_items]
        
        # BPR loss
        pos_scores = torch.sum(u_emb * pos_emb, dim=1)
        neg_scores = torch.sum(u_emb * neg_emb, dim=1)
        mf_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        
        # Regularization loss
        reg_loss = self.reg_weight * (
            torch.norm(u_emb) +
            torch.norm(pos_emb) +
            torch.norm(neg_emb)
        )
        
        return mf_loss + reg_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_emb, item_emb = self.forward()
        scores = torch.matmul(user_emb[user], item_emb.transpose(0, 1))
        return scores