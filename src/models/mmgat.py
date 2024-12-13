# coding: utf-8
import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from common.abstract_recommender import GeneralRecommender

class AdaptiveWeight(nn.Module):
    def __init__(self, init_value=0.5):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(init_value))
        
    def forward(self):
        return torch.sigmoid(self.weight)

class ResidualGraphLayer(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.1):
        super().__init__()
        self.gat = GATConv(dim, dim // heads, heads=heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim)
        )
        
    def forward(self, x, edge_index):
        # Multi-head attention with residual
        res = x
        x = self.gat(x, edge_index)
        x = self.norm1(x + res)
        
        # Feed-forward network with residual
        res = x
        x = self.ffn(x)
        x = self.norm2(x + res)
        return x

class FeatureEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        self.norm = nn.LayerNorm(output_dim)
        
    def forward(self, x):
        h = self.project(x)
        h = self.norm(h)
        return h

class MMGAT(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MMGAT, self).__init__(config, dataset)
        
        # Basic settings
        self.embedding_dim = config["embedding_size"]
        self.feat_embed_dim = config["feat_embed_dim"]
        self.hidden_dim = self.feat_embed_dim * 2
        self.n_layers = config["n_mm_layers"]
        self.knn_k = config["knn_k"]
        self.reg_weight = config["reg_weight"]
        self.dropout = config["dropout"]
        
        # Initialize embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        
        # Initialize with smaller values to prevent saturation
        nn.init.normal_(self.user_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.02)
        
        # Modal encoders
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_encoder = FeatureEncoder(
                self.v_feat.shape[1], self.hidden_dim, self.feat_embed_dim, self.dropout
            )
            
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_encoder = FeatureEncoder(
                self.t_feat.shape[1], self.hidden_dim, self.feat_embed_dim, self.dropout
            )
        
        # Graph layers with residual connections
        self.ui_layers = nn.ModuleList([
            ResidualGraphLayer(self.embedding_dim, dropout=self.dropout)
            for _ in range(self.n_layers)
        ])
        
        self.modal_layers = nn.ModuleList([
            ResidualGraphLayer(self.feat_embed_dim, dropout=self.dropout)
            for _ in range(self.n_layers)
        ])
        
        # Adaptive weights
        self.modal_weight = AdaptiveWeight(0.001)
        self.reg_adaptation = AdaptiveWeight(0.001)
        
        # Initialize graphs
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.edge_index = self.build_edges()
        self.modal_edge_index = self.build_modal_graph()
        
        self.to(self.device)

    def build_edges(self):
        rows = self.interaction_matrix.row
        cols = self.interaction_matrix.col + self.n_users
        
        edge_index = torch.tensor(np.vstack([
            np.concatenate([rows, cols]),
            np.concatenate([cols, rows])
        ]), dtype=torch.long).to(self.device)
        
        return edge_index

    def build_modal_graph(self):
        if not (self.v_feat is not None or self.t_feat is not None):
            return None
            
        # Build KNN graph based on available modality
        features = self.v_feat if self.v_feat is not None else self.t_feat
        features = F.normalize(features, p=2, dim=1)
        
        sim = torch.mm(features, features.t())
        values, indices = sim.topk(k=self.knn_k, dim=1)
        rows = torch.arange(features.size(0), device=self.device).view(-1, 1).expand_as(indices)
        
        edge_index = torch.stack([
            torch.cat([rows.reshape(-1), indices.reshape(-1)]),
            torch.cat([indices.reshape(-1), rows.reshape(-1)])
        ]).to(self.device)
        
        return edge_index

    def process_modalities(self):
        # Process available modalities separately
        img_emb = txt_emb = None
        
        if self.v_feat is not None:
            img_emb = self.image_encoder(self.image_embedding.weight)
            for layer in self.modal_layers:
                if self.modal_edge_index is not None:
                    img_emb = layer(img_emb, self.modal_edge_index)
                    
        if self.t_feat is not None:
            txt_emb = self.text_encoder(self.text_embedding.weight)
            for layer in self.modal_layers:
                if self.modal_edge_index is not None:
                    txt_emb = layer(txt_emb, self.modal_edge_index)
        
        # Combine modalities with learned weights if both are present
        if img_emb is not None and txt_emb is not None:
            w = self.modal_weight()
            modal_emb = w * img_emb + (1 - w) * txt_emb
        else:
            modal_emb = img_emb if img_emb is not None else txt_emb
            
        return modal_emb, (img_emb, txt_emb)

    def forward(self):
        # Process modalities
        modal_emb, (img_emb, txt_emb) = self.process_modalities()
        
        # Graph convolution with residual connections
        x = torch.cat([self.user_embedding.weight, self.item_embedding.weight])
        residuals = []
        
        for i, layer in enumerate(self.ui_layers):
            if self.training:
                x = F.dropout(x, p=self.dropout)
            x = layer(x, self.edge_index)
            residuals.append(x)
            
        # Weighted residual aggregation
        x = torch.stack(residuals, dim=0).mean(dim=0)
        user_emb, item_emb = torch.split(x, [self.n_users, self.n_items])
        
        # Combine with modal features
        if modal_emb is not None:
            item_emb = item_emb + modal_emb
            
        return user_emb, item_emb, (img_emb, txt_emb)

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        
        user_emb, item_emb, (img_emb, txt_emb) = self.forward()
        
        u_emb = user_emb[users]
        pos_emb = item_emb[pos_items]
        neg_emb = item_emb[neg_items]
        
        # Main recommendation loss with margin
        pos_scores = torch.sum(u_emb * pos_emb, dim=1)
        neg_scores = torch.sum(u_emb * neg_emb, dim=1)
        margin = 1.0
        rec_loss = torch.mean(torch.relu(margin - (pos_scores - neg_scores)))
        
        # Modal contrastive loss with temperature scaling
        modal_loss = 0.0
        if img_emb is not None and txt_emb is not None:
            temp = 0.07  # Temperature parameter
            i_emb = F.normalize(img_emb[pos_items], dim=1)
            t_emb = F.normalize(txt_emb[pos_items], dim=1)
            
            sim_matrix = torch.matmul(i_emb, t_emb.t()) / temp
            labels = torch.arange(len(pos_items)).to(self.device)
            modal_loss = F.cross_entropy(sim_matrix, labels)
        
        # Adaptive regularization
        reg_weight = self.reg_adaptation()
        reg_loss = reg_weight * (
            torch.norm(u_emb) +
            torch.norm(pos_emb) +
            torch.norm(neg_emb)
        )
        
        # Combined loss with learned weights
        modal_weight = self.modal_weight()
        total_loss = rec_loss + modal_weight * modal_loss + reg_loss
        return total_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_emb, item_emb, _ = self.forward()
        scores = torch.matmul(user_emb[user], item_emb.transpose(0, 1))
        return scores