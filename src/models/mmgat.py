# coding: utf-8
import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from common.abstract_recommender import GeneralRecommender

class ModalProjection(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        return self.net(x)

class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.gat = GATConv(in_dim, out_dim // 4, heads=4, dropout=0.1)
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.PReLU()
        
    def forward(self, x, edge_index):
        x = self.gat(x, edge_index)
        x = self.norm(x)
        return self.act(x)

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
        
        # Embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        
        # Modal networks
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_proj = ModalProjection(
                self.v_feat.shape[1], 
                self.hidden_dim, 
                self.embedding_dim, 
                self.dropout
            )
            
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_proj = ModalProjection(
                self.t_feat.shape[1], 
                self.hidden_dim, 
                self.embedding_dim, 
                self.dropout
            )
            
        # Graph layers
        self.gnn_layers = nn.ModuleList([
            GATLayer(self.embedding_dim, self.embedding_dim) 
            for _ in range(self.n_layers)
        ])
        
        # Modal fusion
        if self.v_feat is not None and self.t_feat is not None:
            self.fusion = nn.Sequential(
                nn.Linear(self.embedding_dim * 2, self.embedding_dim),
                nn.LayerNorm(self.embedding_dim),
                nn.PReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.embedding_dim, self.embedding_dim),
                nn.LayerNorm(self.embedding_dim)
            )
        
        # Build graphs
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.edge_index = None
        self.modal_edge = None
        self.build_graph()
        
        self.to(self.device)
        
    def build_graph(self):
        # Build user-item interaction graph
        rows = self.interaction_matrix.row
        cols = self.interaction_matrix.col + self.n_users
        
        edge_index = torch.tensor(np.vstack([
            np.concatenate([rows, cols]),
            np.concatenate([cols, rows])
        ]), dtype=torch.long)
        
        self.edge_index = edge_index.to(self.device)
        
        # Build modal graph if needed
        if self.v_feat is not None or self.t_feat is not None:
            features = []
            if self.v_feat is not None:
                v_feat = F.normalize(self.v_feat, p=2, dim=1)
                features.append(v_feat)
            if self.t_feat is not None:
                t_feat = F.normalize(self.t_feat, p=2, dim=1)
                features.append(t_feat)
            
            feature = sum(features) / len(features)
            sim = torch.mm(feature, feature.t())
            
            values, indices = torch.topk(sim, k=self.knn_k, dim=1)
            rows = torch.arange(feature.size(0)).view(-1, 1).expand_as(indices)
            self.modal_edge = torch.stack([
                torch.cat([rows.reshape(-1), indices.reshape(-1)]),
                torch.cat([indices.reshape(-1), rows.reshape(-1)])
            ]).to(self.device)

    def message_dropout(self, x):
        if self.training:
            mask = torch.bernoulli(torch.ones_like(x) * (1 - self.dropout))
            x = x * mask
            x = x / (1 - self.dropout)
        return x

    def forward(self):
        # Process modalities
        modal_emb = None
        if self.v_feat is not None and self.t_feat is not None:
            v_emb = self.image_proj(self.image_embedding.weight)
            t_emb = self.text_proj(self.text_embedding.weight)
            
            # Apply graph convolution on modal features
            if self.modal_edge is not None:
                for layer in self.gnn_layers:
                    v_emb = layer(v_emb, self.modal_edge)
                    t_emb = layer(t_emb, self.modal_edge)
                
            # Fuse modalities
            modal_emb = self.fusion(torch.cat([v_emb, t_emb], dim=1))
            
        elif self.v_feat is not None:
            modal_emb = self.image_proj(self.image_embedding.weight)
            if self.modal_edge is not None:
                for layer in self.gnn_layers:
                    modal_emb = layer(modal_emb, self.modal_edge)
                    
        elif self.t_feat is not None:
            modal_emb = self.text_proj(self.text_embedding.weight)
            if self.modal_edge is not None:
                for layer in self.gnn_layers:
                    modal_emb = layer(modal_emb, self.modal_edge)
        
        # Process user-item graph
        x = torch.cat([self.user_embedding.weight, self.item_embedding.weight])
        embs = [x]
        
        for layer in self.gnn_layers:
            x = self.message_dropout(x)
            x = layer(x, self.edge_index)
            embs.append(x)
            
        x = torch.stack(embs, dim=1).mean(dim=1)
        user_emb, item_emb = torch.split(x, [self.n_users, self.n_items])
        
        # Combine with modal embeddings
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