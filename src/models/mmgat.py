# coding: utf-8
import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv
from common.abstract_recommender import GeneralRecommender

class ModalGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.gat = GATConv(hidden_dim, output_dim, heads=1, dropout=dropout)
        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index):
        x = self.proj(x)
        x = self.dropout(F.gelu(x))
        x = self.gat(x, edge_index)
        return self.norm(x)

class MMGAT(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MMGAT, self).__init__(config, dataset)
        
        self.embedding_dim = config["embedding_size"]
        self.feat_embed_dim = config["feat_embed_dim"]
        self.hidden_dim = self.feat_embed_dim * 2
        self.n_layers = config["n_mm_layers"]
        self.dropout = config["dropout"]
        self.knn_k = config["knn_k"]
        self.reg_weight = config["reg_weight"]
        
        # Basic embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_normal_(self.user_embedding.weight)
        nn.init.xavier_normal_(self.item_embedding.weight)
        
        # Modal networks
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_gnn = ModalGNN(self.v_feat.shape[1], self.hidden_dim, self.feat_embed_dim, self.dropout)
            self.build_image_graph()
            
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_gnn = ModalGNN(self.t_feat.shape[1], self.hidden_dim, self.feat_embed_dim, self.dropout)
            self.build_text_graph()
        
        # Interaction graph
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.edge_index = self.build_ui_graph()
        
        # Graph layers
        self.gnn_layers = nn.ModuleList([
            GCNConv(self.embedding_dim, self.embedding_dim)
            for _ in range(self.n_layers)
        ])
        
        # Modal fusion
        if self.v_feat is not None and self.t_feat is not None:
            self.fusion = nn.Sequential(
                nn.Linear(self.feat_embed_dim * 2, self.feat_embed_dim),
                nn.LayerNorm(self.feat_embed_dim),
                nn.GELU()
            )
        
        self.to(self.device)
        
    def build_ui_graph(self):
        rows = torch.from_numpy(self.interaction_matrix.row)
        cols = torch.from_numpy(self.interaction_matrix.col + self.n_users)
        edge_index = torch.stack([
            torch.cat([rows, cols]),
            torch.cat([cols, rows])
        ]).to(self.device)
        return edge_index
        
    def build_knn_graph(self, features):
        sim = torch.mm(F.normalize(features, dim=1), F.normalize(features, dim=1).t())
        _, indices = torch.topk(sim, k=self.knn_k, dim=1)
        rows = torch.arange(features.size(0), device=features.device).view(-1, 1).expand_as(indices)
        edge_index = torch.stack([rows.reshape(-1), indices.reshape(-1)])
        return edge_index
        
    def build_image_graph(self):
        self.image_edge_index = self.build_knn_graph(self.image_embedding.weight).to(self.device)
        
    def build_text_graph(self):
        self.text_edge_index = self.build_knn_graph(self.text_embedding.weight).to(self.device)

    def forward(self):
        # Process modalities
        modal_feat = None
        if self.v_feat is not None and self.t_feat is not None:
            img_feat = self.image_gnn(self.image_embedding.weight, self.image_edge_index)
            txt_feat = self.text_gnn(self.text_embedding.weight, self.text_edge_index)
            modal_feat = self.fusion(torch.cat([img_feat, txt_feat], dim=1))
        elif self.v_feat is not None:
            modal_feat = self.image_gnn(self.image_embedding.weight, self.image_edge_index)
        elif self.t_feat is not None:
            modal_feat = self.text_gnn(self.text_embedding.weight, self.text_edge_index)
        
        # Process user-item graph
        x = torch.cat([self.user_embedding.weight, self.item_embedding.weight])
        h = x
        
        for layer in self.gnn_layers:
            h = layer(h, self.edge_index)
            h = F.dropout(h, p=self.dropout, training=self.training)
            
        user_emb, item_emb = torch.split(h, [self.n_users, self.n_items])
        
        if modal_feat is not None:
            item_emb = item_emb + modal_feat
            
        return user_emb, item_emb
        
    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        
        user_emb, item_emb = self.forward()
        
        user_e = user_emb[users]
        pos_e = item_emb[pos_items]
        neg_e = item_emb[neg_items]
        
        # BPR loss
        pos_scores = torch.sum(user_e * pos_e, dim=1)
        neg_scores = torch.sum(user_e * neg_e, dim=1)
        rec_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        
        # L2 regularization
        reg_loss = self.reg_weight * (
            torch.norm(user_e) +
            torch.norm(pos_e) +
            torch.norm(neg_e)
        )
        
        return rec_loss + reg_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_emb, item_emb = self.forward()
        scores = torch.matmul(user_emb[user], item_emb.transpose(0, 1))
        return scores