# coding: utf-8
import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from common.abstract_recommender import GeneralRecommender

class ModalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
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

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, heads=4, dropout=0.1, residual=True):
        super().__init__()
        self.gat = GATConv(in_dim, out_dim // heads, heads=heads, dropout=dropout)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.residual = residual
        
    def forward(self, x, edge_index):
        out = self.gat(x, edge_index)
        if self.residual and x.shape == out.shape:
            out = x + self.dropout(out)
        return self.norm(out)

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
        self.temp = 0.2  # Temperature for contrastive learning
        
        # User-Item embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        
        # Modal encoders with larger capacity
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_encoder = nn.Sequential(
                ModalEncoder(self.v_feat.shape[1], self.hidden_dim, self.feat_embed_dim, self.dropout),
                ModalEncoder(self.feat_embed_dim, self.hidden_dim, self.feat_embed_dim, self.dropout)
            )
            
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_encoder = nn.Sequential(
                ModalEncoder(self.t_feat.shape[1], self.hidden_dim, self.feat_embed_dim, self.dropout),
                ModalEncoder(self.feat_embed_dim, self.hidden_dim, self.feat_embed_dim, self.dropout)
            )
            
        # Multi-head graph attention layers
        self.mm_layers = nn.ModuleList([
            GraphAttentionLayer(self.feat_embed_dim, self.feat_embed_dim, heads=4, dropout=self.dropout)
            for _ in range(self.n_layers)
        ])
        
        self.ui_layers = nn.ModuleList([
            GraphAttentionLayer(self.embedding_dim, self.embedding_dim, heads=4, dropout=self.dropout)
            for _ in range(self.n_layers)
        ])
        
        # Enhanced modal fusion with gating mechanism
        if self.v_feat is not None and self.t_feat is not None:
            self.fusion = nn.Sequential(
                nn.Linear(self.feat_embed_dim * 2, self.feat_embed_dim * 2),
                nn.LayerNorm(self.feat_embed_dim * 2),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.feat_embed_dim * 2, self.feat_embed_dim * 2)
            )
            self.fusion_gate = nn.Linear(self.feat_embed_dim * 2, self.feat_embed_dim)
        
        # Load and process interaction data
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.edge_index = self.build_edges()
        self.mm_edge_index = None
        self.mm_edge_weight = None
        self.build_modal_graph()
        
        self.to(self.device)

    def build_edges(self):
        # Build user-item interaction graph
        rows = self.interaction_matrix.row
        cols = self.interaction_matrix.col + self.n_users
        
        edge_index = torch.tensor(np.vstack([
            np.concatenate([rows, cols]),
            np.concatenate([cols, rows])
        ]), dtype=torch.long).to(self.device)
        
        return edge_index

    def build_modal_graph(self):
        if self.v_feat is None and self.t_feat is None:
            return
            
        # Build KNN graph for modalities
        if self.v_feat is not None and self.t_feat is not None:
            v_feat = F.normalize(self.v_feat, p=2, dim=1)
            t_feat = F.normalize(self.t_feat, p=2, dim=1)
            sim = 0.5 * (torch.mm(v_feat, v_feat.t()) + torch.mm(t_feat, t_feat.t()))
        elif self.v_feat is not None:
            feats = F.normalize(self.v_feat, p=2, dim=1)
            sim = torch.mm(feats, feats.t())
        else:
            feats = F.normalize(self.t_feat, p=2, dim=1)
            sim = torch.mm(feats, feats.t())
        
        # Get top-k similar items
        values, indices = sim.topk(k=self.knn_k, dim=1)
        rows = torch.arange(sim.size(0), device=self.device).view(-1, 1).expand_as(indices)
        
        # Build edge index and weights
        self.mm_edge_index = torch.stack([
            torch.cat([rows.reshape(-1), indices.reshape(-1)]),
            torch.cat([indices.reshape(-1), rows.reshape(-1)])
        ]).to(self.device)
        
        edge_weights = torch.cat([values.reshape(-1), values.reshape(-1)])
        self.mm_edge_weight = edge_weights.to(self.device)

    def process_modalities(self):
        img_emb = txt_emb = None
        
        if self.v_feat is not None:
            img_emb = self.image_encoder(self.image_embedding.weight)
            for layer in self.mm_layers:
                img_emb = layer(img_emb, self.mm_edge_index)
                
        if self.t_feat is not None:
            txt_emb = self.text_encoder(self.text_embedding.weight)
            for layer in self.mm_layers:
                txt_emb = layer(txt_emb, self.mm_edge_index)
                
        if img_emb is not None and txt_emb is not None:
            concat_emb = torch.cat([img_emb, txt_emb], dim=1)
            fusion_weights = torch.sigmoid(self.fusion_gate(concat_emb))
            modal_emb = self.fusion(concat_emb)
            modal_emb = modal_emb.view(-1, 2, self.feat_embed_dim)
            modal_emb = (modal_emb * fusion_weights.unsqueeze(1)).sum(dim=1)
        else:
            modal_emb = img_emb if img_emb is not None else txt_emb
            
        return modal_emb, img_emb, txt_emb

    def forward(self):
        # Process modalities with attention mechanism
        modal_emb, img_emb, txt_emb = self.process_modalities()
        
        # Process user-item graph with residual connections
        x = torch.cat([self.user_embedding.weight, self.item_embedding.weight])
        all_embs = [x]
        
        for layer in self.ui_layers:
            if self.training:
                x = F.dropout(x, p=self.dropout, training=True)
            x = layer(x, self.edge_index)
            all_embs.append(x)
            
        # Aggregate embeddings with attention weights
        stacked_embs = torch.stack(all_embs, dim=1)
        alpha = F.softmax(torch.matmul(stacked_embs, stacked_embs.mean(dim=1).unsqueeze(-1)), dim=1)
        x = (stacked_embs * alpha).sum(dim=1)
        
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
        
        # BPR loss with hard negative mining
        pos_scores = torch.sum(u_emb * pos_emb, dim=1)
        neg_scores = torch.sum(u_emb * neg_emb, dim=1)
        
        mf_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        
        # Enhanced modal contrastive loss with momentum
        modal_loss = 0.0
        if img_emb is not None and txt_emb is not None:
            pos_sim = torch.sum(img_emb[pos_items] * txt_emb[pos_items], dim=1) / self.temp
            neg_sim = torch.sum(img_emb[pos_items].unsqueeze(1) * txt_emb[neg_items].unsqueeze(0), dim=2) / self.temp
            
            modal_loss = -torch.mean(
                pos_sim - torch.logsumexp(torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1), dim=1)
            )
        
        # L2 regularization on embeddings
        reg_loss = self.reg_weight * (
            torch.norm(u_emb) +
            torch.norm(pos_emb) +
            torch.norm(neg_emb)
        )
        
        loss = mf_loss + 0.2 * modal_loss + reg_loss
        return loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_emb, item_emb, _, _ = self.forward()
        
        # Compute scores with temperature scaling
        scores = torch.matmul(user_emb[user], item_emb.transpose(0, 1)) / np.sqrt(self.embedding_dim)
        return scores