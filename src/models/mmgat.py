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
    def __init__(self, dim, heads=4, dropout=0.1):
        super().__init__()
        self.gat = GATConv(dim, dim // heads, heads=heads, dropout=dropout)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()
        
    def forward(self, x, edge_index):
        return self.norm(x + self.dropout(self.act(self.gat(x, edge_index))))

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
        
        # User-Item embeddings with better initialization
        self.user_embedding = nn.Parameter(
            torch.zeros(self.n_users, self.embedding_dim).normal_(mean=0, std=0.01)
        )
        self.item_embedding = nn.Parameter(
            torch.zeros(self.n_items, self.embedding_dim).normal_(mean=0, std=0.01)
        )
        
        # Modal encoders
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_encoder = ModalEncoder(self.v_feat.shape[1], self.hidden_dim, self.feat_embed_dim, self.dropout)
            
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_encoder = ModalEncoder(self.t_feat.shape[1], self.hidden_dim, self.feat_embed_dim, self.dropout)
            
        # Graph layers with residual connections
        self.mm_layers = nn.ModuleList([
            GraphAttentionLayer(self.feat_embed_dim, heads=4, dropout=self.dropout)
            for _ in range(self.n_layers)
        ])
        
        self.ui_layers = nn.ModuleList([
            GraphAttentionLayer(self.embedding_dim, heads=4, dropout=self.dropout)
            for _ in range(self.n_layers)
        ])
        
        # Modal fusion with gating mechanism
        if self.v_feat is not None and self.t_feat is not None:
            self.fusion = nn.Sequential(
                nn.Linear(self.feat_embed_dim * 2, self.feat_embed_dim * 2),
                nn.LayerNorm(self.feat_embed_dim * 2),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.feat_embed_dim * 2, self.feat_embed_dim)
            )
            self.gate = nn.Sequential(
                nn.Linear(self.feat_embed_dim * 2, 2),
                nn.Softmax(dim=-1)
            )
        
        # Load data and build graphs
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.edge_index, self.edge_norm = self.build_edges()
        self.mm_edge_index, self.mm_edge_norm = None, None
        self.build_modal_graph()
        
        # Layer-wise learnable temperature
        self.temp = nn.Parameter(torch.ones(self.n_layers))
        
        self.to(self.device)

    def compute_normalized_laplacian(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return values

    def build_edges(self):
        rows = self.interaction_matrix.row
        cols = self.interaction_matrix.col + self.n_users
        
        edge_index = torch.tensor(np.vstack([
            np.concatenate([rows, cols]),
            np.concatenate([cols, rows])
        ]), dtype=torch.long).to(self.device)
        
        edge_norm = self.compute_normalized_laplacian(
            edge_index, 
            (self.n_users + self.n_items, self.n_users + self.n_items)
        ).to(self.device)
        
        return edge_index, edge_norm

    def build_modal_graph(self):
        if self.v_feat is None and self.t_feat is None:
            return
            
        if self.v_feat is not None:
            feats = F.normalize(self.v_feat, p=2, dim=1)
            sim = torch.mm(feats, feats.t())
            
        if self.t_feat is not None:
            feats = F.normalize(self.t_feat, p=2, dim=1)
            sim = torch.mm(feats, feats.t())
            
        values, indices = sim.topk(k=self.knn_k, dim=1)
        rows = torch.arange(feats.size(0), device=self.device).view(-1, 1).expand_as(indices)
        
        edge_index = torch.stack([
            torch.cat([rows.reshape(-1), indices.reshape(-1)]),
            torch.cat([indices.reshape(-1), rows.reshape(-1)])
        ]).to(self.device)
        
        edge_norm = self.compute_normalized_laplacian(
            edge_index, 
            (feats.size(0), feats.size(0))
        ).to(self.device)
        
        self.mm_edge_index = edge_index
        self.mm_edge_norm = edge_norm

    def forward(self):
        # Process modalities with layer normalization
        img_emb = txt_emb = None
        
        if self.v_feat is not None:
            img_emb = self.image_encoder(self.image_embedding.weight)
            img_emb_list = [img_emb]
            for i, layer in enumerate(self.mm_layers):
                img_emb = layer(img_emb, self.mm_edge_index)
                img_emb = F.normalize(img_emb, p=2, dim=1) * F.softplus(self.temp[i])
                img_emb_list.append(img_emb)
            img_emb = torch.stack(img_emb_list, dim=1).mean(dim=1)
                
        if self.t_feat is not None:
            txt_emb = self.text_encoder(self.text_embedding.weight)
            txt_emb_list = [txt_emb]
            for i, layer in enumerate(self.mm_layers):
                txt_emb = layer(txt_emb, self.mm_edge_index)
                txt_emb = F.normalize(txt_emb, p=2, dim=1) * F.softplus(self.temp[i])
                txt_emb_list.append(txt_emb)
            txt_emb = torch.stack(txt_emb_list, dim=1).mean(dim=1)
        
        # Gated modal fusion
        if img_emb is not None and txt_emb is not None:
            modal_cat = torch.cat([img_emb, txt_emb], dim=1)
            gate_weights = self.gate(modal_cat)
            modal_emb = self.fusion(modal_cat)
            modal_emb = modal_emb * (gate_weights[:, 0].unsqueeze(1)) + modal_emb * (gate_weights[:, 1].unsqueeze(1))
        else:
            modal_emb = img_emb if img_emb is not None else txt_emb
        
        # Process user-item graph with gradient stabilization
        x = torch.cat([self.user_embedding, self.item_embedding])
        x = F.normalize(x, p=2, dim=1)
        all_embs = [x]
        
        for i, layer in enumerate(self.ui_layers):
            if self.training:
                x = F.dropout(x, p=self.dropout, training=True)
            x = layer(x, self.edge_index)
            x = F.normalize(x, p=2, dim=1) * F.softplus(self.temp[i])
            all_embs.append(x)
            
        x = torch.stack(all_embs, dim=1)
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
        
        # InfoNCE loss with temperature scaling
        pos_scores = torch.sum(u_emb * pos_emb, dim=1) / 0.2
        neg_scores = torch.sum(u_emb * neg_emb, dim=1) / 0.2
        
        mf_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        
        # Modal alignment with momentum
        modal_loss = 0.0
        if img_emb is not None and txt_emb is not None:
            img_pos = F.normalize(img_emb[pos_items], p=2, dim=1)
            txt_pos = F.normalize(txt_emb[pos_items], p=2, dim=1)
            img_neg = F.normalize(img_emb[neg_items], p=2, dim=1)
            txt_neg = F.normalize(txt_emb[neg_items], p=2, dim=1)
            
            pos_sim = torch.sum(img_pos * txt_pos, dim=1) / 0.1
            neg_sim = torch.sum(img_pos * txt_neg, dim=1) / 0.1
            
            modal_loss = -torch.mean(F.logsigmoid(pos_sim - neg_sim))
            
            # Cross-modal momentum
            with torch.no_grad():
                momentum = 0.999
                img_emb.data = momentum * img_emb.data + (1 - momentum) * txt_emb.data
                txt_emb.data = momentum * txt_emb.data + (1 - momentum) * img_emb.data
        
        # L2 regularization with gradient clipping
        reg_loss = self.reg_weight * (
            torch.sum(torch.clamp(u_emb, min=-1, max=1) ** 2) +
            torch.sum(torch.clamp(pos_emb, min=-1, max=1) ** 2) +
            torch.sum(torch.clamp(neg_emb, min=-1, max=1) ** 2)
        ) / u_emb.shape[0]
        
        return mf_loss + 0.2 * modal_loss + reg_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_emb, item_emb, _, _ = self.forward()
        
        scores = torch.matmul(user_emb[user], item_emb.transpose(0, 1))
        scores = scores / 0.2  # Same temperature as training
        
        return scores