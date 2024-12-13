# coding: utf-8
import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from common.abstract_recommender import GeneralRecommender

class CrossModalAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(dim, num_heads)
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, q, k, v, mask=None):
        attn_output, _ = self.attention(q, k, v, key_padding_mask=mask)
        return self.norm(q + attn_output)

class ModalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.encoder(x)

class GraphAttentionLayer(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.gat = GATConv(dim, dim // 4, heads=4, dropout=dropout)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index, edge_weight=None):
        out = self.gat(x, edge_index, edge_weight)
        return self.norm(x + self.dropout(out))

class MMGAT(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MMGAT, self).__init__(config, dataset)
        
        self.embedding_dim = config["embedding_size"]
        self.feat_embed_dim = config["feat_embed_dim"]
        self.hidden_dim = self.feat_embed_dim * 4
        self.n_layers = config["n_mm_layers"]
        self.dropout = 0.6  # Increased dropout
        self.reg_weight = config["reg_weight"]
        self.knn_k = config["knn_k"]
        
        # Initialize embeddings
        self.user_embedding = nn.Parameter(torch.zeros(self.n_users, self.embedding_dim))
        self.item_embedding = nn.Parameter(torch.zeros(self.n_items, self.embedding_dim))
        nn.init.xavier_normal_(self.user_embedding)
        nn.init.xavier_normal_(self.item_embedding)
        
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_encoder = ModalEncoder(self.v_feat.shape[1], self.hidden_dim, self.feat_embed_dim, self.dropout)
        
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_encoder = ModalEncoder(self.t_feat.shape[1], self.hidden_dim, self.feat_embed_dim, self.dropout)
        
        # Cross-modal fusion
        if self.v_feat is not None and self.t_feat is not None:
            self.cross_modal_attention = CrossModalAttention(self.feat_embed_dim)
            self.modal_fusion = nn.Sequential(
                nn.Linear(self.feat_embed_dim * 2, self.feat_embed_dim),
                nn.ReLU(),
                nn.LayerNorm(self.feat_embed_dim),
                nn.Dropout(self.dropout)
            )

        # Graph layers
        self.graph_layers = nn.ModuleList([
            GraphAttentionLayer(self.embedding_dim, self.dropout)
            for _ in range(self.n_layers)
        ])
        
        # Load data
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.edge_index, self.edge_weight = self.build_graph()
        self.mm_edge_index = None
        self.build_modal_graph()
        
        # Additional projections
        self.modal_proj = nn.Linear(self.feat_embed_dim, self.embedding_dim)
        self.predictor = nn.Linear(self.embedding_dim, self.embedding_dim)
        
        self.to(self.device)

    def build_graph(self):
        rows = self.interaction_matrix.row
        cols = self.interaction_matrix.col + self.n_users
        
        edge_index = torch.tensor(np.vstack([
            np.concatenate([rows, cols]),
            np.concatenate([cols, rows])
        ]), dtype=torch.long).to(self.device)
        
        # Compute edge weights based on degree
        deg = np.bincount(rows)
        deg_inv_sqrt = np.power(deg + 1e-12, -0.5)
        row_weight = deg_inv_sqrt[rows]
        
        deg = np.bincount(cols)
        deg_inv_sqrt = np.power(deg + 1e-12, -0.5)
        col_weight = deg_inv_sqrt[cols]
        
        edge_weight = np.concatenate([row_weight * col_weight, row_weight * col_weight])
        return edge_index, torch.FloatTensor(edge_weight).to(self.device)

    def build_modal_graph(self):
        if not (self.v_feat is not None or self.t_feat is not None):
            return
            
        if self.v_feat is not None and self.t_feat is not None:
            v_feat = F.normalize(self.v_feat, p=2, dim=1)
            t_feat = F.normalize(self.t_feat, p=2, dim=1)
            sim = torch.mm(v_feat, v_feat.t()) + torch.mm(t_feat, t_feat.t())
            sim = sim / 2
        elif self.v_feat is not None:
            feats = F.normalize(self.v_feat, p=2, dim=1)
            sim = torch.mm(feats, feats.t())
        else:
            feats = F.normalize(self.t_feat, p=2, dim=1)
            sim = torch.mm(feats, feats.t())
            
        values, indices = sim.topk(k=self.knn_k, dim=1)
        rows = torch.arange(sim.size(0)).view(-1, 1).expand_as(indices)
        
        self.mm_edge_index = torch.stack([
            torch.cat([rows.reshape(-1), indices.reshape(-1)]),
            torch.cat([indices.reshape(-1), rows.reshape(-1)])
        ]).to(self.device)

    def forward(self):
        # Process modalities
        modal_emb = None
        if self.v_feat is not None:
            img_emb = self.image_encoder(self.image_embedding.weight)
            modal_emb = img_emb
        
        if self.t_feat is not None:
            txt_emb = self.text_encoder(self.text_embedding.weight)
            if modal_emb is None:
                modal_emb = txt_emb
            else:
                # Cross-modal attention fusion
                img_emb = self.cross_modal_attention(img_emb.unsqueeze(0), txt_emb.unsqueeze(0), txt_emb.unsqueeze(0))
                txt_emb = self.cross_modal_attention(txt_emb.unsqueeze(0), img_emb.unsqueeze(0), img_emb.unsqueeze(0))
                modal_emb = self.modal_fusion(torch.cat([img_emb.squeeze(0), txt_emb.squeeze(0)], dim=-1))

        # Project modal embeddings
        if modal_emb is not None:
            modal_emb = self.modal_proj(modal_emb)
            modal_emb = F.normalize(modal_emb, p=2, dim=1)

        # Process user-item graph
        x = torch.cat([self.user_embedding, self.item_embedding])
        all_embs = [x]

        for layer in self.graph_layers:
            if self.training:
                x = F.dropout(x, p=self.dropout)
            x = layer(x, self.edge_index, self.edge_weight)
            all_embs.append(x)
            
        x = torch.stack(all_embs, dim=1).mean(dim=1)
        user_emb, item_emb = torch.split(x, [self.n_users, self.n_items])

        # Combine with modal embeddings
        if modal_emb is not None:
            item_emb = item_emb + F.dropout(modal_emb, p=self.dropout, training=self.training)

        user_emb = F.normalize(user_emb, p=2, dim=1)
        item_emb = F.normalize(item_emb, p=2, dim=1)

        return self.predictor(user_emb), self.predictor(item_emb)

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
        
        loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))

        # L2 regularization
        reg_loss = self.reg_weight * (
            torch.norm(u_emb) +
            torch.norm(pos_emb) +
            torch.norm(neg_emb)
        )

        return loss + reg_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_emb, item_emb = self.forward()
        scores = torch.matmul(user_emb[user], item_emb.transpose(0, 1))
        return scores