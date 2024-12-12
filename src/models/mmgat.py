# coding: utf-8
import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, TransformerConv
from common.abstract_recommender import GeneralRecommender

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        q = self.q_proj(q).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(k).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(v).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(batch_size, -1, self.dim)
        out = self.o_proj(out)
        return self.norm(out)

class ModalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        self.gat = GATConv(output_dim, output_dim, heads=4, dropout=dropout, concat=False)
        self.sage = SAGEConv(output_dim, output_dim, normalize=True)
        self.transformer = TransformerConv(output_dim, output_dim)
        
        self.fusion = nn.Sequential(
            nn.Linear(output_dim * 3, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, edge_index):
        x = self.input_proj(x)
        
        # Multi-scale graph learning
        gat_out = self.gat(x, edge_index)
        sage_out = self.sage(x, edge_index)
        trans_out = self.transformer(x, edge_index)
        
        # Combine different graph embeddings
        combined = torch.cat([gat_out, sage_out, trans_out], dim=-1)
        out = self.fusion(combined)
        return F.normalize(out, p=2, dim=-1)

class CrossModalFusion(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.self_attn = MultiHeadAttention(dim, num_heads)
        self.cross_attn = MultiHeadAttention(dim, num_heads)
        
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, 2),
            nn.Softmax(dim=-1)
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU()
        )
        
    def forward(self, x1, x2):
        # Self attention
        x1_self = self.self_attn(x1.unsqueeze(0), x1.unsqueeze(0), x1.unsqueeze(0)).squeeze(0)
        x2_self = self.self_attn(x2.unsqueeze(0), x2.unsqueeze(0), x2.unsqueeze(0)).squeeze(0)
        
        # Cross attention
        x1_cross = self.cross_attn(x1.unsqueeze(0), x2.unsqueeze(0), x2.unsqueeze(0)).squeeze(0)
        x2_cross = self.cross_attn(x2.unsqueeze(0), x1.unsqueeze(0), x1.unsqueeze(0)).squeeze(0)
        
        # Gated fusion
        x1_cat = torch.cat([x1_self, x1_cross], dim=-1)
        x2_cat = torch.cat([x2_self, x2_cross], dim=-1)
        
        x1_gate = self.gate(x1_cat)
        x2_gate = self.gate(x2_cat)
        
        x1_fused = x1_gate[:, 0:1] * x1_self + x1_gate[:, 1:2] * x1_cross
        x2_fused = x2_gate[:, 0:1] * x2_self + x2_gate[:, 1:2] * x2_cross
        
        # Final fusion
        out = self.fusion(torch.cat([x1_fused, x2_fused], dim=-1))
        return F.normalize(out, p=2, dim=-1)

class MMGAT(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MMGAT, self).__init__(config, dataset)
        
        self.embedding_dim = config["embedding_size"]
        self.feat_embed_dim = config["feat_embed_dim"]
        self.hidden_dim = self.feat_embed_dim * 2
        self.n_layers = config["n_mm_layers"]
        self.dropout = config["dropout"]
        self.reg_weight = config["reg_weight"]
        
        # User-Item embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight, gain=1/2)
        nn.init.xavier_uniform_(self.item_embedding.weight, gain=1/2)
        
        # Modal encoders
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_encoder = ModalEncoder(self.v_feat.shape[1], self.hidden_dim, self.feat_embed_dim)
            
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_encoder = ModalEncoder(self.t_feat.shape[1], self.hidden_dim, self.feat_embed_dim)
            
        # Cross-modal fusion
        if self.v_feat is not None and self.t_feat is not None:
            self.modal_fusion = CrossModalFusion(self.feat_embed_dim)
            
        # Graph layers
        self.gcn_layers = nn.ModuleList([
            GCNConv(self.embedding_dim, self.embedding_dim)
            for _ in range(self.n_layers)
        ])
        
        # Load data
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.edge_index = self.build_edge_index()
        self.edge_weight = self.compute_edge_weight()
        self.modal_edge_index = None
        self.build_modal_graph()
        
        self.to(self.device)
        
    def build_edge_index(self):
        rows = self.interaction_matrix.row
        cols = self.interaction_matrix.col + self.n_users
        
        edge_index = torch.tensor(np.vstack([
            np.concatenate([rows, cols, cols, rows]),
            np.concatenate([cols, rows, rows, cols])
        ]), dtype=torch.long).to(self.device)
        
        return edge_index
        
    def compute_edge_weight(self):
        row, col = self.edge_index
        deg = torch.bincount(row).float()
        deg_inv_sqrt = torch.pow(deg + 1e-12, -0.5)
        edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        return edge_weight
        
    def build_modal_graph(self):
        if not (self.v_feat is not None or self.t_feat is not None):
            return
            
        features = []
        if self.v_feat is not None:
            features.append(F.normalize(self.image_encoder.input_proj(self.v_feat), p=2, dim=1))
        if self.t_feat is not None:
            features.append(F.normalize(self.text_encoder.input_proj(self.t_feat), p=2, dim=1))
            
        feature = torch.stack(features).mean(dim=0) if len(features) > 1 else features[0]
        sim = torch.mm(feature, feature.t())
        
        # Build KNN graph
        values, indices = sim.topk(k=20, dim=1)
        rows = torch.arange(feature.size(0), device=self.device).repeat_interleave(20)
        self.modal_edge_index = torch.stack([rows, indices.reshape(-1)])

    def forward(self):
        # Process modalities
        modal_emb = None
        if self.v_feat is not None and self.t_feat is not None:
            img_feat = self.image_encoder(self.image_embedding.weight, self.modal_edge_index)
            txt_feat = self.text_encoder(self.text_embedding.weight, self.modal_edge_index)
            modal_emb = self.modal_fusion(img_feat, txt_feat)
        elif self.v_feat is not None:
            modal_emb = self.image_encoder(self.image_embedding.weight, self.modal_edge_index)
        elif self.t_feat is not None:
            modal_emb = self.text_encoder(self.text_embedding.weight, self.modal_edge_index)
            
        # Graph convolution
        x = torch.cat([self.user_embedding.weight, self.item_embedding.weight])
        all_embs = [x]
        
        for layer in self.gcn_layers:
            if self.training:
                mask = torch.rand_like(self.edge_weight) > self.dropout
                edge_weight = self.edge_weight * mask / (1 - self.dropout)
            else:
                edge_weight = self.edge_weight
                
            x = layer(x, self.edge_index, edge_weight)
            x = F.normalize(x, p=2, dim=1)
            all_embs.append(x)
            
        x = torch.stack(all_embs, dim=1).mean(dim=1)
        user_emb, item_emb = torch.split(x, [self.n_users, self.n_items])
        
        if modal_emb is not None:
            item_emb = item_emb + modal_emb
            
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
        rec_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        
        # L2 regularization
        reg_loss = self.reg_weight * (
            torch.norm(u_emb) +
            torch.norm(pos_emb) +
            torch.norm(neg_emb)
        )
        
        return rec_loss + reg_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_emb, item_emb = self.forward()
        scores = torch.matmul(user_emb[user], item_emb.transpose(0, 1))
        return scores