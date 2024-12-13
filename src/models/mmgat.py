import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from common.abstract_recommender import GeneralRecommender

class AdaptiveNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm_layer = nn.LayerNorm(dim)
        self.adaptive_weight = nn.Parameter(torch.ones(1))
        
    def forward(self, x):
        norm_x = self.norm_layer(x)
        return self.adaptive_weight * norm_x + (1 - self.adaptive_weight) * x

class MultiheadCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)
        
        self.dropout = nn.Dropout(dropout)
        self.norm = AdaptiveNorm(dim)
        
    def forward(self, x, context):
        B, N, C = x.shape
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(context).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(context).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        x = torch.matmul(attn, v).transpose(1, 2).reshape(B, N, C)
        x = self.o_proj(x)
        return self.norm(x)

class EnhancedGATLayer(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.gat = GATConv(dim, dim // num_heads, heads=num_heads, dropout=dropout, concat=True)
        self.cross_attn = MultiheadCrossAttention(dim, num_heads=num_heads, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim)
        )
        self.norm1 = AdaptiveNorm(dim)
        self.norm2 = AdaptiveNorm(dim)
        self.norm3 = AdaptiveNorm(dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index, context=None):
        # Graph attention
        h = self.gat(x, edge_index)
        h = self.norm1(h + x)
        
        # Cross attention with context if available
        if context is not None:
            h = h.unsqueeze(0)
            context = context.unsqueeze(0)
            h = self.cross_attn(h, context).squeeze(0)
            h = self.norm2(h)
        
        # Feed forward
        h = self.norm3(h + self.dropout(self.ffn(h)))
        return h

class ModalFusion(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads=8, dropout=dropout)
        self.norm1 = AdaptiveNorm(dim)
        self.norm2 = AdaptiveNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim)
        )
        
    def forward(self, x1, x2):
        # Self attention between modalities
        x = torch.stack([x1, x2], dim=0)
        h, _ = self.attention(x, x, x)
        h = self.norm1(h + x)
        
        # Feed forward
        h = self.norm2(h + self.ffn(h))
        return torch.mean(h, dim=0)

class MMGAT(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MMGAT, self).__init__(config, dataset)
        
        self.embedding_dim = config["embedding_size"]
        self.feat_embed_dim = config["feat_embed_dim"]
        self.n_layers = config["n_mm_layers"]
        self.dropout = config["dropout"]
        self.reg_weight = config["reg_weight"]
        self.knn_k = config["knn_k"]
        self.temperature = 0.2
        
        # Embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        
        nn.init.trunc_normal_(self.user_embedding.weight, std=0.02)
        nn.init.trunc_normal_(self.item_embedding.weight, std=0.02)
        
        # Modal processing
        if self.v_feat is not None:
            self.v_feat = F.normalize(self.v_feat, p=2, dim=1)
            self.image_encoder = nn.Sequential(
                nn.Linear(self.v_feat.shape[1], self.feat_embed_dim * 2),
                nn.LayerNorm(self.feat_embed_dim * 2),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.feat_embed_dim * 2, self.feat_embed_dim)
            )
            
        if self.t_feat is not None:
            self.t_feat = F.normalize(self.t_feat, p=2, dim=1)
            self.text_encoder = nn.Sequential(
                nn.Linear(self.t_feat.shape[1], self.feat_embed_dim * 2),
                nn.LayerNorm(self.feat_embed_dim * 2),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.feat_embed_dim * 2, self.feat_embed_dim)
            )
        
        # Modal fusion
        if self.v_feat is not None and self.t_feat is not None:
            self.fusion = ModalFusion(self.feat_embed_dim, self.dropout)
            self.modal_gate = nn.Parameter(torch.ones(2))
        
        # Graph layers
        self.gat_layers = nn.ModuleList([
            EnhancedGATLayer(self.embedding_dim, dropout=self.dropout)
            for _ in range(self.n_layers)
        ])
        
        # Loss weights
        self.loss_weights = nn.Parameter(torch.ones(3))  # [bpr, modal, reg]
        
        # Load data
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
        
        return edge_index

    def compute_similarity_batched(self, features):
        n_items = features.size(0)
        edge_index_list = []
        edge_weight_list = []
        
        for i in range(0, n_items, self.batch_size):
            end_idx = min(i + self.batch_size, n_items)
            batch_features = features[i:end_idx]
            
            # Compute similarity for current batch
            sim = torch.mm(batch_features, features.t())
            topk_values, topk_indices = sim.topk(k=self.knn_k, dim=1)
            
            # Generate edge indices
            rows = torch.arange(i, end_idx, device=self.device)
            rows = rows.view(-1, 1).expand_as(topk_indices)
            
            edge_index_list.append(torch.stack([rows.reshape(-1), topk_indices.reshape(-1)]))
            edge_weight_list.append(F.softmax(topk_values, dim=1).reshape(-1))
            
            del sim
            torch.cuda.empty_cache()
        
        edge_index = torch.cat(edge_index_list, dim=1)
        edge_weight = torch.cat(edge_weight_list)
        
        return edge_index, edge_weight

    def build_modal_graph(self):
        if self.v_feat is None and self.t_feat is None:
            return
            
        if self.v_feat is not None and self.t_feat is not None:
            # Dynamic modal weighting
            weights = F.softmax(self.modal_gate, dim=0)
            v_features = self.image_encoder(self.v_feat)
            t_features = self.text_encoder(self.t_feat)
            features = weights[0] * v_features + weights[1] * t_features
        else:
            feat = self.v_feat if self.v_feat is not None else self.t_feat
            encoder = self.image_encoder if self.v_feat is not None else self.text_encoder
            features = encoder(feat)
        
        features = F.normalize(features, p=2, dim=1)
        self.mm_edge_index, self.mm_edge_weight = self.compute_similarity_batched(features)
        
    def forward(self):
        # Process modalities
        modal_emb = None
        
        if self.v_feat is not None or self.t_feat is not None:
            if self.v_feat is not None and self.t_feat is not None:
                v_emb = self.image_encoder(self.v_feat)
                t_emb = self.text_encoder(self.t_feat)
                modal_emb = self.fusion(v_emb, t_emb)
            else:
                feat = self.v_feat if self.v_feat is not None else self.t_feat
                encoder = self.image_encoder if self.v_feat is not None else self.text_encoder
                modal_emb = encoder(feat)
            
            modal_emb = F.normalize(modal_emb, p=2, dim=1)
        
        # Process graph structure
        x = torch.cat([self.user_embedding.weight, self.item_embedding.weight])
        
        layer_outputs = [x]
        for layer in self.gat_layers:
            x = layer(x, self.edge_index, modal_emb)
            x = F.normalize(x, p=2, dim=1)
            layer_outputs.append(x)
        
        # Weighted multi-layer combination
        attention_weights = F.softmax(torch.randn(len(layer_outputs), device=self.device), dim=0)
        x = torch.stack(layer_outputs, dim=0)
        x = (x * attention_weights.view(-1, 1, 1)).sum(0)
        
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
        
        # InfoNCE loss with temperature scaling
        pos_scores = torch.sum(u_emb * pos_emb, dim=1) / self.temperature
        neg_scores = torch.sum(u_emb * neg_emb, dim=1) / self.temperature
        
        bpr_loss = -torch.log(
            torch.exp(pos_scores) /
            (torch.exp(pos_scores) + torch.exp(neg_scores))
        ).mean()
        
        # Contrastive modal loss
        modal_loss = torch.tensor(0.0, device=self.device)
        if self.v_feat is not None and self.t_feat is not None:
            v_emb = self.image_encoder(self.v_feat)
            t_emb = self.text_encoder(self.t_feat)
            
            pos_sim = F.cosine_similarity(v_emb[pos_items], t_emb[pos_items])
            neg_sim = F.cosine_similarity(v_emb[pos_items], t_emb[neg_items])
            
            modal_loss = -torch.log(
                torch.exp(pos_sim / self.temperature) /
                (torch.exp(pos_sim / self.temperature) + torch.exp(neg_sim / self.temperature))
            ).mean()
        
        # L2 regularization with weight decay
        l2_loss = self.reg_weight * (
            torch.norm(u_emb) +
            torch.norm(pos_emb) +
            torch.norm(neg_emb)
        )
        
        # Dynamic loss weighting
        loss_weights = F.softmax(self.loss_weights, dim=0)
        total_loss = (
            loss_weights[0] * bpr_loss +
            loss_weights[1] * modal_loss +
            loss_weights[2] * l2_loss
        )
        
        return total_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_emb, item_emb = self.forward()
        scores = torch.matmul(user_emb[user], item_emb.transpose(0, 1))
        return scores