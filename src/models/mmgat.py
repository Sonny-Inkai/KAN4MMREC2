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
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.shape[0]
        
        q = self.q_proj(q).view(batch_size, -1, self.num_heads, q.shape[-1] // self.num_heads).transpose(1, 2)
        k = self.k_proj(k).view(batch_size, -1, self.num_heads, k.shape[-1] // self.num_heads).transpose(1, 2)
        v = self.v_proj(v).view(batch_size, -1, self.num_heads, v.shape[-1] // self.num_heads).transpose(1, 2)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        x = torch.matmul(attn, v).transpose(1, 2).reshape(batch_size, -1, q.shape[-1] * self.num_heads)
        x = self.out_proj(x)
        return self.norm(x)

class ModalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
    def forward(self, x):
        return self.encoder(x)

class GraphAttentionLayer(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.gat = GATConv(dim, dim // num_heads, heads=num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim)
        )
        
    def forward(self, x, edge_index, edge_weight=None):
        # Residual connection
        residual = x
        x = self.norm(x + self.dropout(self.gat(x, edge_index, edge_weight)))
        x = self.norm(x + self.dropout(self.ffn(x)))
        return x + residual

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
        self.temp = 0.07  # Temperature parameter for contrastive loss
        
        # User-Item embeddings with better initialization
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        self.apply(self._init_weights)
        
        # Modal encoders
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_encoder = ModalEncoder(self.v_feat.shape[1], self.hidden_dim, self.feat_embed_dim, self.dropout)
            
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_encoder = ModalEncoder(self.t_feat.shape[1], self.hidden_dim, self.feat_embed_dim, self.dropout)
        
        # Cross-modal attention for better fusion
        if self.v_feat is not None and self.t_feat is not None:
            self.cross_attention = CrossModalAttention(self.feat_embed_dim)
            self.fusion = nn.Sequential(
                nn.Linear(self.feat_embed_dim * 2, self.feat_embed_dim),
                nn.LayerNorm(self.feat_embed_dim),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.feat_embed_dim, self.feat_embed_dim),
                nn.LayerNorm(self.feat_embed_dim)
            )
        
        # Graph layers with residual connections
        self.mm_layers = nn.ModuleList([
            GraphAttentionLayer(self.feat_embed_dim, dropout=self.dropout)
            for _ in range(self.n_layers)
        ])
        
        self.ui_layers = nn.ModuleList([
            GraphAttentionLayer(self.embedding_dim, dropout=self.dropout)
            for _ in range(self.n_layers)
        ])
        
        # Load and process interaction data
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.edge_index = self.build_edges()
        self.edge_weight = None
        self.mm_edge_index = None
        self.mm_edge_weight = None
        self.build_modal_graph()
        
        self.to(self.device)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
    
    def build_edges(self):
        rows = self.interaction_matrix.row
        cols = self.interaction_matrix.col + self.n_users
        
        edge_index = torch.tensor(np.vstack([
            np.concatenate([rows, cols]),
            np.concatenate([cols, rows])
        ]), dtype=torch.long).to(self.device)
        
        # Calculate edge weights based on interaction frequency
        values = torch.ones(edge_index.size(1)).to(self.device)
        self.edge_weight = values
        
        return edge_index

    def build_modal_graph(self):
        if self.v_feat is None and self.t_feat is None:
            return
            
        if self.v_feat is not None:
            feats = F.normalize(self.v_feat, p=2, dim=1)
        if self.t_feat is not None:
            feats = F.normalize(self.t_feat, p=2, dim=1)
            
        # Build KNN graph with cosine similarity
        sim = torch.mm(feats, feats.t())
        values, indices = sim.topk(k=self.knn_k, dim=1)
        rows = torch.arange(feats.size(0), device=self.device).view(-1, 1).expand_as(indices)
        
        self.mm_edge_index = torch.stack([
            torch.cat([rows.reshape(-1), indices.reshape(-1)]),
            torch.cat([indices.reshape(-1), rows.reshape(-1)])
        ]).to(self.device)
        
        # Edge weights based on similarity scores
        self.mm_edge_weight = torch.cat([values.reshape(-1), values.reshape(-1)]).to(self.device)

    def forward(self):
        # Process modalities with enhanced encoders
        img_emb = txt_emb = None
        
        if self.v_feat is not None:
            img_emb = self.image_encoder(self.image_embedding.weight)
            
        if self.t_feat is not None:
            txt_emb = self.text_encoder(self.text_embedding.weight)
        
        # Modal fusion with cross-attention
        if img_emb is not None and txt_emb is not None:
            # Cross-modal attention
            img_attended = self.cross_attention(img_emb.unsqueeze(0), txt_emb.unsqueeze(0), txt_emb.unsqueeze(0)).squeeze(0)
            txt_attended = self.cross_attention(txt_emb.unsqueeze(0), img_emb.unsqueeze(0), img_emb.unsqueeze(0)).squeeze(0)
            
            # Combine attended features
            modal_emb = self.fusion(torch.cat([img_attended, txt_attended], dim=1))
        else:
            modal_emb = img_emb if img_emb is not None else txt_emb
        
        # Apply modal graph attention with edge weights
        if modal_emb is not None:
            for layer in self.mm_layers:
                modal_emb = layer(modal_emb, self.mm_edge_index, self.mm_edge_weight)
        
        # Process user-item graph with enhanced attention
        x = torch.cat([self.user_embedding.weight, self.item_embedding.weight])
        all_embs = [x]
        
        for layer in self.ui_layers:
            if self.training:
                x = F.dropout(x, p=self.dropout)
            x = layer(x, self.edge_index, self.edge_weight)
            all_embs.append(x)
            
        # Aggregate embeddings with learned weights
        weights = F.softmax(torch.randn(len(all_embs)).to(self.device), dim=0)
        x = torch.stack(all_embs, dim=1) @ weights.view(-1, 1)
        x = x.squeeze(1)
        
        user_emb, item_emb = torch.split(x, [self.n_users, self.n_items])
        
        # Combine with modal embeddings using residual connection
        if modal_emb is not None:
            item_emb = item_emb + F.normalize(modal_emb, p=2, dim=1)
            
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
        
        # InfoNCE loss for better contrast
        pos_exp = torch.exp(pos_scores / self.temp)
        neg_exp = torch.exp(neg_scores / self.temp)
        mf_loss = -torch.mean(torch.log(pos_exp / (pos_exp + neg_exp)))
        
        # Modal contrastive loss with momentum
        modal_loss = 0.0
        if img_emb is not None and txt_emb is not None:
            img_pos = F.normalize(img_emb[pos_items], p=2, dim=1)
            txt_pos = F.normalize(txt_emb[pos_items], p=2, dim=1)
            img_neg = F.normalize(img_emb[neg_items], p=2, dim=1)
            txt_neg = F.normalize(txt_emb[neg_items], p=2, dim=1)
            
            pos_sim = torch.sum(img_pos * txt_pos, dim=1) / self.temp
            neg_sim = torch.sum(img_pos * txt_neg, dim=1) / self.temp
            
            modal_loss = -torch.mean(torch.log(torch.exp(pos_sim) / (torch.exp(pos_sim) + torch.exp(neg_sim))))
        
        # L2 regularization with gradient clipping
        reg_loss = self.reg_weight * (
            torch.norm(u_emb) +
            torch.norm(pos_emb) +
            torch.norm(neg_emb)
        )
        
        # Combine losses with dynamic weighting
        total_loss = mf_loss + 0.2 * modal_loss + reg_loss
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        
        return total_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_emb, item_emb, _, _ = self.forward()
        
        # Compute scores with temperature scaling
        scores = torch.matmul(user_emb[user], item_emb.transpose(0, 1)) / self.temp
        return scores