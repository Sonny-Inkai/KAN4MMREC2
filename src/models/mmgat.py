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
            nn.Linear(hidden_dim, hidden_dim),  # Added extra layer for better representation
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
        self.proj = nn.Linear(dim, dim)  # Added projection layer
        
    def forward(self, x, edge_index):
        attended = self.gat(x, edge_index)
        projected = self.proj(attended)
        return self.norm(x + self.dropout(projected))

class MultiModalFusion(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, 2),
            nn.Softmax(dim=1)
        )
        self.fusion = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )
        
    def forward(self, img_emb, txt_emb):
        # Ensure same dimensions for both modalities
        combined = torch.cat([img_emb, txt_emb], dim=1)
        weights = self.attention(combined)
        weighted = torch.cat([
            weights[:, 0:1] * img_emb,
            weights[:, 1:2] * txt_emb
        ], dim=1)
        return self.fusion(weighted)

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
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight, gain=1.0)
        nn.init.xavier_uniform_(self.item_embedding.weight, gain=1.0)
        
        # Modal encoders - ensure output dimension matches embedding_dim
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_encoder = ModalEncoder(self.v_feat.shape[1], self.hidden_dim, self.embedding_dim, self.dropout)
            
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_encoder = ModalEncoder(self.t_feat.shape[1], self.hidden_dim, self.embedding_dim, self.dropout)
            
        # Graph layers
        self.mm_layers = nn.ModuleList([
            GraphAttentionLayer(self.embedding_dim, heads=4, dropout=self.dropout)
            for _ in range(self.n_layers)
        ])
        
        self.ui_layers = nn.ModuleList([
            GraphAttentionLayer(self.embedding_dim, heads=4, dropout=self.dropout)
            for _ in range(self.n_layers)
        ])
        
        # Modal fusion
        if self.v_feat is not None and self.t_feat is not None:
            self.fusion = MultiModalFusion(self.embedding_dim, self.dropout)
            
        # Final projection to ensure consistent dimensions
        self.final_projection = nn.Linear(self.embedding_dim, self.embedding_dim)
        
        # Load data
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.edge_index = self.build_edges()
        self.mm_edge_index = None
        self.build_modal_graph()
        
        self.temp = nn.Parameter(torch.FloatTensor([0.07]))
        
        self.to(self.device)

    def forward(self):
        # Process modalities
        img_emb = txt_emb = None
        
        if self.v_feat is not None:
            img_emb = self.image_encoder(self.image_embedding.weight)
            img_res = img_emb
            for layer in self.mm_layers:
                img_emb = layer(img_emb, self.mm_edge_index)
            img_emb = img_emb + img_res
                
        if self.t_feat is not None:
            txt_emb = self.text_encoder(self.text_embedding.weight)
            txt_res = txt_emb
            for layer in self.mm_layers:
                txt_emb = layer(txt_emb, self.mm_edge_index)
            txt_emb = txt_emb + txt_res
        
        # Fusion with matching dimensions
        if img_emb is not None and txt_emb is not None:
            modal_emb = self.fusion(img_emb, txt_emb)
        else:
            modal_emb = img_emb if img_emb is not None else txt_emb
        
        # Process user-item graph
        x = torch.cat([self.user_embedding.weight, self.item_embedding.weight])
        x_res = x
        all_embs = [x]
        
        for layer in self.ui_layers:
            if self.training:
                x = F.dropout(x, p=self.dropout)
            x = layer(x, self.edge_index)
            all_embs.append(x)
            
        x = torch.stack(all_embs, dim=1).mean(dim=1) + x_res
        user_emb, item_emb = torch.split(x, [self.n_users, self.n_items])
        
        # Combine with modal embeddings using element-wise gating
        if modal_emb is not None:
            gate = torch.sigmoid(self.final_projection(item_emb))
            item_emb = item_emb + gate * modal_emb
            
        return user_emb, item_emb, img_emb, txt_emb

    def info_nce_loss(self, query, positive_key, negative_keys):
        # InfoNCE loss implementation
        positive_score = torch.sum(query * positive_key, dim=1, keepdim=True)
        negative_score = query @ negative_keys.t()
        logits = torch.cat([positive_score, negative_score], dim=1)
        labels = torch.zeros(len(query), device=query.device, dtype=torch.long)
        return F.cross_entropy(logits / self.temp, labels)

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        
        user_emb, item_emb, img_emb, txt_emb = self.forward()
        
        u_emb = user_emb[users]
        pos_emb = item_emb[pos_items]
        neg_emb = item_emb[neg_items]
        
        # Enhanced BPR loss with hard negative mining
        pos_scores = torch.sum(u_emb * pos_emb, dim=1)
        neg_scores = torch.sum(u_emb * neg_emb, dim=1)
        
        # Margin ranking loss with adaptive margin
        margin = torch.clamp(neg_scores.detach() - pos_scores.detach() + 0.5, min=0.1)
        mf_loss = torch.mean(torch.clamp(neg_scores - pos_scores + margin, min=0.0))
        
        # Modal contrastive loss with InfoNCE
        modal_loss = 0.0
        if img_emb is not None and txt_emb is not None:
            img_pos = img_emb[pos_items]
            txt_pos = txt_emb[pos_items]
            txt_neg = txt_emb[neg_items]
            
            modal_loss = (
                self.info_nce_loss(img_pos, txt_pos, txt_neg) +
                self.info_nce_loss(txt_pos, img_pos, img_emb[torch.randperm(self.n_items)])
            ) / 2
        
        # L2 regularization with gradient clipping
        reg_loss = self.reg_weight * (
            torch.norm(u_emb) +
            torch.norm(pos_emb) +
            torch.norm(neg_emb)
        )
        
        return mf_loss + 0.2 * modal_loss + reg_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_emb, item_emb, _, _ = self.forward()
        scores = torch.matmul(user_emb[user], item_emb.transpose(0, 1))
        return scores