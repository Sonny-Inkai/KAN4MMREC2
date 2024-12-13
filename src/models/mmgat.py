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
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        return self.encoder(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        q = self.q_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.dim)
        
        return self.out_proj(attn_output)

class GraphAttentionLayer(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(dim, dropout=dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index):
        # Self attention
        x = x + self.dropout(self.attention(x.unsqueeze(0)).squeeze(0))
        x = self.norm1(x)
        
        # Feed forward
        x = x + self.dropout(self.feed_forward(x))
        x = self.norm2(x)
        
        return x

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
        
        # Initialize embeddings with Xavier uniform
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight, gain=1.0)
        nn.init.xavier_uniform_(self.item_embedding.weight, gain=1.0)
        
        # Modal encoders with residual connections
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_encoder = ModalEncoder(self.v_feat.shape[1], self.hidden_dim, self.feat_embed_dim, self.dropout)
            self.image_projector = nn.Linear(self.feat_embed_dim, self.embedding_dim)
            
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_encoder = ModalEncoder(self.t_feat.shape[1], self.hidden_dim, self.feat_embed_dim, self.dropout)
            self.text_projector = nn.Linear(self.feat_embed_dim, self.embedding_dim)
            
        # Graph attention layers
        self.mm_layers = nn.ModuleList([
            GraphAttentionLayer(self.embedding_dim, self.dropout)
            for _ in range(self.n_layers)
        ])
        
        self.ui_layers = nn.ModuleList([
            GraphAttentionLayer(self.embedding_dim, self.dropout)
            for _ in range(self.n_layers)
        ])
        
        # Gating mechanism for modal fusion
        if self.v_feat is not None and self.t_feat is not None:
            self.modal_gate = nn.Sequential(
                nn.Linear(self.embedding_dim * 2, 2),
                nn.Softmax(dim=-1)
            )
        
        # Load and process interaction data
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.edge_index = self.build_edges()
        self.mm_edge_index = None
        self.build_modal_graph()
        
        # Weight initialization for better gradient flow
        self.apply(self._init_weights)
        self.to(self.device)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=1.0)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
                
    def build_edges(self):
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
            
        if self.v_feat is not None:
            feats = F.normalize(self.v_feat, p=2, dim=1)
        elif self.t_feat is not None:
            feats = F.normalize(self.t_feat, p=2, dim=1)
            
        sim = torch.mm(feats, feats.t())
        values, indices = sim.topk(k=self.knn_k, dim=1)
        rows = torch.arange(feats.size(0), device=self.device).view(-1, 1).expand_as(indices)
        
        self.mm_edge_index = torch.stack([
            torch.cat([rows.reshape(-1), indices.reshape(-1)]),
            torch.cat([indices.reshape(-1), rows.reshape(-1)])
        ]).to(self.device)

    def forward(self):
        # Process modalities with gradient scaling
        img_emb = txt_emb = None
        
        if self.v_feat is not None:
            img_emb = self.image_encoder(self.image_embedding.weight)
            img_emb = self.image_projector(img_emb)
            
        if self.t_feat is not None:
            txt_emb = self.text_encoder(self.text_embedding.weight)
            txt_emb = self.text_projector(txt_emb)
        
        # Adaptive modal fusion with gating
        if img_emb is not None and txt_emb is not None:
            concat_emb = torch.cat([img_emb, txt_emb], dim=-1)
            gates = self.modal_gate(concat_emb)
            modal_emb = gates[:, 0:1] * img_emb + gates[:, 1:2] * txt_emb
        else:
            modal_emb = img_emb if img_emb is not None else txt_emb
        
        # Process user-item graph with residual connections
        x = torch.cat([self.user_embedding.weight, self.item_embedding.weight])
        all_embs = [x]
        
        for layer in self.ui_layers:
            if self.training:
                x = F.dropout(x, p=self.dropout, training=True)
            x = layer(x, self.edge_index)
            all_embs.append(x)
            
        # Attention-based aggregation of layers
        stacked_embs = torch.stack(all_embs, dim=1)
        alpha = F.softmax(torch.matmul(stacked_embs, stacked_embs.mean(1).unsqueeze(-1)), dim=1)
        x = (stacked_embs * alpha).sum(1)
        
        user_emb, item_emb = torch.split(x, [self.n_users, self.n_items])
        
        # Combine with modal embeddings using residual connection
        if modal_emb is not None:
            item_emb = item_emb + modal_emb
            
        return user_emb, item_emb, img_emb, txt_emb

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        
        user_emb, item_emb, img_emb, txt_emb = self.forward()
        
        u_emb = user_emb[users]
        pos_emb = item_emb[pos_items]
        neg_emb = item_emb[neg_items]
        
        # BPR loss with temperature scaling
        temperature = 0.2
        pos_scores = torch.sum(u_emb * pos_emb, dim=1) / temperature
        neg_scores = torch.sum(u_emb * neg_emb, dim=1) / temperature
        bpr_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        
        # Improved modal contrastive loss with hard negative mining
        modal_loss = 0.0
        if img_emb is not None and txt_emb is not None:
            pos_sim = F.cosine_similarity(img_emb[pos_items], txt_emb[pos_items])
            
            # Hard negative mining
            with torch.no_grad():
                neg_sims = F.cosine_similarity(
                    img_emb[pos_items].unsqueeze(1),
                    txt_emb[neg_items].unsqueeze(0),
                    dim=2
                )
                hardest_negs = neg_sims.max(dim=1)[0]
            
            modal_loss = -torch.mean(F.logsigmoid(pos_sim - hardest_negs))
        
        # L2 regularization with gradient clipping
        reg_loss = self.reg_weight * (
            torch.norm(u_emb, p=2) +
            torch.norm(pos_emb, p=2) +
            torch.norm(neg_emb, p=2)
        )
        
        # Weighted combination of losses
        total_loss = bpr_loss + 0.2 * modal_loss + reg_loss
        
        # Gradient scaling for better training stability
        if self.training:
            total_loss = total_loss * (1.0 / (1.0 + torch.exp(-total_loss.detach())))
        
        return total_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_emb, item_emb, _, _ = self.forward()
        scores = torch.matmul(user_emb[user], item_emb.transpose(0, 1))
        return scores