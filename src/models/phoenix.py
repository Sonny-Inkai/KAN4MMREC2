# phoenix.py
# coding: utf-8

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from common.abstract_recommender import GeneralRecommender
from torch_geometric.nn import GCNConv
from torch_geometric.utils import add_self_loops, degree

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(dim, num_heads)
        self.layer_norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        x = x.unsqueeze(0)
        attn_output, _ = self.attention(x, x, x)
        attn_output = self.layer_norm(attn_output + x)  # Add residual connection
        return attn_output.squeeze(0)

class PHOENIX(GeneralRecommender):
    def __init__(self, config, dataset):
        super(PHOENIX, self).__init__(config, dataset)

        # Enhanced model parameters
        self.embedding_dim = config['embedding_size']  # Increased from 64
        self.feat_embed_dim = config['feat_embed_dim']  # Increased from 64
        self.n_layers = config['n_layers']  # Increased from 2
        self.dropout = config['dropout']  # Reduced from 0.8
        self.alpha = config['alpha']
        self.beta = config['beta']
        self.k = config['knn_k']
        self.tau = config['tau']  # Temperature parameter
        self.weight_decay = config['weight_decay']

        self.device = config['device']
        self.n_users = self.n_users
        self.n_items = self.n_items

        # Initialize embeddings with Xavier uniform
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim).to(self.device)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim).to(self.device)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        # Initialize modality-specific components
        self.modality_embeddings = {}
        self.modality_transforms = {}
        self.modality_attention = {}
        self.modalities = []

        # Handle visual features
        if hasattr(self, 'v_feat') and self.v_feat is not None:
            self.modalities.append('visual')
            self.v_feat = F.normalize(self.v_feat.to(self.device), dim=1)
            self.modality_embeddings['visual'] = self.v_feat
            self.modality_transforms['visual'] = nn.Sequential(
                nn.Linear(self.v_feat.shape[1], self.feat_embed_dim),
                nn.LayerNorm(self.feat_embed_dim),
                nn.GELU()
            ).to(self.device)
            self.modality_attention['visual'] = MultiHeadAttention(self.feat_embed_dim).to(self.device)

        # Handle textual features
        if hasattr(self, 't_feat') and self.t_feat is not None:
            self.modalities.append('textual')
            self.t_feat = F.normalize(self.t_feat.to(self.device), dim=1)
            self.modality_embeddings['textual'] = self.t_feat
            self.modality_transforms['textual'] = nn.Sequential(
                nn.Linear(self.t_feat.shape[1], self.feat_embed_dim),
                nn.LayerNorm(self.feat_embed_dim),
                nn.GELU()
            ).to(self.device)
            self.modality_attention['textual'] = MultiHeadAttention(self.feat_embed_dim).to(self.device)

        # Enhanced GCN layers with residual connections
        self.gcn_layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.gcn_layers.append(nn.ModuleDict({
                'gcn': GCNConv(self.embedding_dim, self.embedding_dim).to(self.device),
                'norm': nn.LayerNorm(self.embedding_dim).to(self.device),
                'attention': MultiHeadAttention(self.embedding_dim).to(self.device)
            }))

        # Enhanced gating mechanism
        total_feat_dim = self.embedding_dim + len(self.modalities) * self.feat_embed_dim
        self.gate_layer = nn.Sequential(
            nn.Linear(total_feat_dim, 256),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, len(self.modalities) + 1)
        ).to(self.device)

        # Enhanced projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
            nn.GELU(),
            nn.Linear(self.embedding_dim, self.embedding_dim)
        ).to(self.device)

        # Final attention layer
        self.final_attention = MultiHeadAttention(self.embedding_dim).to(self.device)

        # Build graph
        self.build_graph(dataset)

    def build_graph(self, dataset):
        # Enhanced graph building with self-loops
        interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        user_np = interaction_matrix.row
        item_np = interaction_matrix.col + self.n_users
        edge_index_ui = np.array([user_np, item_np])
        edge_index_iu = np.array([item_np, user_np])
        edge_index = np.concatenate([edge_index_ui, edge_index_iu], axis=1)
        self.edge_index = torch.tensor(edge_index, dtype=torch.long).to(self.device)
        
        # Add self-loops with learned weights
        self.edge_index, _ = add_self_loops(self.edge_index)

    def forward(self):
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight

        x = torch.cat([user_emb, item_emb], dim=0)
        residual = x

        # Enhanced GCN forward pass with residual connections
        for i, layer in enumerate(self.gcn_layers):
            # GCN operation
            x_conv = layer['gcn'](x, self.edge_index)
            # Attention
            x_attn = layer['attention'](x_conv)
            # Residual connection and normalization
            x = layer['norm'](x_attn + residual)
            x = F.gelu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            residual = x

        user_gcn_emb, item_gcn_emb = x[:self.n_users], x[self.n_users:]

        # Enhanced modal fusion
        modality_embeddings = []
        for modality in self.modalities:
            feat = self.modality_embeddings[modality]
            transform = self.modality_transforms[modality]
            attention = self.modality_attention[modality]
            
            emb = transform(feat)
            emb = attention(emb)
            modality_embeddings.append(emb)

        if modality_embeddings:
            # Enhanced gating mechanism
            item_all_emb = torch.cat([item_gcn_emb] + modality_embeddings, dim=1)
            gate_values = self.gate_layer(item_all_emb)
            gate_weights = F.gelu(gate_values)
            gate_weights = F.dropout(gate_weights, p=self.dropout, training=self.training)
            gate_weights = torch.softmax(gate_weights, dim=1)
            
            # Weighted fusion with residual connection
            item_final_emb = gate_weights[:, 0].unsqueeze(1) * item_gcn_emb
            for idx, modality_emb in enumerate(modality_embeddings):
                item_final_emb += gate_weights[:, idx+1].unsqueeze(1) * modality_emb
            
            # Add residual connection
            item_final_emb = item_final_emb + item_gcn_emb
        else:
            item_final_emb = item_gcn_emb

        # Final attention and normalization
        item_final_emb = self.final_attention(item_final_emb)
        
        return user_gcn_emb, item_final_emb

    def calculate_loss(self, interaction):
        users = interaction[0].to(self.device)
        pos_items = interaction[1].to(self.device)
        neg_items = interaction[2].to(self.device)

        user_gcn_emb, item_final_emb = self.forward()

        user_emb = user_gcn_emb[users]
        pos_item_emb = item_final_emb[pos_items]
        neg_item_emb = item_final_emb[neg_items]

        # Enhanced BPR Loss with margin
        margin = 0.5
        pos_scores = torch.sum(user_emb * pos_item_emb, dim=-1)
        neg_scores = torch.sum(user_emb * neg_item_emb, dim=-1)
        bpr_loss = torch.mean(F.softplus(-(pos_scores - neg_scores) + margin))

        # Enhanced SSL Loss
        ssl_loss = self.calculate_ssl_loss(user_emb, pos_item_emb)
        
        # L2 Regularization
        l2_reg = torch.norm(user_emb) + torch.norm(pos_item_emb) + torch.norm(neg_item_emb)
        
        # Combined loss with adaptive weights
        loss = bpr_loss + self.beta * ssl_loss + self.weight_decay * l2_reg
        return loss

    def calculate_ssl_loss(self, user_emb, item_emb):
        # Enhanced SSL loss calculation
        user_emb_aug = self.projection_head(F.dropout(user_emb, p=self.dropout))
        item_emb_aug = self.projection_head(F.dropout(item_emb, p=self.dropout))

        user_emb_aug = F.normalize(user_emb_aug, dim=1)
        item_emb_aug = F.normalize(item_emb_aug, dim=1)

        sim_matrix = torch.matmul(user_emb_aug, item_emb_aug.t()) / self.tau
        batch_size = user_emb.size(0)
        labels = torch.arange(batch_size).to(self.device)
        
        loss_user = F.cross_entropy(sim_matrix, labels)
        loss_item = F.cross_entropy(sim_matrix.t(), labels)
        
        return (loss_user + loss_item) / 2

    def full_sort_predict(self, interaction):
        user = interaction[0].to(self.device)
        user_gcn_emb, item_final_emb = self.forward()
        user_emb = user_gcn_emb[user]
        scores = torch.matmul(user_emb, item_final_emb.t())
        return scores