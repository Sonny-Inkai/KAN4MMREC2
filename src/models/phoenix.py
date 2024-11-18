# phoenix_plus.py
# coding: utf-8

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from common.abstract_recommender import GeneralRecommender
from torch_geometric.nn import GCNConv
from torch_geometric.utils import add_self_loops

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
        
        # Model parameters
        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.n_layers = config['n_layers']
        self.dropout = config['dropout']
        self.alpha = config['alpha']
        self.beta = config['beta']
        self.tau = config['tau']
        self.weight_decay = config['weight_decay']
        
        self.device = config['device']
        self.n_users = self.n_users
        self.n_items = self.n_items

        # Initialize embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim).to(self.device)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim).to(self.device)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        # Attention for multi-modal features
        self.modality_attention = MultiHeadAttention(self.feat_embed_dim).to(self.device)
        
        # GCN layers
        self.gcn_layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.gcn_layers.append(GCNConv(self.embedding_dim, self.embedding_dim).to(self.device))

        # Final attention layer
        self.final_attention = MultiHeadAttention(self.embedding_dim).to(self.device)

        # Build graph
        self.build_graph(dataset)

    def build_graph(self, dataset):
        interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        user_np = interaction_matrix.row
        item_np = interaction_matrix.col + self.n_users
        edge_index = np.array([user_np, item_np])
        self.edge_index = torch.tensor(edge_index, dtype=torch.long).to(self.device)
        self.edge_index, _ = add_self_loops(self.edge_index)

    def forward(self):
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight

        x = torch.cat([user_emb, item_emb], dim=0)

        # GCN forward pass
        for layer in self.gcn_layers:
            x = layer(x, self.edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        user_gcn_emb, item_gcn_emb = x[:self.n_users], x[self.n_users:]
        
        # Attention-based feature aggregation
        combined_emb = self.modality_attention(item_gcn_emb)
        
        # Final attention layer
        combined_emb = self.final_attention(combined_emb)

        return user_gcn_emb, combined_emb

    def calculate_loss(self, interaction):
        users = interaction[0].to(self.device)
        pos_items = interaction[1].to(self.device)
        neg_items = interaction[2].to(self.device)

        user_gcn_emb, item_final_emb = self.forward()

        user_emb = user_gcn_emb[users]
        pos_item_emb = item_final_emb[pos_items]
        neg_item_emb = item_final_emb[neg_items]

        # BPR Loss
        pos_scores = torch.sum(user_emb * pos_item_emb, dim=-1)
        neg_scores = torch.sum(user_emb * neg_item_emb, dim=-1)
        bpr_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))

        # L2 Regularization
        l2_reg = self.weight_decay * (torch.norm(user_emb) + torch.norm(pos_item_emb) + torch.norm(neg_item_emb))

        # Combined loss
        loss = bpr_loss + l2_reg
        return loss

    def full_sort_predict(self, interaction):
        user = interaction[0].to(self.device)
        user_gcn_emb, item_final_emb = self.forward()
        user_emb = user_gcn_emb[user]
        scores = torch.matmul(user_emb, item_final_emb.t())
        return scores