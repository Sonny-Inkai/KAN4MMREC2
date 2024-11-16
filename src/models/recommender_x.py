# RECOMMENDER_X.py
# coding: utf-8
# @email: your_email@example.com

"""
RECOMMENDER-X: Dynamic Context-Aware Multimodal Graph Neural Networks for Personalized Recommendation
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree

from common.abstract_recommender import GeneralRecommender

class RECOMMENDER_X(GeneralRecommender):
    def __init__(self, config, dataset):
        super(RECOMMENDER_X, self).__init__(config, dataset)

        self.embedding_dim = config["embedding_size"]
        self.feat_embed_dim = config["feat_embed_dim"]
        self.n_layers = config["n_layers"]  # Number of GNN layers
        self.time_decay = config["time_decay"]
        self.context_features = config["context_features"]
        self.device = config["device"]

        # Initialize embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        # Modality-specific embeddings
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)

        # Context-aware attention mechanism
        self.context_attention = ContextAttention(self.embedding_dim, len(self.context_features))

        # GNN layers
        self.gnn_layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.gnn_layers.append(DynamicGraphConvolution(self.embedding_dim))

        # Global-Local Fusion Network
        self.global_local_fusion = GlobalLocalFusion(self.embedding_dim)

        # Edge index and edge weight initialization
        self.edge_index, self.edge_weight = self.build_dynamic_graph(dataset)

    def build_dynamic_graph(self, dataset):
        # Build dynamic graph based on interactions with time decay
        interactions = dataset.inter_matrix(form='coo').astype(np.float32)
        rows = torch.tensor(interactions.row, dtype=torch.long)
        cols = torch.tensor(interactions.col + self.n_users, dtype=torch.long)
        edge_index = torch.stack([torch.cat([rows, cols]), torch.cat([cols, rows])], dim=0)

        # Time decay weights
        if 'timestamp' in dataset.df.columns:
            timestamps = dataset.df['timestamp'].values
            current_time = timestamps.max()
            time_deltas = current_time - timestamps
            edge_weight = torch.tensor(self.time_decay ** time_deltas, dtype=torch.float32)
            edge_weight = torch.cat([edge_weight, edge_weight])
        else:
            edge_weight = torch.ones(edge_index.size(1), dtype=torch.float32)

        return edge_index.to(self.device), edge_weight.to(self.device)

    def forward(self, user_ids, context=None):
        # Embeddings
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight

        # Modality-specific embeddings
        if self.v_feat is not None:
            image_feat = self.image_trs(self.image_embedding.weight)
            item_emb = item_emb + image_feat
        if self.t_feat is not None:
            text_feat = self.text_trs(self.text_embedding.weight)
            item_emb = item_emb + text_feat

        # Context-aware attention
        if context is not None:
            context_emb = self.context_attention(context)
            user_emb[user_ids] = user_emb[user_ids] + context_emb

        x = torch.cat([user_emb, item_emb], dim=0)

        # GNN propagation
        for gnn in self.gnn_layers:
            x = gnn(x, self.edge_index, self.edge_weight)

        user_out = x[:self.n_users]
        item_out = x[self.n_users:]

        # Global-Local Fusion
        user_global = self.user_embedding(user_ids)
        user_final = self.global_local_fusion(user_global, user_out[user_ids])

        return user_final, item_out

    def calculate_loss(self, interaction):
        user = interaction[0]
        pos_item = interaction[1]
        neg_item = interaction[2]
        context = interaction[3] if len(interaction) > 3 else None

        user_final, item_out = self.forward(user, context)
        pos_item_emb = item_out[pos_item]
        neg_item_emb = item_out[neg_item]

        pos_scores = torch.sum(user_final * pos_item_emb, dim=1)
        neg_scores = torch.sum(user_final * neg_item_emb, dim=1)

        mf_loss = F.softplus(neg_scores - pos_scores).mean()
        reg_loss = (user_final.norm(2).pow(2) + pos_item_emb.norm(2).pow(2) + neg_item_emb.norm(2).pow(2)) / 2
        loss = mf_loss + self.config["reg_weight"] * reg_loss

        return loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        context = interaction[1] if len(interaction) > 1 else None

        user_final, item_out = self.forward(user, context)
        scores = torch.matmul(user_final, item_out.t())

        return scores

class DynamicGraphConvolution(MessagePassing):
    def __init__(self, embedding_dim):
        super(DynamicGraphConvolution, self).__init__(aggr='add')
        self.embedding_dim = embedding_dim

    def forward(self, x, edge_index, edge_weight):
        # Compute normalization
        row, col = edge_index
        deg = degree(row, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col] * edge_weight

        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

class ContextAttention(nn.Module):
    def __init__(self, embedding_dim, context_dim):
        super(ContextAttention, self).__init__()
        self.attention = nn.Linear(context_dim, embedding_dim)

    def forward(self, context):
        # context: (batch_size, context_dim)
        attention_weights = torch.sigmoid(self.attention(context))
        return attention_weights

class GlobalLocalFusion(nn.Module):
    def __init__(self, embedding_dim):
        super(GlobalLocalFusion, self).__init__()
        self.gate = nn.Linear(embedding_dim * 2, embedding_dim)

    def forward(self, global_emb, local_emb):
        combined = torch.cat([global_emb, local_emb], dim=1)
        gate = torch.sigmoid(self.gate(combined))
        output = gate * global_emb + (1 - gate) * local_emb
        return output

# If you have any specific configuration or dataset attributes, make sure to adjust the code accordingly.
