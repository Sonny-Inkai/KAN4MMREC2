# coding: utf-8
# @email: your_email@example.com

"""
PHOENIX: Hierarchical Graph Neural Network with Dynamic Modality Interaction for Multimodal Recommendation
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp

from common.abstract_recommender import GeneralRecommender
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops, degree

class PHOENIX(GeneralRecommender):
    def __init__(self, config, dataset):
        super(PHOENIX, self).__init__(config, dataset)

        # Model Hyperparameters
        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.n_layers = config['n_layers']
        self.dropout = config['dropout']
        self.alpha = config['alpha']  # For balancing losses

        self.n_users = self.n_users
        self.n_items = self.n_items

        # Initialize User and Item Embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        # Initialize Modality-specific Embeddings
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.v_transform = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)
            nn.init.xavier_uniform_(self.v_transform.weight)
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.t_transform = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)
            nn.init.xavier_uniform_(self.t_transform.weight)

        # GCN Layers for User-Item Graph
        self.ui_gcn_layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.ui_gcn_layers.append(GCNConv(self.embedding_dim, self.embedding_dim))

        # GCN Layers for Modality-specific Item-Item Graphs
        self.v_gcn_layers = nn.ModuleList()
        self.t_gcn_layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.v_gcn_layers.append(GCNConv(self.feat_embed_dim, self.feat_embed_dim))
            self.t_gcn_layers.append(GCNConv(self.feat_embed_dim, self.feat_embed_dim))

        # Attention Mechanism
        total_feat_dim = self.embedding_dim
        if self.v_feat is not None:
            total_feat_dim += self.feat_embed_dim
        if self.t_feat is not None:
            total_feat_dim += self.feat_embed_dim
        self.attention_layer = nn.Linear(total_feat_dim, 1)
        nn.init.xavier_uniform_(self.attention_layer.weight)

        # Contrastive Loss Function
        self.ssl_criterion = nn.BCEWithLogitsLoss()

        # Build Graphs
        self.build_graph(dataset)

    def build_graph(self, dataset):
        # Build User-Item Interaction Graph
        interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        user_np = interaction_matrix.row
        item_np = interaction_matrix.col + self.n_users  # Offset item indices
        ratings = interaction_matrix.data
        edge_index = np.array([np.concatenate([user_np, item_np]),
                               np.concatenate([item_np, user_np])])
        edge_index = torch.tensor(edge_index, dtype=torch.long).to(self.device)
        self.ui_edge_index = edge_index

        # Build Modality-specific Item-Item Graphs
        self.item_num = self.n_items
        self.v_edge_index = None
        self.t_edge_index = None

        if self.v_feat is not None:
            self.v_edge_index = self.build_knn_graph(self.v_feat)

        if self.t_feat is not None:
            self.t_edge_index = self.build_knn_graph(self.t_feat)

    def build_knn_graph(self, features, k=20):
        # Normalize features
        features = F.normalize(features, p=2, dim=1)
        # Compute cosine similarity
        sim_matrix = torch.mm(features, features.t())
        # Get top k neighbors
        _, knn_indices = torch.topk(sim_matrix, k=k+1, dim=-1)  # +1 for self-loop
        knn_indices = knn_indices[:, 1:]  # Exclude self-loop
        # Build edge index
        row_indices = torch.arange(self.item_num, device=self.device).unsqueeze(1).expand(-1, k).flatten()
        col_indices = knn_indices.flatten()
        edge_index = torch.stack([row_indices, col_indices], dim=0)
        # Ensure edge_index is on the correct device (optional)
        edge_index = edge_index.to(self.device)
        return edge_index


    def forward(self):
        # User-Item Graph Embedding Propagation
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight
        x = torch.cat([user_emb, item_emb], dim=0)
        for gcn in self.ui_gcn_layers:
            x = gcn(x, self.ui_edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        user_gcn_emb, item_gcn_emb = x[:self.n_users], x[self.n_users:]

        # Modality-specific Item Embeddings
        modality_embeddings = []
        if self.v_feat is not None:
            v_emb = self.v_transform(self.image_embedding.weight)
            for gcn in self.v_gcn_layers:
                v_emb = gcn(v_emb, self.v_edge_index)
                v_emb = F.relu(v_emb)
                v_emb = F.dropout(v_emb, p=self.dropout, training=self.training)
            modality_embeddings.append(v_emb)

        if self.t_feat is not None:
            t_emb = self.t_transform(self.text_embedding.weight)
            for gcn in self.t_gcn_layers:
                t_emb = gcn(t_emb, self.t_edge_index)
                t_emb = F.relu(t_emb)
                t_emb = F.dropout(t_emb, p=self.dropout, training=self.training)
            modality_embeddings.append(t_emb)

        # Combine Modality Embeddings
        if modality_embeddings:
            item_all_emb = torch.cat([item_gcn_emb] + modality_embeddings, dim=1)
        else:
            item_all_emb = item_gcn_emb

        # Attention Mechanism
        attention_scores = self.attention_layer(item_all_emb)
        attention_weights = torch.sigmoid(attention_scores)
        item_final_emb = item_all_emb * attention_weights

        # Return final embeddings
        return user_gcn_emb, item_final_emb

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        user_gcn_emb, item_final_emb = self.forward()

        # Embeddings for users and items
        user_emb = user_gcn_emb[users]
        pos_item_emb = item_final_emb[pos_items]
        neg_item_emb = item_final_emb[neg_items]

        # BPR Loss
        pos_scores = torch.sum(user_emb * pos_item_emb, dim=-1)
        neg_scores = torch.sum(user_emb * neg_item_emb, dim=-1)
        bpr_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))

        # Contrastive Loss (e.g., between modalities)
        contrastive_loss = self.compute_contrastive_loss(item_final_emb)

        # Total Loss
        loss = bpr_loss + self.alpha * contrastive_loss

        return loss

    def compute_contrastive_loss(self, item_final_emb):
        # Implement contrastive loss between modalities or between views
        # For simplicity, let's assume we have modality embeddings
        losses = []
        if self.v_feat is not None and self.t_feat is not None:
            v_emb = self.v_transform(self.v_feat)
            t_emb = self.t_transform(self.t_feat)
            v_emb = F.normalize(v_emb, dim=1)
            t_emb = F.normalize(t_emb, dim=1)
            logits = torch.mm(v_emb, t_emb.t())
            labels = torch.arange(self.n_items).to(self.device)
            contrastive_loss = F.cross_entropy(logits, labels)
            losses.append(contrastive_loss)
        if losses:
            return sum(losses) / len(losses)
        else:
            return 0.0

    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_gcn_emb, item_final_emb = self.forward()
        user_emb = user_gcn_emb[user]
        scores = torch.matmul(user_emb, item_final_emb.t())
        return scores

