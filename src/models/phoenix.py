# phoenix.py
# coding: utf-8

"""
PHOENIX: Unified Graph Collaborative Filtering with Adaptive Modal Fusion for Multimodal Recommendation
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from common.abstract_recommender import GeneralRecommender
from torch_geometric.nn import GCNConv
from torch_geometric.utils import add_self_loops, degree

class PHOENIX(GeneralRecommender):
    def __init__(self, config, dataset):
        super(PHOENIX, self).__init__(config, dataset)

        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.n_layers = config['n_layers']
        self.dropout = config['dropout']
        self.alpha = config['alpha']
        self.beta = config['beta']
        self.k = config['knn_k']

        self.device = config['device']
        self.n_users = self.n_users
        self.n_items = self.n_items

        # Initialize User and Item Embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim).to(self.device)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim).to(self.device)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        # Initialize Modality-specific Embeddings
        self.modality_embeddings = {}
        self.modality_transforms = {}
        self.modalities = []

        if hasattr(self, 'v_feat') and self.v_feat is not None:
            self.modalities.append('visual')
            self.v_feat = F.normalize(self.v_feat.to(self.device), dim=1)
            self.modality_embeddings['visual'] = self.v_feat
            self.modality_transforms['visual'] = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim).to(self.device)
            nn.init.xavier_uniform_(self.modality_transforms['visual'].weight)

        if hasattr(self, 't_feat') and self.t_feat is not None:
            self.modalities.append('textual')
            self.t_feat = F.normalize(self.t_feat.to(self.device), dim=1)
            self.modality_embeddings['textual'] = self.t_feat
            self.modality_transforms['textual'] = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim).to(self.device)
            nn.init.xavier_uniform_(self.modality_transforms['textual'].weight)

        # GCN Layers for Unified Graph
        self.gcn_layers = nn.ModuleList([
            GCNConv(self.embedding_dim, self.embedding_dim).to(self.device)
            for _ in range(self.n_layers)
        ])

        # Gating Mechanism for Modal Fusion
        total_feat_dim = self.embedding_dim + len(self.modalities) * self.feat_embed_dim
        self.gate_layer = nn.Linear(total_feat_dim, len(self.modalities) + 1).to(self.device)
        nn.init.xavier_uniform_(self.gate_layer.weight)

        # Contrastive Learning Projection Head
        self.projection_head = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, self.embedding_dim)
        ).to(self.device)
        nn.init.xavier_uniform_(self.projection_head[0].weight)
        nn.init.xavier_uniform_(self.projection_head[2].weight)

        # Temperature parameter for SSL
        self.tau = 0.2

        # Build Graph
        self.build_graph(dataset)

    def build_graph(self, dataset):
        interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        user_np = interaction_matrix.row
        item_np = interaction_matrix.col + self.n_users
        ratings = interaction_matrix.data
        edge_index_ui = np.array([user_np, item_np])
        edge_index_iu = np.array([item_np, user_np])
        edge_index = np.concatenate([edge_index_ui, edge_index_iu], axis=1)
        self.edge_index = torch.tensor(edge_index, dtype=torch.long).to(self.device)

    def forward(self):
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight

        x = torch.cat([user_emb, item_emb], dim=0)
        
        for gcn in self.gcn_layers:
            x = gcn(x, self.edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        user_gcn_emb, item_gcn_emb = x[:self.n_users], x[self.n_users:]

        modality_embeddings = []
        for modality in self.modalities:
            feat = self.modality_embeddings[modality]
            transform = self.modality_transforms[modality]
            emb = transform(feat)
            modality_embeddings.append(emb)

        if modality_embeddings:
            item_all_emb = torch.cat([item_gcn_emb] + modality_embeddings, dim=1)
            gate_values = self.gate_layer(item_all_emb)
            gate_weights = torch.softmax(gate_values, dim=1)
            g_user_item = gate_weights[:, 0].unsqueeze(1)
            g_modalities = torch.chunk(gate_weights[:, 1:], len(self.modalities), dim=1)
            
            item_final_emb = g_user_item * item_gcn_emb
            for idx, modality_emb in enumerate(modality_embeddings):
                item_final_emb += g_modalities[idx] * modality_emb
        else:
            item_final_emb = item_gcn_emb

        return user_gcn_emb, item_final_emb

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

        # SSL Loss for batch samples only
        ssl_loss = self.calculate_ssl_loss(user_emb, pos_item_emb)

        loss = bpr_loss + self.beta * ssl_loss

        return loss

    def calculate_ssl_loss(self, user_emb, item_emb):
        """Calculate SSL loss for the batch samples only"""
        # Generate augmented embeddings
        user_emb_aug = self.projection_head(F.dropout(user_emb, p=self.dropout))
        item_emb_aug = self.projection_head(F.dropout(item_emb, p=self.dropout))

        # Normalize embeddings
        user_emb_aug = F.normalize(user_emb_aug, dim=1)
        item_emb_aug = F.normalize(item_emb_aug, dim=1)

        # Compute similarity matrix
        sim_matrix = torch.matmul(user_emb_aug, item_emb_aug.t()) / self.tau
        
        # Create labels for positive pairs (diagonal elements should be positive pairs)
        batch_size = user_emb.size(0)
        labels = torch.arange(batch_size).to(self.device)
        
        # Calculate NCE loss from both directions
        loss_user = F.cross_entropy(sim_matrix, labels)
        loss_item = F.cross_entropy(sim_matrix.t(), labels)
        
        ssl_loss = (loss_user + loss_item) / 2
        return ssl_loss

    def full_sort_predict(self, interaction):
        user = interaction[0].to(self.device)
        user_gcn_emb, item_final_emb = self.forward()
        user_emb = user_gcn_emb[user]
        scores = torch.matmul(user_emb, item_final_emb.t())
        return scores