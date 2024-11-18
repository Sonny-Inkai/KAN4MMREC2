# coding: utf-8
# @email: your_email@example.com

"""
PHOENIX: A Heterogeneous Graph Attention Network for Multimodal Recommendation
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender
from torch_geometric.nn import GATConv
from torch_sparse import SparseTensor
from torch_geometric.utils import dropout_adj


class PHOENIX(GeneralRecommender):
    def __init__(self, config, dataset):
        super(PHOENIX, self).__init__(config, dataset)

        # Model hyperparameters
        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_size']
        self.num_layers = config['num_layers']
        self.num_heads = config['num_heads']
        self.dropout_rate = config['dropout_rate']
        self.alpha = config['alpha']  # LeakyReLU negative slope

        # Device configuration
        self.device = config['device']

        # Embedding layers
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        # Modality-specific transformations
        self.v_feat_embed = None
        self.t_feat_embed = None

        if self.v_feat is not None:
            self.v_linear = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)
            nn.init.xavier_uniform_(self.v_linear.weight)
            self.v_feat_embed = self.v_linear(self.v_feat).to(self.device)

        if self.t_feat is not None:
            self.t_linear = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)
            nn.init.xavier_uniform_(self.t_linear.weight)
            self.t_feat_embed = self.t_linear(self.t_feat).to(self.device)

        # Heterogeneous Graph Attention Networks for each modality
        self.gat_layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.gat_layers.append(
                GATConv(
                    in_channels=self.embedding_dim,
                    out_channels=self.embedding_dim // self.num_heads,
                    heads=self.num_heads,
                    dropout=self.dropout_rate,
                    concat=True,
                    negative_slope=self.alpha,
                )
            )

        # Modality attention fusion layer
        self.modality_attention = nn.Linear(self.embedding_dim * 2, self.embedding_dim)
        nn.init.xavier_uniform_(self.modality_attention.weight)

        # Final prediction layer
        self.predictor = nn.Linear(self.embedding_dim, self.embedding_dim)
        nn.init.xavier_uniform_(self.predictor.weight)

        # Loss function
        self.bce_loss = nn.BCEWithLogitsLoss()

        # Prepare adjacency matrices
        self.prepare_graph(dataset)

    def prepare_graph(self, dataset):
        # Build user-item interaction graph
        interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        edge_index = np.vstack((interaction_matrix.row, interaction_matrix.col + self.n_users))
        edge_index = torch.tensor(edge_index, dtype=torch.long).to(self.device)

        # Build item-item similarity graph based on multimodal features
        item_feat = []
        if self.v_feat_embed is not None:
            item_feat.append(self.v_feat_embed)
        if self.t_feat_embed is not None:
            item_feat.append(self.t_feat_embed)
        if item_feat:
            item_features = torch.cat(item_feat, dim=1)
            item_sim = F.cosine_similarity(
                item_features.unsqueeze(1), item_features.unsqueeze(0), dim=-1
            )
            # Construct item-item edges
            k = 10  # Number of nearest neighbors
            _, top_k_indices = torch.topk(item_sim, k=k, dim=-1)
            item_indices = torch.arange(self.n_items).unsqueeze(1).expand(-1, k).flatten()
            neighbor_indices = top_k_indices.flatten()
            item_edge_index = torch.stack([item_indices, neighbor_indices]) + self.n_users
            edge_index = torch.cat([edge_index, item_edge_index], dim=1)
        else:
            item_features = None

        # Convert to SparseTensor for efficient processing
        self.edge_index = edge_index
        self.adj_t = SparseTensor(row=edge_index[0], col=edge_index[1], sparse_sizes=(self.n_users + self.n_items, self.n_users + self.n_items)).to(self.device)

    def forward(self, users, pos_items, neg_items):
        # Generate node embeddings
        x = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        all_embeddings = [x]

        for gat_layer in self.gat_layers:
            x = gat_layer(x, self.edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
            all_embeddings.append(x)

        x = torch.mean(torch.stack(all_embeddings, dim=1), dim=1)
        user_embeddings, item_embeddings = x[:self.n_users], x[self.n_users:]

        # Modality-specific item embeddings
        if self.v_feat_embed is not None and self.t_feat_embed is not None:
            modality_embeddings = torch.cat([self.v_feat_embed, self.t_feat_embed], dim=1)
            modality_embeddings = self.modality_attention(modality_embeddings)
            item_embeddings = item_embeddings + modality_embeddings

        elif self.v_feat_embed is not None:
            item_embeddings = item_embeddings + self.v_feat_embed

        elif self.t_feat_embed is not None:
            item_embeddings = item_embeddings + self.t_feat_embed

        # Get user and item embeddings for the batch
        u_embeddings = user_embeddings[users]
        pos_i_embeddings = item_embeddings[pos_items]
        neg_i_embeddings = item_embeddings[neg_items]

        return u_embeddings, pos_i_embeddings, neg_i_embeddings

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        u_embeddings, pos_i_embeddings, neg_i_embeddings = self.forward(users, pos_items, neg_items)

        # BPR loss
        pos_scores = torch.mul(u_embeddings, pos_i_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_i_embeddings).sum(dim=1)

        bpr_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))

        # Regularization
        reg_loss = (
            u_embeddings.norm(2).pow(2) +
            pos_i_embeddings.norm(2).pow(2) +
            neg_i_embeddings.norm(2).pow(2)
        ) / 2

        # Total loss
        loss = bpr_loss + self.config['reg_weight'] * reg_loss

        # Self-supervised contrastive loss
        ssl_loss = self.calculate_ssl_loss(u_embeddings, pos_i_embeddings)
        loss += self.config['ssl_weight'] * ssl_loss

        return loss

    def calculate_ssl_loss(self, u_embeddings, pos_i_embeddings):
        # Generate two augmented views
        def augment_embeddings(embeddings):
            noise = F.dropout(embeddings, p=self.config['ssl_dropout'])
            return embeddings + noise

        u_embeddings_aug1 = augment_embeddings(u_embeddings)
        u_embeddings_aug2 = augment_embeddings(u_embeddings)

        pos_i_embeddings_aug1 = augment_embeddings(pos_i_embeddings)
        pos_i_embeddings_aug2 = augment_embeddings(pos_i_embeddings)

        # Compute contrastive loss
        batch_size = u_embeddings.size(0)
        temperature = self.config['ssl_temp']

        z1 = F.normalize(u_embeddings_aug1, dim=1)
        z2 = F.normalize(u_embeddings_aug2, dim=1)

        pos_scores = torch.exp(torch.sum(z1 * z2, dim=1) / temperature)
        total_scores = torch.exp(torch.matmul(z1, z2.t()) / temperature).sum(dim=1)

        ssl_loss_u = -torch.log(pos_scores / total_scores).mean()

        z1 = F.normalize(pos_i_embeddings_aug1, dim=1)
        z2 = F.normalize(pos_i_embeddings_aug2, dim=1)

        pos_scores = torch.exp(torch.sum(z1 * z2, dim=1) / temperature)
        total_scores = torch.exp(torch.matmul(z1, z2.t()) / temperature).sum(dim=1)

        ssl_loss_i = -torch.log(pos_scores / total_scores).mean()

        return ssl_loss_u + ssl_loss_i

    def full_sort_predict(self, interaction):
        users = interaction[0]

        # Generate node embeddings
        x = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        all_embeddings = [x]

        for gat_layer in self.gat_layers:
            x = gat_layer(x, self.edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
            all_embeddings.append(x)

        x = torch.mean(torch.stack(all_embeddings, dim=1), dim=1)
        user_embeddings, item_embeddings = x[:self.n_users], x[self.n_users:]

        # Modality-specific item embeddings
        if self.v_feat_embed is not None and self.t_feat_embed is not None:
            modality_embeddings = torch.cat([self.v_feat_embed, self.t_feat_embed], dim=1)
            modality_embeddings = self.modality_attention(modality_embeddings)
            item_embeddings = item_embeddings + modality_embeddings

        elif self.v_feat_embed is not None:
            item_embeddings = item_embeddings + self.v_feat_embed

        elif self.t_feat_embed is not None:
            item_embeddings = item_embeddings + self.t_feat_embed

        u_embeddings = user_embeddings[users]

        # Compute scores
        scores = torch.matmul(u_embeddings, item_embeddings.t())
        return scores

