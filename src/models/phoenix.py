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
from torch_geometric.utils import add_self_loops, remove_self_loops, to_undirected
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse


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
        self.reg_weight = config['reg_weight']
        self.ssl_weight = config['ssl_weight']
        self.ssl_temp = config['ssl_temp']
        self.ssl_dropout = config['ssl_dropout']

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
                    out_channels=self.embedding_dim,
                    heads=self.num_heads,
                    concat=False,
                    dropout=self.dropout_rate,
                    negative_slope=self.alpha,
                )
            )

        # Modality attention fusion layer
        self.modality_attention = nn.Linear(self.embedding_dim * 2, self.embedding_dim)
        nn.init.xavier_uniform_(self.modality_attention.weight)

        # Loss function
        self.bce_loss = nn.BCEWithLogitsLoss()

        # Prepare adjacency matrices
        self.prepare_graph(dataset)

    def prepare_graph(self, dataset):
        # Build user-item interaction graph
        interaction_matrix = dataset.inter_matrix(form='coo').astype(np.int64)
        user_indices = torch.from_numpy(interaction_matrix.row).to(self.device)
        item_indices = torch.from_numpy(interaction_matrix.col).to(self.device) + self.n_users
        edge_index_ui = torch.stack([user_indices, item_indices], dim=0)

        # Build item-item similarity graph based on multimodal features
        item_feat_list = []
        if self.v_feat_embed is not None:
            item_feat_list.append(self.v_feat_embed)
        if self.t_feat_embed is not None:
            item_feat_list.append(self.t_feat_embed)

        if item_feat_list:
            item_features = torch.cat(item_feat_list, dim=1)
            # Compute cosine similarity
            item_norm = item_features / item_features.norm(dim=1, keepdim=True)
            similarity_matrix = torch.mm(item_norm, item_norm.t())

            # Build adjacency matrix
            k = 10  # Number of nearest neighbors
            top_k_values, top_k_indices = torch.topk(similarity_matrix, k=k, dim=-1)
            item_indices_row = torch.arange(self.n_items).unsqueeze(1).expand(-1, k).flatten().to(self.device)
            item_indices_col = top_k_indices.flatten().to(self.device)
            edge_index_ii = torch.stack([item_indices_row + self.n_users, item_indices_col + self.n_users], dim=0)
        else:
            edge_index_ii = torch.empty((2, 0), dtype=torch.long).to(self.device)

        # Combine edge indices
        edge_index = torch.cat([edge_index_ui, edge_index_ui.flip([0]), edge_index_ii], dim=1)
        edge_index = to_undirected(edge_index)

        # Remove self-loops
        edge_index, _ = remove_self_loops(edge_index)
        edge_index = add_self_loops(edge_index, num_nodes=self.n_users + self.n_items)[0]

        self.edge_index = edge_index

    def forward(self):
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

        return user_embeddings, item_embeddings

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        user_embeddings, item_embeddings = self.forward()
        u_embeddings = user_embeddings[users]
        pos_i_embeddings = item_embeddings[pos_items]
        neg_i_embeddings = item_embeddings[neg_items]

        # BPR loss
        pos_scores = torch.mul(u_embeddings, pos_i_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_i_embeddings).sum(dim=1)

        bpr_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))

        # Regularization
        reg_loss = (
            u_embeddings.norm(2).pow(2) +
            pos_i_embeddings.norm(2).pow(2) +
            neg_i_embeddings.norm(2).pow(2)
        ) / 2 / u_embeddings.size(0)

        # Self-supervised contrastive loss
        ssl_loss = self.calculate_ssl_loss(users, pos_items)

        # Total loss
        loss = bpr_loss + self.reg_weight * reg_loss + self.ssl_weight * ssl_loss

        return loss

    def calculate_ssl_loss(self, users, pos_items):
        # Generate two augmented views
        user_embeddings, item_embeddings = self.forward()

        def augment_embeddings(embeddings):
            noise = F.dropout(embeddings, p=self.ssl_dropout, training=True)
            return embeddings + noise

        u_embeddings_aug1 = augment_embeddings(user_embeddings[users])
        u_embeddings_aug2 = augment_embeddings(user_embeddings[users])

        pos_i_embeddings_aug1 = augment_embeddings(item_embeddings[pos_items])
        pos_i_embeddings_aug2 = augment_embeddings(item_embeddings[pos_items])

        # Compute contrastive loss for users
        ssl_loss_u = self.info_nce_loss(u_embeddings_aug1, u_embeddings_aug2)

        # Compute contrastive loss for items
        ssl_loss_i = self.info_nce_loss(pos_i_embeddings_aug1, pos_i_embeddings_aug2)

        return ssl_loss_u + ssl_loss_i

    def info_nce_loss(self, z1, z2):
        temperature = self.ssl_temp
        batch_size = z1.size(0)

        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        pos_scores = torch.exp(torch.sum(z1 * z2, dim=1) / temperature)
        total_scores = torch.exp(torch.mm(z1, z2.t()) / temperature).sum(dim=1)

        loss = -torch.log(pos_scores / total_scores).mean()
        return loss

    def full_sort_predict(self, interaction):
        users = interaction[0]

        user_embeddings, item_embeddings = self.forward()
        u_embeddings = user_embeddings[users]

        # Compute scores
        scores = torch.matmul(u_embeddings, item_embeddings.t())
        return scores

