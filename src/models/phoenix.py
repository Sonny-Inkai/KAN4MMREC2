# coding: utf-8
# Phoenix Model for Multimodal Recommendation

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, degree
from common.abstract_recommender import GeneralRecommender

class PHOENIX(GeneralRecommender):
    def __init__(self, config, dataset):
        super(PHOENIX, self).__init__(config, dataset)

        # Configuration and parameters
        self.embedding_dim = config["embedding_size"]
        self.feat_embed_dim = config["feat_embed_dim"]
        self.n_layers = config["n_mm_layers"]
        self.knn_k = config["knn_k"]
        self.reg_weight = config["reg_weight"]
        self.dropout = config["dropout"]
        self.device = config["device"]

        # User and Item Embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        # Visual and Textual Feature Embeddings
        if self.v_feat is not None:
            self.visual_transform = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)
        if self.t_feat is not None:
            self.textual_transform = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)

        # Graph Convolution Layers
        self.graph_convs = nn.ModuleList()
        for _ in range(self.n_layers):
            self.graph_convs.append(GraphConvolution(self.embedding_dim, self.embedding_dim))

        # Multimodal Graph Construction
        self.mm_adj = self.build_multimodal_graph(dataset)

    def build_multimodal_graph(self, dataset):
        # Build adjacency matrix using both visual and textual features
        adj_matrix = torch.eye(self.n_users + self.n_items).to(self.device)
        if self.v_feat is not None:
            adj_matrix += self.build_knn_graph(self.v_feat, topk=self.knn_k)
        if self.t_feat is not None:
            adj_matrix += self.build_knn_graph(self.t_feat, topk=self.knn_k)
        return adj_matrix

    def build_knn_graph(self, features, topk):
        context_norm = features.div(torch.norm(features, p=2, dim=-1, keepdim=True))
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        _, knn_indices = torch.topk(sim, topk, dim=-1)
        adj_size = sim.size()
        del sim

        indices0 = torch.arange(knn_indices.shape[0]).to(self.device)
        indices0 = indices0.unsqueeze(1).expand(-1, topk)
        indices = torch.stack((indices0.flatten(), knn_indices.flatten()), 0)
        return torch.sparse_coo_tensor(indices, torch.ones_like(indices[0]), adj_size)

    def forward(self):
        # Base User and Item Embeddings
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight

        # Process Visual and Textual Features
        visual_emb, textual_emb = 0, 0
        if self.v_feat is not None:
            visual_emb = F.relu(self.visual_transform(self.v_feat))
        if self.t_feat is not None:
            textual_emb = F.relu(self.textual_transform(self.t_feat))

        # Enhanced Item Embedding with Visual and Textual Information
        item_emb = item_emb + visual_emb + textual_emb

        # Concatenate User and Item Embeddings for GCN
        all_emb = torch.cat((user_emb, item_emb), dim=0)

        # Graph Convolution Layers
        for conv in self.graph_convs:
            all_emb = F.relu(conv(all_emb, self.mm_adj))

        # Separate User and Item Embeddings
        user_emb, item_emb = torch.split(all_emb, [self.n_users, self.n_items], dim=0)
        return user_emb, item_emb

    def calculate_loss(self, interaction):
        users, pos_items, neg_items = interaction

        # Forward Pass to Get Embeddings
        user_emb, item_emb = self.forward()

        # Extract Embeddings for Current Batch
        u_emb = user_emb[users]
        pos_i_emb = item_emb[pos_items]
        neg_i_emb = item_emb[neg_items]

        # Calculate BPR Loss
        pos_scores = torch.sum(u_emb * pos_i_emb, dim=1)
        neg_scores = torch.sum(u_emb * neg_i_emb, dim=1)
        bpr_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))

        # Add Regularization Loss
        reg_loss = self.reg_weight * (
            torch.norm(u_emb) ** 2 + torch.norm(pos_i_emb) ** 2 + torch.norm(neg_i_emb) ** 2
        )

        return bpr_loss + reg_loss

    def full_sort_predict(self, interaction):
        users = interaction

        # Forward Pass to Get Embeddings
        user_emb, item_emb = self.forward()

        # Get Embeddings for the Queried Users
        user_emb = user_emb[users]

        # Compute Scores for All Items
        scores = torch.matmul(user_emb, item_emb.t())
        return scores


class GraphConvolution(MessagePassing):
    def __init__(self, in_channels, out_channels, aggr='add'):
        super(GraphConvolution, self).__init__(aggr=aggr)
        self.linear = nn.Linear(in_channels, out_channels)
        self.aggr = aggr

    def forward(self, x, edge_index):
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = self.linear(x)
        return self.propagate(edge_index, x=x)

    def message(self, x_j, edge_index, size):
        row, col = edge_index
        deg = degree(row, size[0], dtype=x_j.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        return aggr_out
