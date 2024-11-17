# coding: utf-8
# @email: enoche.chow@gmail.com
r"""
AMGAN: Adaptive Multimodal Graph Attention for Dynamic Representation Learning in Multimedia Recommendation Systems
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, remove_self_loops, dropout_adj

from common.abstract_recommender import GeneralRecommender
from common.loss import EmbLoss


class DynamicGraphUpdateModule(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(DynamicGraphUpdateModule, self).__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)

    def forward(self, edge_features, hidden_state=None):
        # Use an RNN to model changes in edge weights over time
        output, hidden_state = self.rnn(edge_features, hidden_state)
        return output, hidden_state


class MultimodalGraphAttentionLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(MultimodalGraphAttentionLayer, self).__init__(aggr="add")  # Aggregate messages using sum
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fc = nn.Linear(in_channels, out_channels)
        self.attention = nn.Linear(2 * out_channels, 1)

    def forward(self, x, edge_index):
        # Add self-loops to the adjacency matrix
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        # Transform node features
        x = self.fc(x)
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j, edge_index, size):
        # Compute attention coefficients
        attention_input = torch.cat([x_i, x_j], dim=-1)
        alpha = F.leaky_relu(self.attention(attention_input))
        alpha = torch.sigmoid(alpha)
        return x_j * alpha


class TemporalSelfAttentionEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(TemporalSelfAttentionEncoder, self).__init__()
        self.self_attention = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, x):
        # Self-attention expects input of shape (sequence_length, batch_size, embed_dim)
        x = x.permute(1, 0, 2)
        output, _ = self.self_attention(x, x, x)
        return output.permute(1, 0, 2)


class AMGAN(GeneralRecommender):
    def __init__(self, config, dataset):
        super(AMGAN, self).__init__(config, dataset)
        
        self.num_user = self.n_users
        self.num_item = self.n_items
        self.embedding_dim = config['embedding_size']
        self.hidden_dim = config['hidden_size']
        self.num_heads = config['num_heads']
        self.reg_weight = config['reg_weight']

        # Embeddings
        self.user_embedding = nn.Embedding(self.num_user, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.num_item, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        # Multimodal Feature Transformation
        if self.v_feat is not None:
            self.image_transform = nn.Linear(self.v_feat.shape[1], self.embedding_dim)
        if self.t_feat is not None:
            self.text_transform = nn.Linear(self.t_feat.shape[1], self.embedding_dim)

        # Dynamic Graph Update Module
        self.dynamic_graph_update = DynamicGraphUpdateModule(self.embedding_dim, self.hidden_dim)

        # Multimodal Graph Attention Layer
        self.graph_attention = MultimodalGraphAttentionLayer(self.embedding_dim, self.embedding_dim)

        # Temporal Self-Attention Encoder
        self.temporal_attention = TemporalSelfAttentionEncoder(self.embedding_dim, self.num_heads)

        # Loss function
        self.reg_loss = EmbLoss()

    def forward(self, edge_index, edge_features, hidden_state=None):
        # Dynamic graph update
        edge_features, hidden_state = self.dynamic_graph_update(edge_features, hidden_state)

        # Get node embeddings
        user_embedding = self.user_embedding.weight
        item_embedding = self.item_embedding.weight
        all_embeddings = torch.cat([user_embedding, item_embedding], dim=0)

        # Multimodal feature transformation
        if self.v_feat is not None:
            image_features = F.leaky_relu(self.image_transform(self.v_feat))
            all_embeddings = torch.cat([all_embeddings, image_features], dim=1)
        if self.t_feat is not None:
            text_features = F.leaky_relu(self.text_transform(self.t_feat))
            all_embeddings = torch.cat([all_embeddings, text_features], dim=1)

        # Graph attention
        node_embeddings = self.graph_attention(all_embeddings, edge_index)

        # Temporal attention
        node_embeddings = self.temporal_attention(node_embeddings)

        return node_embeddings

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1] + self.num_user
        neg_items = interaction[2] + self.num_user

        # Get the representations
        edge_index, _ = dropout_adj(self.edge_index, p=self.dropout)
        edge_features = torch.ones(edge_index.size(1), self.embedding_dim).to(self.device)  # Dummy edge features
        all_embeddings = self.forward(edge_index, edge_features)

        user_tensor = all_embeddings[users]
        pos_item_tensor = all_embeddings[pos_items]
        neg_item_tensor = all_embeddings[neg_items]

        # BPR loss
        pos_scores = torch.sum(user_tensor * pos_item_tensor, dim=-1)
        neg_scores = torch.sum(user_tensor * neg_item_tensor, dim=-1)
        bpr_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))

        # Regularization loss
        reg_loss = self.reg_weight * self.reg_loss(user_tensor, pos_item_tensor)

        return bpr_loss + reg_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]

        # Get the representations
        edge_index, _ = dropout_adj(self.edge_index, p=self.dropout)
        edge_features = torch.ones(edge_index.size(1), self.embedding_dim).to(self.device)  # Dummy edge features
        all_embeddings = self.forward(edge_index, edge_features)

        user_embedding = all_embeddings[user]
        item_embeddings = all_embeddings[self.num_user:]

        # Compute scores
        scores = torch.matmul(user_embedding, item_embeddings.t())
        return scores
