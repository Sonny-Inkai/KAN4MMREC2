# coding: utf-8
# @email: enoche.chow@gmail.com
r"""
AMGAN: Adaptive Multimodal Graph Attention Network for Dynamic Representation Learning in Multimedia Recommendation Systems
# Update: 01/11/2024
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import dropout_adj, softmax
from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss
from common.init import xavier_uniform_initialization


class DynamicGraphUpdate(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim):
        super(DynamicGraphUpdate, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.user_gru = nn.GRU(embedding_dim, embedding_dim, batch_first=True)
        self.item_gru = nn.GRU(embedding_dim, embedding_dim, batch_first=True)
        self.user_embeddings = nn.Embedding(n_users, embedding_dim)
        self.item_embeddings = nn.Embedding(n_items, embedding_dim)
        nn.init.xavier_uniform_(self.user_embeddings.weight)
        nn.init.xavier_uniform_(self.item_embeddings.weight)

    def forward(self, user_sequence, item_sequence):
        user_embed = self.user_embeddings(user_sequence)
        item_embed = self.item_embeddings(item_sequence)
        user_output, _ = self.user_gru(user_embed)
        item_output, _ = self.item_gru(item_embed)
        return user_output[:, -1], item_output[:, -1]


class MultimodalGraphAttentionLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(MultimodalGraphAttentionLayer, self).__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.linear = nn.Linear(in_channels, out_channels)
        self.attention_weights = Parameter(torch.Tensor(out_channels, 1))
        nn.init.xavier_uniform_(self.attention_weights)

    def forward(self, x, edge_index):
        x = self.linear(x)
        edge_index, _ = dropout_adj(edge_index, p=0.2)
        return self.propagate(edge_index, x=x, size=(x.size(0), x.size(0)))

    def message(self, x_j, index, ptr, size_i):
        alpha = torch.matmul(x_j, self.attention_weights)
        alpha = softmax(alpha, index, ptr, size_i)
        return x_j * alpha


class TemporalSelfAttentionEncoder(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super(TemporalSelfAttentionEncoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.self_attention = nn.MultiheadAttention(embedding_dim, num_heads, batch_first=True)

    def forward(self, x):
        attn_output, _ = self.self_attention(x, x, x)
        return attn_output


class AMGAN(GeneralRecommender):
    def __init__(self, config, dataset):
        super(AMGAN, self).__init__(config, dataset)
        self.num_user = self.n_users
        self.num_item = self.n_items
        self.embedding_dim = config['embedding_size']
        self.num_heads = config['num_heads']
        self.reg_weight = config['reg_weight']
        self.dynamic_graph = DynamicGraphUpdate(self.num_user, self.num_item, self.embedding_dim)
        self.graph_attention = MultimodalGraphAttentionLayer(self.embedding_dim, self.embedding_dim)
        self.temporal_attention = TemporalSelfAttentionEncoder(self.embedding_dim, self.num_heads)

        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.embedding_dim)
            nn.init.xavier_uniform_(self.image_trs.weight)
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.embedding_dim)
            nn.init.xavier_uniform_(self.text_trs.weight)

        # Initialize edge_index for training
        train_interactions = dataset.inter_matrix(form='coo').astype(np.float32)
        edge_index = torch.tensor(self.pack_edge_index(train_interactions), dtype=torch.long)
        self.edge_index = edge_index.t().contiguous().to(self.device)

    def pack_edge_index(self, inter_mat):
        rows = inter_mat.row
        cols = inter_mat.col + self.n_users
        return np.column_stack((rows, cols))

    def forward(self, user_sequence, item_sequence):
        user_output, item_output = self.dynamic_graph(user_sequence, item_sequence)
        x = torch.cat((user_output, item_output), dim=0)
        edge_index = torch.cat((self.edge_index, self.edge_index[[1, 0]]), dim=1)

        # Project visual and textual embeddings
        multimodal_rep = self.graph_attention(x, edge_index)
        if self.v_feat is not None:
            visual_features = F.relu(self.image_trs(self.image_embedding.weight))
            multimodal_rep += visual_features
        if self.t_feat is not None:
            textual_features = F.relu(self.text_trs(self.text_embedding.weight))
            multimodal_rep += textual_features

        temporal_output = self.temporal_attention(multimodal_rep.unsqueeze(0))
        return temporal_output.squeeze(0)

    def calculate_loss(self, interaction):
        user_sequence = interaction[0]
        pos_item_sequence = interaction[1]
        neg_item_sequence = interaction[2]
        pos_output = self.forward(user_sequence, pos_item_sequence)
        neg_output = self.forward(user_sequence, neg_item_sequence)

        pos_scores = torch.sum(pos_output * self.dynamic_graph.item_embeddings(pos_item_sequence), dim=1)
        neg_scores = torch.sum(neg_output * self.dynamic_graph.item_embeddings(neg_item_sequence), dim=1)
        loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores)))
        reg_loss = self.reg_weight * (self.dynamic_graph.user_embeddings.weight ** 2).mean()
        reg_loss += self.reg_weight * (self.dynamic_graph.item_embeddings.weight ** 2).mean()
        return loss + reg_loss

    def full_sort_predict(self, interaction):
        user_sequence = interaction[0]
        user_output = self.dynamic_graph.user_embeddings(user_sequence)
        item_output = self.dynamic_graph.item_embeddings.weight
        scores = torch.matmul(user_output, item_output.t())
        return scores
