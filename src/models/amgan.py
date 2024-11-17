# coding: utf-8
# @title Adaptive Multimodal Graph Attention Network (AMGAN)
# "Adaptive Multimodal Graph Attention for Dynamic Representation Learning in Multimedia Recommendation Systems"
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import numpy as np
from torch.nn import Parameter
from torch_scatter import scatter_add
from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss

class AMGAN(GeneralRecommender):
    def __init__(self, config, dataset):
        super(AMGAN, self).__init__(config, dataset)

        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.num_user = self.n_users
        self.num_item = self.n_items
        self.n_layers = config['n_layers']
        self.dropout = config['dropout']
        self.device = config['device']

        # Define user and item embeddings
        self.user_embedding = nn.Embedding(self.num_user, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.num_item, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        # Define multimodal feature embeddings
        if self.v_feat is not None:
            self.visual_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.visual_trs = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)
        
        # Multimodal graph attention layers
        self.graph_attention_layer = GraphAttentionLayer(self.embedding_dim)
        self.dynamic_graph_update = DynamicGraphUpdate(self.embedding_dim)
        
        # Temporal self-attention encoder
        self.temporal_self_attention = nn.MultiheadAttention(embed_dim=self.embedding_dim, num_heads=4)

    def forward(self, edge_index):
        # Initial embeddings
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight
        
        # Concatenate user and item embeddings
        x = torch.cat([user_emb, item_emb], dim=0)

        # Update graph structure using recent interactions
        dynamic_edge_index = self.dynamic_graph_update(edge_index, x)

        # Apply multimodal graph attention
        x = self.graph_attention_layer(x, dynamic_edge_index)
        
        # Apply temporal self-attention
        x = x.unsqueeze(0)  # Add sequence dimension for attention
        x, _ = self.temporal_self_attention(x, x, x)
        x = x.squeeze(0)

        # Split user and item embeddings after transformation
        user_rep, item_rep = torch.split(x, [self.num_user, self.num_item], dim=0)
        
        return user_rep, item_rep

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        user_rep, item_rep = self.forward(self.edge_index)

        u_g_embeddings = user_rep[users]
        pos_i_g_embeddings = item_rep[pos_items]
        neg_i_g_embeddings = item_rep[neg_items]

        pos_scores = torch.sum(u_g_embeddings * pos_i_g_embeddings, dim=1)
        neg_scores = torch.sum(u_g_embeddings * neg_i_g_embeddings, dim=1)

        loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores)))
        return loss

    def full_sort_predict(self, interaction):
        users = interaction[0]
        user_rep, item_rep = self.forward(self.edge_index)

        u_embeddings = user_rep[users]
        scores = torch.matmul(u_embeddings, item_rep.t())
        return scores


class GraphAttentionLayer(MessagePassing):
    def __init__(self, in_channels, aggr='add'):
        super(GraphAttentionLayer, self).__init__(aggr=aggr)
        self.in_channels = in_channels
        self.attn_linear = nn.Linear(2 * in_channels, 1)
        nn.init.xavier_uniform_(self.attn_linear.weight)

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j, edge_index):
        # Concatenate and compute attention scores
        alpha = torch.cat([x_i, x_j], dim=1)
        alpha = F.leaky_relu(self.attn_linear(alpha))
        alpha = torch.exp(alpha)
        alpha_sum = scatter_add(alpha, edge_index[0], dim=0)
        alpha = alpha / (alpha_sum[edge_index[0]] + 1e-7)
        return x_j * alpha


class DynamicGraphUpdate(nn.Module):
    def __init__(self, embedding_dim):
        super(DynamicGraphUpdate, self).__init__()
        self.rnn = nn.GRU(input_size=embedding_dim, hidden_size=embedding_dim, batch_first=True)

    def forward(self, edge_index, x):
        # Convert edge index to a dense adjacency matrix
        adj_matrix = torch.zeros((x.size(0), x.size(0))).to(x.device)
        adj_matrix[edge_index[0], edge_index[1]] = 1
        
        # Feed adjacency matrix through RNN to update structure
        adj_matrix = adj_matrix.unsqueeze(0)  # Add batch dimension
        _, updated_adj_matrix = self.rnn(adj_matrix)
        updated_adj_matrix = updated_adj_matrix.squeeze(0)

        # Thresholding to make the matrix binary
        updated_adj_matrix = (updated_adj_matrix > 0.5).float()
        updated_edge_index = torch.nonzero(updated_adj_matrix, as_tuple=False).t()
        return updated_edge_index
