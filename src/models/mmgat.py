# coding: utf-8
# @email: enoche.chow@gmail.com
r"""
GAT: Graph Attention Network for Multimodal Recommendation
# Update: 01/08/2022
"""

import os
import random
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss, L2Loss
from torch_geometric.nn import GATConv


class MMGAT(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MMGAT, self).__init__(config, dataset)

        self.embedding_dim = config["embedding_size"]
        self.n_layers = config["n_layers"]
        self.n_nodes = self.n_users + self.n_items

        # Load dataset info
        self.interaction_matrix = dataset.inter_matrix(form="coo").astype(np.float32)
        self.norm_adj = self.get_norm_adj_mat().to(self.device)

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        # GAT layers
        self.gat_layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.gat_layers.append(GATConv(self.embedding_dim, self.embedding_dim, heads=1, dropout=0.6))

    def get_norm_adj_mat(self):
        A = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M.col, inter_M.row + self.n_users), [1] * inter_M.nnz)))

        for key, value in data_dict.items():
            A[key] = value
            
        # Normalize adjacency matrix
        sumArr = (A > 0).sum(axis=1)
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)

        return torch.sparse.FloatTensor(i, data, torch.Size((self.n_nodes, self.n_nodes)))

    def forward(self):
        # Combine user and item embeddings
        all_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)

        # Pass through GAT layers
        for gat in self.gat_layers:
            all_embeddings = gat(all_embeddings, self.norm_adj)

        u_embeddings, i_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        return u_embeddings, i_embeddings

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(users * pos_items, dim=1)
        neg_scores = torch.sum(users * neg_items, dim=1)
        maxi = F.logsigmoid(pos_scores - neg_scores)
        return -torch.mean(maxi)

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        u_embeddings, i_embeddings = self.forward()

        u_g_embeddings = u_embeddings[users]
        pos_i_g_embeddings = i_embeddings[pos_items]
        neg_i_g_embeddings = i_embeddings[neg_items]

        return self.bpr_loss(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings)

    def full_sort_predict(self, interaction):
        user = interaction[0]
        u_embeddings, i_embeddings = self.forward()
        u_embeddings = u_embeddings[user]

        # Dot product with all item embeddings
        scores = torch.matmul(u_embeddings, i_embeddings.transpose(0, 1))
        return scores
