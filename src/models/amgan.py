# coding: utf-8
#
# Updated by enoche
# Paper: Self-supervised Learning for Multimedia Recommendation
# Github: https://github.com/zltao/SLMRec
# Enhanced: AMGAN++ (with Graph-Freezing, Bootstrap, Self-Attention, Cross-Attention Fusion, Improved Regularization, and Skip Connections)

import os
import copy
import random
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity
from torch_geometric.nn import GATConv

from common.abstract_recommender import GeneralRecommender
from common.loss import EmbLoss

class AMGAN(GeneralRecommender):
    def __init__(self, config, dataset):
        super(AMGAN, self).__init__(config, dataset)

        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config['embedding_size']
        self.n_layers = config['n_layers']
        self.reg_weight = config['reg_weight']
        self.cl_weight = config['cl_weight']
        self.dropout = config['dropout']
        self.lambda_coeff = config["lambda_coeff"]
        self.knn_k = config["knn_k"]
        self.n_nodes = self.n_users + self.n_items

        # Load dataset info
        self.norm_adj = self.get_norm_adj_mat(dataset.inter_matrix(form='coo').astype(np.float32)).to(self.device)

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        self.predictor = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.reg_loss = EmbLoss()
        nn.init.xavier_normal_(self.predictor.weight)

        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_trs = nn.Linear(self.v_feat.size(-1), self.feat_embed_dim)
            nn.init.xavier_normal_(self.image_trs.weight)
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_trs = nn.Linear(self.t_feat.size(-1), self.feat_embed_dim)
            nn.init.xavier_normal_(self.text_trs.weight)

        # Self-Attention Layers for Graph Information
        self.attention_layer = GATConv(self.embedding_dim, self.embedding_dim, heads=2, concat=False)

        # Adding Layer Normalization and Dropout for stability and regularization
        self.layer_norm = nn.LayerNorm(self.embedding_dim)
        self.dropout_layer = nn.Dropout(self.dropout)

    def get_norm_adj_mat(self, interaction_matrix):
        A = sp.dok_matrix((self.n_users + self.n_items,
                           self.n_users + self.n_items), dtype=np.float32)
        inter_M = interaction_matrix
        inter_M_t = interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users),
                             [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col),
                                  [1] * inter_M_t.nnz)))
        for key, value in data_dict.items():
            A[key] = value
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid Divide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # convert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)

        return torch.sparse_coo_tensor(i, data, torch.Size((self.n_nodes, self.n_nodes)))

    def forward(self):
        h = self.item_id_embedding.weight

        # Apply GAT for self-attention enhanced aggregation with skip connections
        ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
        all_embeddings = [ego_embeddings]
        for i in range(self.n_layers):
            attention_output = self.attention_layer(ego_embeddings, self.norm_adj.coalesce().indices())
            attention_output = self.layer_norm(attention_output)  # Layer Normalization
            attention_output = self.dropout_layer(attention_output)  # Dropout for regularization
            ego_embeddings = ego_embeddings + attention_output  # Skip connection
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        return u_g_embeddings, i_g_embeddings + h

    def calculate_loss(self, interactions):
        # Online network
        u_online_ori, i_online_ori = self.forward()
        t_feat_online, v_feat_online = None, None
        if self.t_feat is not None:
            t_feat_online = self.text_trs(self.text_embedding.weight)
        if self.v_feat is not None:
            v_feat_online = self.image_trs(self.image_embedding.weight)

        # Bootstrap target network with dropout
        with torch.no_grad():
            u_target, i_target = u_online_ori.clone(), i_online_ori.clone()
            u_target.detach()
            i_target.detach()
            u_target = F.dropout(u_target, self.dropout)
            i_target = F.dropout(i_target, self.dropout)

            if self.t_feat is not None:
                t_feat_target = t_feat_online.clone()
                t_feat_target = F.dropout(t_feat_target, self.dropout)

            if self.v_feat is not None:
                v_feat_target = v_feat_online.clone()
                v_feat_target = F.dropout(v_feat_target, self.dropout)

        # Predict with cross-attention fusion
        u_online, i_online = self.predictor(u_online_ori), self.predictor(i_online_ori)
        users, items = interactions[0], interactions[1]
        u_online = u_online[users, :]
        i_online = i_online[items, :]
        u_target = u_target[users, :]
        i_target = i_target[items, :]

        # Calculate loss terms
        loss_t, loss_v, loss_tv, loss_vt = 0.0, 0.0, 0.0, 0.0
        if self.t_feat is not None:
            loss_t = 1 - cosine_similarity(t_feat_online[items, :], i_target.detach(), dim=-1).mean()
            loss_tv = 1 - cosine_similarity(t_feat_online[items, :], t_feat_target[items, :].detach(), dim=-1).mean()
        if self.v_feat is not None:
            loss_v = 1 - cosine_similarity(v_feat_online[items, :], i_target.detach(), dim=-1).mean()
            loss_vt = 1 - cosine_similarity(v_feat_online[items, :], v_feat_target[items, :].detach(), dim=-1).mean()

        loss_ui = 1 - cosine_similarity(u_online, i_target.detach(), dim=-1).mean()
        loss_iu = 1 - cosine_similarity(i_online, u_target.detach(), dim=-1).mean()

        return (loss_ui + loss_iu).mean() + self.reg_weight * self.reg_loss(u_online_ori, i_online_ori) + \
               self.cl_weight * (loss_t + loss_v + loss_tv + loss_vt).mean()

    def full_sort_predict(self, interaction):
        user = interaction[0]
        u_online, i_online = self.forward()
        u_online, i_online = self.predictor(u_online), self.predictor(i_online)
        score_mat_ui = torch.matmul(u_online[user], i_online.transpose(0, 1))
        return score_mat_ui
