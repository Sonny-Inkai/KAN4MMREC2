# coding: utf-8
#
# HYDRARec: Hybrid Graph-Based Self-Supervised Learning for Multimodal Recommendation
# Integrates the best features from SLMRec, BM3, DRAGON, FREEDOM, and advanced innovations
# Created by ChatGPT, leveraging the latest multimodal and graph recommendation techniques

import os
import random
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity
from torch_geometric.nn import GATConv

from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss


class AMGAN(GeneralRecommender):
    def __init__(self, config, dataset):
        super(AMGAN, self).__init__(config, dataset)

        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.n_layers = config['n_layers']
        self.reg_weight = config['reg_weight']
        self.cl_weight = config['cl_weight']
        self.dropout = config['dropout']
        self.knn_k = config['knn_k']
        self.mm_image_weight = config['mm_image_weight']

        self.n_nodes = self.n_users + self.n_items

        # Load dataset info
        self.norm_adj = self.get_norm_adj_mat(dataset.inter_matrix(form='coo').astype(np.float32)).to(self.device)

        # Embeddings for users and items
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        # Multimodal feature layers
        if hasattr(self, 'v_feat') and self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_trs = nn.Linear(self.v_feat.size(1), self.feat_embed_dim)
            nn.init.xavier_normal_(self.image_trs.weight)
        if hasattr(self, 't_feat') and self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_trs = nn.Linear(self.t_feat.size(1), self.feat_embed_dim)
            nn.init.xavier_normal_(self.text_trs.weight)

        # Graph Attention Network for enhancing graph representation learning
        self.gat_layers = nn.ModuleList([GATConv(self.embedding_dim, self.embedding_dim) for _ in range(self.n_layers)])

        # Contrastive Learning Layers
        self.predictor = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.reg_loss = EmbLoss()
        nn.init.xavier_normal_(self.predictor.weight)

    def get_norm_adj_mat(self, interaction_matrix):
        A = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        inter_M = interaction_matrix
        inter_M_t = interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        for key, value in data_dict.items():
            A[key] = value
        
        # Normalize adjacency matrix
        sumArr = (A > 0).sum(axis=1)
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        
        # Convert normalized adjacency matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)

        return torch.sparse_coo_tensor(i, data, torch.Size((self.n_nodes, self.n_nodes)))

    def forward(self):
        ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
        all_embeddings = [ego_embeddings]
        
        # Apply Graph Attention Layers
        for gat_layer in self.gat_layers:
            ego_embeddings = gat_layer(ego_embeddings, self.norm_adj.coalesce().indices())
            all_embeddings.append(ego_embeddings)

        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        return u_g_embeddings, i_g_embeddings

    def calculate_loss(self, interactions):
        # Online network forward pass
        u_online_ori, i_online_ori = self.forward()
        t_feat_online, v_feat_online = None, None
        if hasattr(self, 't_feat') and self.t_feat is not None:
            t_feat_online = self.text_trs(self.text_embedding.weight)
        if hasattr(self, 'v_feat') and self.v_feat is not None:
            v_feat_online = self.image_trs(self.image_embedding.weight)

        # Target embeddings for contrastive learning
        with torch.no_grad():
            u_target, i_target = u_online_ori.clone(), i_online_ori.clone()
            u_target = F.dropout(u_target, self.dropout)
            i_target = F.dropout(i_target, self.dropout)

            if hasattr(self, 't_feat') and self.t_feat is not None:
                t_feat_target = t_feat_online.clone()
                t_feat_target = F.dropout(t_feat_target, self.dropout)

            if hasattr(self, 'v_feat') and self.v_feat is not None:
                v_feat_target = v_feat_online.clone()
                v_feat_target = F.dropout(v_feat_target, self.dropout)

        u_online, i_online = self.predictor(u_online_ori), self.predictor(i_online_ori)

        users, items = interactions[0], interactions[1]
        u_online = u_online[users, :]
        i_online = i_online[items, :]
        u_target = u_target[users, :]
        i_target = i_target[items, :]

        # Contrastive learning losses
        loss_t, loss_v, loss_tv, loss_vt = 0.0, 0.0, 0.0, 0.0
        if hasattr(self, 't_feat') and self.t_feat is not None:
            t_feat_online = self.predictor(t_feat_online)
            t_feat_online = t_feat_online[items, :]
            t_feat_target = t_feat_target[items, :]
            loss_t = 1 - cosine_similarity(t_feat_online, i_target.detach(), dim=-1).mean()
            loss_tv = 1 - cosine_similarity(t_feat_online, t_feat_target.detach(), dim=-1).mean()
        if hasattr(self, 'v_feat') and self.v_feat is not None:
            v_feat_online = self.predictor(v_feat_online)
            v_feat_online = v_feat_online[items, :]
            v_feat_target = v_feat_target[items, :]
            loss_v = 1 - cosine_similarity(v_feat_online, i_target.detach(), dim=-1).mean()
            loss_vt = 1 - cosine_similarity(v_feat_online, v_feat_target.detach(), dim=-1).mean()

        # User-item interaction contrastive loss
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
