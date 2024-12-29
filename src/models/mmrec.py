# coding: utf-8
# @email: enoche.chow@gmail.com
r"""
MMRec: Multimodal Recommendation with Modality-Aware Graph Neural Networks
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
from utils.utils import build_sim, compute_normalized_laplacian


class MMREC(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MMREC, self).__init__(config, dataset)

        self.embedding_dim = config["embedding_size"]
        self.feat_embed_dim = config["feat_embed_dim"]
        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.reg_weight = config["reg_weight"]
        self.dropout = config["dropout"]
        self.modality_agg = config["modality_agg"]

        self.n_nodes = self.n_users + self.n_items

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form="coo").astype(np.float32)
        self.norm_adj = self.get_norm_adj_mat().to(self.device)

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        self.v_feat_embedding = None
        if self.v_feat is not None:
            self.v_feat_embedding = nn.Embedding.from_pretrained(
                self.v_feat, freeze=False
            )
            self.v_feat_trs = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)

        self.t_feat_embedding = None  
        if self.t_feat is not None:
            self.t_feat_embedding = nn.Embedding.from_pretrained(
                self.t_feat, freeze=False
            )
            self.t_feat_trs = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)

        self.modal_aggregator = nn.Parameter(torch.ones(2)/2)

        self.gnn_layers = nn.ModuleList()
        for i in range(self.n_layers):
            self.gnn_layers.append(MMGNNLayer(self.embedding_dim, self.n_heads, self.dropout))

    def get_norm_adj_mat(self):
        A = sp.dok_matrix(
            (self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32
        )
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(
            zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz)
        )
        data_dict.update(
            dict(
                zip(
                    zip(inter_M_t.row + self.n_users, inter_M_t.col),
                    [1] * inter_M_t.nnz,
                )
            )
        )
        for key, value in data_dict.items():
            A[key] = value
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL
    
    def forward(self):
        ego_embeddings = torch.cat(
            (self.user_embedding.weight, self.item_id_embedding.weight), dim=0
        )

        if self.v_feat_embedding is not None:
            v_feat = self.v_feat_trs(self.v_feat_embedding.weight)
            v_feat = torch.cat((torch.zeros(self.n_users, self.feat_embed_dim).to(self.device), v_feat), dim=0)
        if self.t_feat_embedding is not None:  
            t_feat = self.t_feat_trs(self.t_feat_embedding.weight)
            t_feat = torch.cat((torch.zeros(self.n_users, self.feat_embed_dim).to(self.device), t_feat), dim=0)

        all_embeddings = [ego_embeddings]
        
        for i, gnn_layer in enumerate(self.gnn_layers):
            if self.modality_agg == "concat":
                if self.v_feat_embedding is not None and self.t_feat_embedding is not None:
                    ego_embeddings = torch.cat((ego_embeddings, v_feat, t_feat), dim=1)
                elif self.v_feat_embedding is not None:
                    ego_embeddings = torch.cat((ego_embeddings, v_feat), dim=1)  
                elif self.t_feat_embedding is not None:
                    ego_embeddings = torch.cat((ego_embeddings, t_feat), dim=1)
            elif self.modality_agg == "weighted":
                if self.v_feat_embedding is not None and self.t_feat_embedding is not None:
                    ego_embeddings = self.modal_aggregator[0] * ego_embeddings + self.modal_aggregator[1] * (v_feat + t_feat)/2
                elif self.v_feat_embedding is not None:
                    ego_embeddings = self.modal_aggregator[0] * ego_embeddings + self.modal_aggregator[1] * v_feat
                elif self.t_feat_embedding is not None:
                    ego_embeddings = self.modal_aggregator[0] * ego_embeddings + self.modal_aggregator[1] * t_feat
            
            ego_embeddings = gnn_layer(ego_embeddings, self.norm_adj)
            all_embeddings += [ego_embeddings]
        
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        
        u_g_embeddings, i_g_embeddings = torch.split(
            all_embeddings, [self.n_users, self.n_items], dim=0
        )
        return u_g_embeddings, i_g_embeddings

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        u_g_embeddings, i_g_embeddings = self.forward()

        u_embeddings = u_g_embeddings[users]
        pos_embeddings = i_g_embeddings[pos_items]
        neg_embeddings = i_g_embeddings[neg_items]

        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        mf_loss = -torch.log2(torch.sigmoid(pos_scores - neg_scores)).mean()

        reg_loss = (
            L2Loss(u_g_embeddings, self.reg_weight) + 
            L2Loss(i_g_embeddings, self.reg_weight)
        )

        return mf_loss + reg_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]

        u_g_embeddings, i_g_embeddings = self.forward()
        u_embeddings = u_g_embeddings[user]

        scores = torch.matmul(u_embeddings, i_g_embeddings.transpose(0, 1))
        return scores


class MMGNNLayer(nn.Module):
    def __init__(self, embedding_dim, n_heads, dropout):
        super(MMGNNLayer, self).__init__()
        self.dropout = dropout
        self.n_heads = n_heads
        self.attentions = [GATLayer(embedding_dim, embedding_dim//n_heads, dropout=dropout) for _ in range(n_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        return x


class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0, alpha=0.2):
        super(GATLayer, self).__init__()
        self.dropout = dropout
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.alpha = alpha

        self.W = nn.Parameter(torch.zeros(size=(in_dim, out_dim)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a_src = nn.Parameter(torch.zeros(size=(1, out_dim)))
        self.a_dst = nn.Parameter(torch.zeros(size=(1, out_dim)))
        nn.init.xavier_uniform_(self.a_src.data, gain=1.414)
        nn.init.xavier_uniform_(self.a_dst.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, x, adj):
        h = torch.mm(x, self.W)
        N = h.size()[0]
        e_src = h @ self.a_src.t()
        e_dst = h @ self.a_dst.t()
        e = self.leakyrelu(e_src.expand(-1, N) + e_dst.expand(N, -1).t())

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)
        return h_prime 