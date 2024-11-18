# coding: utf-8
# MULTI-GRAIL: Multi-modal Graph Refined Adaptive Integration Learner
# Aiming for SOTA performance in multimodal recommendation.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss
from torch.nn.functional import cosine_similarity

class PHOENIX(GeneralRecommender):
    def __init__(self, config, dataset):
        super(PHOENIX, self).__init__(config, dataset)
        
        self.embedding_dim = config["embedding_size"]
        self.feat_embed_dim = config["feat_embed_dim"]
        self.n_layers = config["n_mm_layers"]
        self.reg_weight = config["reg_weight"]
        self.cl_weight = config["cl_weight"]
        self.knn_k = config["knn_k"]
        self.mm_image_weight = config["mm_image_weight"]
        self.dropout = config["dropout"]

        self.n_nodes = self.n_users + self.n_items

        # Adjacency matrix (interaction graph)
        self.norm_adj = self.get_norm_adj_mat(dataset.inter_matrix(form="coo").astype(np.float32)).to(self.device)

        # User and Item embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        # Visual and textual embeddings (if available)
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)
            nn.init.xavier_normal_(self.image_trs.weight)
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)
            nn.init.xavier_normal_(self.text_trs.weight)

        # Predictor for contrastive learning
        self.predictor = nn.Linear(self.embedding_dim, self.embedding_dim)
        nn.init.xavier_normal_(self.predictor.weight)

        # Regularization loss
        self.reg_loss = EmbLoss()

        # Multi-head attention weights for graph and modalities
        self.modal_weights = nn.Parameter(torch.Tensor([0.5, 0.5]))  # Textual, Visual
        self.softmax = nn.Softmax(dim=0)

    def get_norm_adj_mat(self, interaction_matrix):
        """Generates normalized adjacency matrix for user-item interactions."""
        A = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        inter_M = interaction_matrix
        inter_M_t = interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        for key, value in data_dict.items():
            A[key] = value
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
        """Generates embeddings using multi-modal and interaction graphs."""
        # Propagate item embeddings through multi-modal graphs
        h_item = self.item_id_embedding.weight
        for _ in range(self.n_layers):
            h_item = torch.sparse.mm(self.norm_adj, h_item)

        # Multi-modal embeddings
        t_feat, v_feat = None, None
        if self.t_feat is not None:
            t_feat = self.text_trs(self.text_embedding.weight)
        if self.v_feat is not None:
            v_feat = self.image_trs(self.image_embedding.weight)

        # Fuse multi-modal embeddings
        if self.v_feat is not None and self.t_feat is not None:
            weight = self.softmax(self.modal_weights)
            h_item += weight[0] * t_feat + weight[1] * v_feat
        elif self.t_feat is not None:
            h_item += t_feat
        elif self.v_feat is not None:
            h_item += v_feat

        # User embeddings via GCN propagation
        ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
        all_embeddings = [ego_embeddings]
        for _ in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.norm_adj, ego_embeddings)
            all_embeddings.append(ego_embeddings)
        all_embeddings = torch.stack(all_embeddings, dim=1).mean(dim=1)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)

        return u_g_embeddings, i_g_embeddings + h_item

    def calculate_loss(self, interactions):
        """Combines BPR and contrastive learning losses."""
        u_online, i_online = self.forward()
        users, pos_items, neg_items = interactions[0], interactions[1], interactions[2]

        # BPR Loss
        u_embed, pos_embed, neg_embed = u_online[users], i_online[pos_items], i_online[neg_items]
        pos_scores = torch.sum(u_embed * pos_embed, dim=1)
        neg_scores = torch.sum(u_embed * neg_embed, dim=1)
        bpr_loss = -torch.mean(torch.log2(torch.sigmoid(pos_scores - neg_scores)))

        # Contrastive Learning Loss
        with torch.no_grad():
            u_target, i_target = u_online.clone().detach(), i_online.clone().detach()
        u_online, i_online = self.predictor(u_online), self.predictor(i_online)
        contrastive_loss = 1 - cosine_similarity(u_online[users], i_target[pos_items], dim=-1).mean()

        return bpr_loss + self.cl_weight * contrastive_loss + self.reg_weight * self.reg_loss(u_online, i_online)

    def full_sort_predict(self, interactions):
        """Predicts scores for all items for a given user."""
        user = interactions[0]
        u_online, i_online = self.forward()
        scores = torch.matmul(u_online[user], i_online.transpose(0, 1))
        return scores
