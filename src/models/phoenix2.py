# phoenix.py
import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss
from utils.utils import build_sim, compute_normalized_laplacian

class PHOENIX(GeneralRecommender):
    def __init__(self, config, dataset):
        super(PHOENIX, self).__init__(config, dataset)

        self.embedding_dim = config["embedding_size"]
        self.feat_embed_dim = config["feat_embed_dim"]
        self.knn_k = config["knn_k"] 
        self.lambda_coeff = config["lambda_coeff"]
        self.n_layers = config["n_mm_layers"]
        self.reg_weight = config["reg_weight"]
        self.contrast_weight = config["contrast_weight"]
        self.dropout = config["dropout"]
        self.mm_fusion_mode = config["mm_fusion_mode"]
        self.n_ui_layers = config["n_ui_layers"]
        self.temp = config["temp"]
        
        self.n_nodes = self.n_users + self.n_items

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form="coo").astype(np.float32)
        self.norm_adj = self.get_norm_adj_mat().to(self.device)

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)

        self.predictor = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.reg_loss = EmbLoss()

        # Build item-item graph
        self.build_item_graph()

    def build_item_graph(self):
        if self.v_feat is not None:
            self.image_adj = self.get_knn_adj_mat(self.image_embedding.weight)
        if self.t_feat is not None:
            self.text_adj = self.get_knn_adj_mat(self.text_embedding.weight)

    def get_knn_adj_mat(self, mm_embeddings):
        # Calculate similarity matrix
        sim_mat = build_sim(mm_embeddings)
        
        # Get top k similar items
        vals, inds = torch.topk(sim_mat, self.knn_k + 1)
        row_inds = torch.arange(sim_mat.size(0), device=self.device).view(-1, 1).expand(-1, self.knn_k)
        row_inds = row_inds.reshape(-1)
        col_inds = inds[:, 1:].reshape(-1)  # Exclude self-loop
        
        # Create sparse adjacency matrix
        indices = torch.stack([row_inds, col_inds])
        values = torch.ones_like(row_inds, device=self.device).float()
        adj = torch.sparse_coo_tensor(indices, values, 
                                    (mm_embeddings.size(0), mm_embeddings.size(0)),
                                    device=self.device)
        
        # Normalize adjacency matrix
        rowsum = torch.sparse.sum(adj, dim=1).to_dense()
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        
        # Calculate normalized adjacency matrix
        adj_dense = adj.to_dense()
        norm_adj = torch.mm(torch.mm(d_mat_inv_sqrt, adj_dense), d_mat_inv_sqrt)
        
        return norm_adj.to_sparse()

    def get_norm_adj_mat(self):
        A = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
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
        SparseL = torch.sparse_coo_tensor(i, data, torch.Size(L.shape))
        return SparseL

    def forward(self):
        h = self.item_id_embedding.weight

        # Perform graph convolutions on the item-item graph
        if self.v_feat is not None:
            h_v = h
            for _ in range(self.n_layers):
                h_v = torch.sparse.mm(self.image_adj, h_v)
        if self.t_feat is not None:
            h_t = h
            for _ in range(self.n_layers):
                h_t = torch.sparse.mm(self.text_adj, h_t)

        # Perform graph convolutions on the user-item graph
        ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
        all_embeddings = [ego_embeddings]
        for _ in range(self.n_ui_layers):
            ego_embeddings = torch.sparse.mm(self.norm_adj, ego_embeddings)
            all_embeddings.append(ego_embeddings)
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)

        # Make sure all embeddings have the same dimension as embedding_dim
        if self.mm_fusion_mode == "concat":
            final_dim = self.embedding_dim
            if self.v_feat is not None and self.t_feat is not None:
                h_v = h_v[:, :final_dim]
                h_t = h_t[:, :final_dim]
                i_embeddings = (i_g_embeddings + h_v + h_t) / 3
            elif self.v_feat is not None:
                h_v = h_v[:, :final_dim]
                i_embeddings = (i_g_embeddings + h_v) / 2
            elif self.t_feat is not None:
                h_t = h_t[:, :final_dim]
                i_embeddings = (i_g_embeddings + h_t) / 2
            else:
                i_embeddings = i_g_embeddings
        elif self.mm_fusion_mode == "mean":
            if self.v_feat is not None and self.t_feat is not None:
                i_embeddings = (i_g_embeddings + h_v + h_t) / 3
            elif self.v_feat is not None:
                i_embeddings = (i_g_embeddings + h_v) / 2
            elif self.t_feat is not None:
                i_embeddings = (i_g_embeddings + h_t) / 2
            else:
                i_embeddings = i_g_embeddings
        else:
            raise ValueError("Invalid mm_fusion_mode: " + self.mm_fusion_mode)

        return u_g_embeddings, i_embeddings
    
    def bpr_loss(self, users, pos_items, neg_items):
        users_emb = self.user_embedding(users.long())
        pos_emb = self.item_id_embedding(pos_items.long())
        neg_emb = self.item_id_embedding(neg_items.long())
        pos_scores = torch.sum(torch.mul(users_emb, pos_emb), dim=1)
        neg_scores = torch.sum(torch.mul(users_emb, neg_emb), dim=1)
        bpr_loss = -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()
        return bpr_loss

    def contrast_loss(self, users, items):
        users_emb = self.predictor(self.user_embedding(users.long()))
        items_emb = self.predictor(self.item_id_embedding(items.long()))

        users_emb = F.normalize(users_emb, dim=1)
        items_emb = F.normalize(items_emb, dim=1)

        pos_scores = torch.sum(torch.mul(users_emb, items_emb), dim=1)
        pos_scores = torch.exp(pos_scores / self.temp)

        mask = torch.eye(items.size(0), dtype=torch.bool).to(self.device)
        all_scores = torch.mm(users_emb, items_emb.t())
        all_scores = torch.exp(all_scores / self.temp)
        all_scores = all_scores.masked_fill(mask, 0)

        contrast_loss = -torch.log(pos_scores / (all_scores.sum(dim=1) + 1e-8)).mean()
        return contrast_loss

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        ua_embeddings, ia_embeddings = self.forward()

        u_g_embeddings = ua_embeddings[users]
        pos_i_g_embeddings = ia_embeddings[pos_items]
        neg_i_g_embeddings = ia_embeddings[neg_items]

        batch_mf_loss = self.bpr_loss(users, pos_items, neg_items)
        batch_contrast_loss = self.contrast_loss(users, pos_items)
        batch_reg_loss = self.reg_weight * self.reg_loss(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings)

        return batch_mf_loss + self.contrast_weight * batch_contrast_loss + batch_reg_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]

        restore_user_e, restore_item_e = self.forward()
        u_embeddings = restore_user_e[user]

        # Make sure dimensions match for matrix multiplication
        scores = torch.matmul(u_embeddings, restore_item_e.t())
        return scores