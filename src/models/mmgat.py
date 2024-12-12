# coding: utf-8
import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from common.abstract_recommender import GeneralRecommender

class MMGAT(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MMGAT, self).__init__(config, dataset)
        
        self.embedding_dim = config["embedding_size"]
        self.feat_embed_dim = config["feat_embed_dim"]
        self.n_layers = config["n_mm_layers"]
        self.n_ui_layers = config["n_ui_layers"]
        self.reg_weight = config["reg_weight"]
        self.dropout = config["dropout"]
        self.build_item_graph = True
        
        # Embeddings initialization
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        
        # Modality encoders
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_trs = nn.Sequential(
                nn.Linear(self.v_feat.shape[1], self.feat_embed_dim),
                nn.LayerNorm(self.feat_embed_dim),
                nn.PReLU(),
                nn.Dropout(self.dropout)
            )
            
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_trs = nn.Sequential(
                nn.Linear(self.t_feat.shape[1], self.feat_embed_dim),
                nn.LayerNorm(self.feat_embed_dim),
                nn.PReLU(),
                nn.Dropout(self.dropout)
            )
        
        # Load interaction info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.masked_adj = None
        self.edge_indices, self.edge_values = self.get_edge_info()
        self.mm_adj = self.build_modal_graph()
        
        self.to(self.device)

    def get_edge_info(self):
        rows = torch.from_numpy(self.interaction_matrix.row)
        cols = torch.from_numpy(self.interaction_matrix.col)
        edges = torch.stack([rows, cols]).type(torch.LongTensor)
        values = self._normalize_adj_values(edges)
        return edges.to(self.device), values.to(self.device)

    def _normalize_adj_values(self, indices):
        adj = sp.coo_matrix(
            (np.ones(indices.shape[1]), (indices[0].numpy(), indices[1].numpy())),
            shape=(self.n_users, self.n_items)
        )
        rowsum = np.array(adj.sum(1))
        colsum = np.array(adj.sum(0))
        d_inv_sqrt_row = np.power(rowsum + 1e-7, -0.5).flatten()
        d_inv_sqrt_col = np.power(colsum + 1e-7, -0.5).flatten()
        r_mat_inv_sqrt = sp.diags(d_inv_sqrt_row)
        c_mat_inv_sqrt = sp.diags(d_inv_sqrt_col)
        norm_adj = r_mat_inv_sqrt.dot(adj).dot(c_mat_inv_sqrt)
        return torch.FloatTensor(norm_adj.data)

    def build_modal_graph(self):
        if self.v_feat is not None and self.t_feat is not None:
            v_sim = self._build_knn_graph(self.v_feat)
            t_sim = self._build_knn_graph(self.t_feat)
            return 0.5 * (v_sim + t_sim)
        elif self.v_feat is not None:
            return self._build_knn_graph(self.v_feat)
        else:
            return self._build_knn_graph(self.t_feat)

    def _build_knn_graph(self, features, k=10):
        features = F.normalize(features, p=2, dim=1)
        sim = torch.mm(features, features.t())
        _, indices = sim.topk(k=k, dim=-1)
        rows = torch.arange(features.size(0), device=self.device).unsqueeze(1).expand_as(indices)
        adj = torch.zeros_like(sim)
        adj[rows.reshape(-1), indices.reshape(-1)] = 1
        adj = (adj + adj.t()) / 2
        
        d_inv_sqrt = torch.pow(adj.sum(dim=1) + 1e-7, -0.5)
        adj = d_inv_sqrt.unsqueeze(-1) * adj * d_inv_sqrt.unsqueeze(0)
        return adj.to(self.device)

    def pre_epoch_processing(self):
        if self.dropout <= 0.0:
            self.masked_adj = self.get_sparse_adj_mat()
            return
            
        degree_len = int(self.edge_values.size(0) * (1.0 - self.dropout))
        keep_idx = torch.multinomial(self.edge_values, degree_len)
        keep_indices = self.edge_indices[:, keep_idx]
        
        indices = torch.cat([keep_indices, keep_indices.flip(0)], dim=1)
        values = torch.ones(indices.size(1)).to(self.device)
        self.masked_adj = self._sparse_dropout(indices, values)

    def _sparse_dropout(self, indices, values):
        adj = torch.sparse_coo_tensor(
            indices, 
            values,
            (self.n_users + self.n_items, self.n_users + self.n_items)
        )
        row_sum = 1e-7 + torch.sparse.sum(adj, dim=1).to_dense()
        d_inv_sqrt = torch.pow(row_sum, -0.5)
        values = d_inv_sqrt[indices[0]] * d_inv_sqrt[indices[1]]
        return torch.sparse_coo_tensor(indices, values, adj.size())

    def forward(self):
        # Process modalities
        if self.v_feat is not None:
            image_feat = self.image_trs(self.image_embedding.weight)
        if self.t_feat is not None:
            text_feat = self.text_trs(self.text_embedding.weight)
            
        # Update item embeddings with modal information
        h = self.item_embedding.weight
        for _ in range(self.n_layers):
            h = F.normalize(torch.mm(self.mm_adj, h))
            
        # User-item graph convolution
        ego_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        all_embeddings = [ego_embeddings]
        
        for _ in range(self.n_ui_layers):
            ego_embeddings = torch.sparse.mm(self.masked_adj, ego_embeddings)
            all_embeddings.append(ego_embeddings)
            
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items])
        
        return (
            u_g_embeddings, 
            i_g_embeddings + h, 
            image_feat if self.v_feat is not None else None,
            text_feat if self.t_feat is not None else None
        )

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        ua_embeddings, ia_embeddings, image_feat, text_feat = self.forward()

        u_g_embeddings = ua_embeddings[users]
        pos_i_g_embeddings = ia_embeddings[pos_items]
        neg_i_g_embeddings = ia_embeddings[neg_items]

        # Main BPR loss
        pos_scores = torch.sum(u_g_embeddings * pos_i_g_embeddings, dim=1)
        neg_scores = torch.sum(u_g_embeddings * neg_i_g_embeddings, dim=1)
        mf_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))

        # Modal-specific losses
        modal_loss = 0.0
        if self.v_feat is not None:
            modal_loss += -torch.mean(F.logsigmoid(
                torch.sum(ua_embeddings[users] * image_feat[pos_items], dim=1) -
                torch.sum(ua_embeddings[users] * image_feat[neg_items], dim=1)
            ))
            
        if self.t_feat is not None:
            modal_loss += -torch.mean(F.logsigmoid(
                torch.sum(ua_embeddings[users] * text_feat[pos_items], dim=1) -
                torch.sum(ua_embeddings[users] * text_feat[neg_items], dim=1)
            ))

        # Regularization
        reg_loss = self.reg_weight * (
            torch.norm(u_g_embeddings) +
            torch.norm(pos_i_g_embeddings) +
            torch.norm(neg_i_g_embeddings)
        )

        return mf_loss + 0.1 * modal_loss + reg_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        restore_user_e, restore_item_e, _, _ = self.forward()
        scores = torch.matmul(restore_user_e[user], restore_item_e.transpose(0, 1))
        return scores