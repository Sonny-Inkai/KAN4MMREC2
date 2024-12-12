# coding: utf-8
import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from common.abstract_recommender import GeneralRecommender

class ModalTower(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
    def forward(self, x):
        return self.net(x)

class MMGAT(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MMGAT, self).__init__(config, dataset)
        
        self.embedding_dim = config["embedding_size"]
        self.feat_embed_dim = config["feat_embed_dim"]
        self.n_layers = config["n_mm_layers"]
        self.reg_weight = config["reg_weight"]
        self.knn_k = config["knn_k"]
        self.n_nodes = self.n_users + self.n_items
        
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_normal_(self.user_embedding.weight)
        nn.init.xavier_normal_(self.item_embedding.weight)
        
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_tower = ModalTower(self.v_feat.shape[1], self.feat_embed_dim * 2, self.feat_embed_dim)
            
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_tower = ModalTower(self.t_feat.shape[1], self.feat_embed_dim * 2, self.feat_embed_dim)
        
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.norm_adj = None
        self.to(self.device)
        self.build_graph_structure()

    def build_graph_structure(self):
        A = sp.dok_matrix((self.n_nodes, self.n_nodes), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        for key, value in data_dict.items():
            A[key] = value
        
        rowsum = np.array(A.sum(axis=1))
        d_inv = np.power(rowsum + 1e-7, -0.5).flatten()
        D = sp.diags(d_inv)
        L = D * A * D
        L = sp.coo_matrix(L)
        
        indices = np.vstack([L.row, L.col])
        i = torch.LongTensor(indices).to(self.device)
        v = torch.FloatTensor(L.data).to(self.device)
        self.norm_adj = torch.sparse_coo_tensor(i, v, torch.Size(L.shape))
        
        if self.v_feat is not None:
            self.image_adj = self.build_knn_graph(self.v_feat)
        if self.t_feat is not None:
            self.text_adj = self.build_knn_graph(self.t_feat)

    def build_knn_graph(self, features):
        features = features.to(self.device)
        sim = torch.matmul(F.normalize(features, dim=1), F.normalize(features, dim=1).t())
        values, indices = torch.topk(sim, k=self.knn_k, dim=1)
        rows = torch.arange(features.size(0), device=self.device).view(-1, 1).expand_as(indices).flatten()
        cols = indices.flatten()
        edge_index = torch.stack([rows, cols], dim=0)
        edge_weight = values.flatten()
        
        adj = torch.sparse_coo_tensor(
            edge_index, edge_weight, 
            (features.size(0), features.size(0))
        )
        return adj.coalesce()

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(users * pos_items, dim=1)
        neg_scores = torch.sum(users * neg_items, dim=1)
        return -torch.mean(F.logsigmoid(pos_scores - neg_scores))

    def forward(self, adj):
        v_feat, t_feat = None, None
        
        if self.v_feat is not None:
            v_feat = self.image_tower(self.image_embedding.weight)
            v_feat = torch.sparse.mm(self.image_adj, v_feat)
        
        if self.t_feat is not None:
            t_feat = self.text_tower(self.text_embedding.weight)
            t_feat = torch.sparse.mm(self.text_adj, t_feat)
        
        item_feat = torch.zeros_like(self.item_embedding.weight)
        if v_feat is not None:
            item_feat += v_feat
        if t_feat is not None:
            item_feat += t_feat
        if v_feat is not None and t_feat is not None:
            item_feat = item_feat / 2
            
        ego_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        all_embeddings = [ego_embeddings]
        
        for _ in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(adj, ego_embeddings)
            all_embeddings.append(ego_embeddings)
            
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1)
        
        user_all, item_all = torch.split(all_embeddings, [self.n_users, self.n_items])
        item_all = item_all + item_feat
        
        return user_all, item_all

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        
        ua_embeddings, ia_embeddings = self.forward(self.norm_adj)
        
        u_g_embeddings = ua_embeddings[users]
        pos_i_g_embeddings = ia_embeddings[pos_items]
        neg_i_g_embeddings = ia_embeddings[neg_items]
        
        batch_mf_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings)
        
        mf_v_loss, mf_t_loss = 0.0, 0.0
        if self.v_feat is not None:
            v_feat = self.image_tower(self.image_embedding.weight)
            mf_v_loss = self.bpr_loss(ua_embeddings[users], v_feat[pos_items], v_feat[neg_items])
            
        if self.t_feat is not None:
            t_feat = self.text_tower(self.text_embedding.weight)
            mf_t_loss = self.bpr_loss(ua_embeddings[users], t_feat[pos_items], t_feat[neg_items])
        
        return batch_mf_loss + self.reg_weight * (mf_v_loss + mf_t_loss)

    def full_sort_predict(self, interaction):
        user = interaction[0]
        restore_user_e, restore_item_e = self.forward(self.norm_adj)
        scores = torch.matmul(restore_user_e[user], restore_item_e.transpose(0, 1))
        return scores