# phoenix.py
import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
# bug
from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss
from utils.utils import build_sim, compute_normalized_laplacian

class PHOENIX(GeneralRecommender):
    def __init__(self, config, dataset):
        super(PHOENIX, self).__init__(config, dataset)

        self.embedding_dim = 128
        self.feat_embed_dim = 128
        self.knn_k = 10
        self.lambda_coeff = 0.001
        self.n_layers = 2
        self.reg_weight = 0.001
        self.contrast_weight = 0.001
        self.dropout = 0.5
        self.mm_fusion_mode = "concat"
        self.n_ui_layers = 2
        self.temp = 0.1
        
        self.n_nodes = self.n_users + self.n_items

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form="coo").astype(np.float32)
        self.norm_adj = self.get_norm_adj_mat().to(self.device)

        # Initialize user/item ID embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        # Initialize modality feature transformations
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)

        # Personalized Interest & Social Context Modeling
        self.user_preference = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_attribute = nn.Embedding(self.n_items, self.embedding_dim)
        self.user_social_trs = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.item_social_trs = nn.Linear(self.embedding_dim, self.embedding_dim)

        # Cross-modal contrastive learning
        self.predictor = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.reg_loss = EmbLoss()

        # Build item-item graph structure
        self.build_item_graph()

    def build_item_graph(self):
        if self.v_feat is not None:
            self.image_adj = self.get_knn_adj_mat(self.image_embedding.weight)
        if self.t_feat is not None:
            self.text_adj = self.get_knn_adj_mat(self.text_embedding.weight)

    def get_knn_adj_mat(self, mm_embeddings):
        # Calculate similarity matrix
        sim_mat = build_sim(mm_embeddings)
        sim_mat = sim_mat.to(self.device)
        
        # Get top k similar items 
        vals, inds = torch.topk(sim_mat, self.knn_k)
        
        # Create sparse adjacency matrix
        row_inds = torch.arange(sim_mat.size(0), device=self.device).view(-1, 1).expand(-1, self.knn_k).reshape(-1)
        col_inds = inds.reshape(-1)
        
        indices = torch.stack([row_inds, col_inds])
        values = vals.reshape(-1)
        
        adj = torch.sparse_coo_tensor(indices, values, sim_mat.size(), device=self.device)
        
        # Normalize adjacency matrix
        rowsum = torch.sparse.sum(adj, dim=1).to_dense()
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        
        adj_dense = adj.to_dense()
        norm_adj = torch.mm(torch.mm(d_mat_inv_sqrt, adj_dense), d_mat_inv_sqrt)
        
        return norm_adj.to_sparse()

    def get_norm_adj_mat(self):
        # Build normalized user-item interaction graph
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
        h_a = self.item_attribute.weight

        # Perform graph convolutions on item-item graph
        if self.v_feat is not None:
            h_v = h
            for _ in range(self.n_layers):
                h_v = torch.sparse.mm(self.image_adj, h_v)
        if self.t_feat is not None:  
            h_t = h
            for _ in range(self.n_layers):
                h_t = torch.sparse.mm(self.text_adj, h_t)

        # Perform graph convolutions on user-item graph
        ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
        all_embeddings = [ego_embeddings]
        for _ in range(self.n_ui_layers):
            ego_embeddings = torch.sparse.mm(self.norm_adj, ego_embeddings)
            all_embeddings.append(ego_embeddings)
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)

        # Fuse multi-modal representations
        if self.v_feat is not None and self.t_feat is not None:
            i_embeddings = (i_g_embeddings + h_v + h_t + h_a) / 4
        elif self.v_feat is not None:
            i_embeddings = (i_g_embeddings + h_v + h_a) / 3
        elif self.t_feat is not None:
            i_embeddings = (i_g_embeddings + h_t + h_a) / 3
        else:
            i_embeddings = (i_g_embeddings + h_a) / 2

        # Apply dropout
        u_g_embeddings = F.dropout(u_g_embeddings, p=self.dropout, training=self.training)
        i_embeddings = F.dropout(i_embeddings, p=self.dropout, training=self.training)

        # Incorporate personalized interest and social context
        u_pref_embeddings = self.user_preference.weight  
        u_social_embeddings = self.user_social_trs(u_g_embeddings)

        i_attr_embeddings = self.item_attribute.weight
        i_social_embeddings = self.item_social_trs(i_embeddings)

        u_embeddings = u_g_embeddings + u_pref_embeddings + u_social_embeddings  
        i_embeddings = i_embeddings + i_attr_embeddings + i_social_embeddings

        return u_embeddings, i_embeddings