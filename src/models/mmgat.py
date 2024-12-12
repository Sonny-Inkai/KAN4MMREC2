# coding: utf-8
import os
import random
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss, L2Loss

class LightGCNConv(nn.Module):
    def __init__(self):
        super(LightGCNConv, self).__init__()
    
    def forward(self, x, edge_index, edge_weight=None):
        row, col = edge_index
        deg = torch.sparse.sum(torch.sparse_coo_tensor(edge_index, torch.ones_like(row).to(x.device), (x.size(0), x.size(0))), dim=1).to_dense()
        deg_inv_sqrt = torch.pow(deg, -0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        return torch.sparse.mm(torch.sparse_coo_tensor(edge_index, norm, (x.size(0), x.size(0))), x)

class ModalGAT(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super(ModalGAT, self).__init__()
        self.gat = GATConv(in_dim, out_dim, heads=4, concat=False, dropout=dropout)
        self.norm = nn.LayerNorm(out_dim)
        self.prelu = nn.PReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index):
        x = self.dropout(x)
        x = self.gat(x, edge_index)
        x = self.norm(x)
        return self.prelu(x)

class ModalEncoder(nn.Module):
    def __init__(self, feat_dim, hidden_dim, embed_dim, n_layers, dropout=0.1):
        super(ModalEncoder, self).__init__()
        self.transform = nn.Linear(feat_dim, hidden_dim)
        self.gat_layers = nn.ModuleList([
            ModalGAT(hidden_dim, hidden_dim, dropout)
            for _ in range(n_layers)
        ])
        self.projection = nn.Linear(hidden_dim, embed_dim)
        
    def forward(self, features, edge_index):
        x = self.transform(features)
        for gat in self.gat_layers:
            x = x + gat(x, edge_index)
        return F.normalize(self.projection(x))

class MMGAT(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MMGAT, self).__init__(config, dataset)
        
        self.embedding_dim = config["embedding_size"]
        self.feat_embed_dim = config["feat_embed_dim"]
        self.hidden_dim = self.feat_embed_dim * 2
        self.n_layers = config["n_mm_layers"]
        self.n_ui_layers = config["n_ui_layers"]
        self.reg_weight = config["reg_weight"]
        self.knn_k = config["knn_k"]
        self.lambda_coeff = config["lambda_coeff"]
        self.dropout = config["dropout"]
        self.temperature = 0.07
        self.mm_image_weight = config["mm_image_weight"]
        
        self.n_nodes = self.n_users + self.n_items
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        
        # Get normalized adjacency matrix
        self.norm_adj = self.get_norm_adj_mat().to(self.device)
        
        # User-Item Graph Construction
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_normal_(self.user_embedding.weight)
        nn.init.xavier_normal_(self.item_id_embedding.weight)
        
        # Modal Encoders
        self.image_adj = None
        self.text_adj = None
        
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_encoder = ModalEncoder(
                self.v_feat.shape[1], self.hidden_dim, self.feat_embed_dim, 
                self.n_layers, self.dropout
            )
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)
            _, self.image_adj = self.get_knn_adj_mat(self.image_embedding.weight.detach())
            
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_encoder = ModalEncoder(
                self.t_feat.shape[1], self.hidden_dim, self.feat_embed_dim,
                self.n_layers, self.dropout
            )
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)
            _, self.text_adj = self.get_knn_adj_mat(self.text_embedding.weight.detach())
        
        # Modal Fusion
        self.fusion = nn.Sequential(
            nn.Linear(self.feat_embed_dim * 2, self.feat_embed_dim),
            nn.LayerNorm(self.feat_embed_dim),
            nn.PReLU(),
            nn.Dropout(self.dropout)
        )
        
        self.highway = nn.Sequential(
            nn.Linear(self.feat_embed_dim * 2, self.feat_embed_dim),
            nn.Sigmoid()
        )
        
        self.modal_weight = nn.Parameter(torch.Tensor([0.5, 0.5]))
        self.softmax = nn.Softmax(dim=0)
        
        # Loss weights
        self.ssl_weight = config['ssl_weight']
        self.build_item_graph = True
        
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
        return torch.sparse.FloatTensor(i, data, torch.Size((self.n_nodes, self.n_nodes)))

    def get_knn_adj_mat(self, mm_embeddings):
        context_norm = mm_embeddings.div(torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True))
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        _, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
        adj_size = sim.size()
        del sim
        indices0 = torch.arange(knn_ind.shape[0]).to(self.device)
        indices0 = torch.unsqueeze(indices0, 1)
        indices0 = indices0.expand(-1, self.knn_k)
        indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)
        return indices, self.compute_normalized_laplacian(indices, adj_size)

    def compute_normalized_laplacian(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return torch.sparse.FloatTensor(indices, values, adj_size)

    def forward(self, build_item_graph=True):
        # Process modalities and build MM graph
        h = self.item_id_embedding.weight
        weight = self.softmax(self.modal_weight)
        
        if build_item_graph:
            if self.v_feat is not None and self.t_feat is not None:
                image_feats = self.image_trs(self.image_embedding.weight)
                text_feats = self.text_trs(self.text_embedding.weight)
                self.mm_adj = weight[0] * self.image_adj + weight[1] * self.text_adj
            elif self.v_feat is not None:
                image_feats = self.image_trs(self.image_embedding.weight)
                self.mm_adj = self.image_adj
            elif self.t_feat is not None:
                text_feats = self.text_trs(self.text_embedding.weight)
                self.mm_adj = self.text_adj
        
        # Apply MM graph convolution
        for i in range(self.n_layers):
            h = torch.sparse.mm(self.mm_adj, h)
        
        # Process UI graph
        ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
        all_embeddings = [ego_embeddings]
        
        # Apply UI graph convolution
        for _ in range(self.n_ui_layers):
            ego_embeddings = torch.sparse.mm(self.norm_adj, ego_embeddings)
            all_embeddings.append(ego_embeddings)
            
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        
        return u_g_embeddings, i_g_embeddings + h

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        u_embeddings, i_embeddings = self.forward(build_item_graph=self.build_item_graph)
        self.build_item_graph = False

        u_g_embeddings = u_embeddings[users]
        pos_i_g_embeddings = i_embeddings[pos_items]
        neg_i_g_embeddings = i_embeddings[neg_items]

        pos_scores = torch.sum(torch.mul(u_g_embeddings, pos_i_g_embeddings), dim=1)
        neg_scores = torch.sum(torch.mul(u_g_embeddings, neg_i_g_embeddings), dim=1)

        mf_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))

        # Modal Contrastive Loss
        modal_loss = 0.0
        if self.v_feat is not None and self.t_feat is not None:
            image_embeddings = self.image_encoder(self.image_embedding.weight, self.image_adj._indices())
            text_embeddings = self.text_encoder(self.text_embedding.weight, self.text_adj._indices())
            
            pos_image_embeddings = image_embeddings[pos_items]
            pos_text_embeddings = text_embeddings[pos_items]
            
            modal_loss = -torch.mean(F.logsigmoid(
                torch.sum(pos_image_embeddings * pos_text_embeddings, dim=1) / self.temperature
            ))

        # Regularization
        reg_loss = self.reg_weight * (
            torch.norm(u_g_embeddings) +
            torch.norm(pos_i_g_embeddings) +
            torch.norm(neg_i_g_embeddings)
        )

        return mf_loss + modal_loss + reg_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        restore_user_e, restore_item_e = self.forward(build_item_graph=True)
        u_embeddings = restore_user_e[user]
        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
        return scores