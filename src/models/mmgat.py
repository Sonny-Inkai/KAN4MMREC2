# coding: utf-8
# @email: enoche.chow@gmail.com

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


class GATConv(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.1, alpha=0.2):
        super(GATConv, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Linear(2 * out_features, 1)
        
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a.weight)

    def forward(self, input, adj):
        h = self.W(input)
        N = h.size()[0]

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, N, 2 * self.out_features)
        e = F.leaky_relu(self.a(a_input).squeeze(2), negative_slope=self.alpha)
        
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout)
        
        h_prime = torch.matmul(attention, h)
        return h_prime


class MMGAT(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MMGAT, self).__init__(config, dataset)
        
        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.n_layers = config['n_mm_layers']
        self.n_heads = config['n_heads']
        self.dropout = config['dropout']
        self.knn_k = config['knn_k']
        self.reg_weight = config['reg_weight']
        self.lambda_coeff = config['lambda_coeff']
        
        self.n_nodes = self.n_users + self.n_items
        
        # Load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.norm_adj = self.get_norm_adj_mat().to(self.device)
        self.edge_indices, self.edge_values = self.get_edge_info()
        self.edge_indices = self.edge_indices.to(self.device)
        self.edge_values = self.edge_values.to(self.device)
        
        # Embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)
        
        # Modal Transformation
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)
            
        # Graph Attention Layers
        self.gat_layers = nn.ModuleList([
            GATConv(self.embedding_dim, self.embedding_dim, dropout=self.dropout)
            for _ in range(self.n_layers)
        ])
        
        # Modal Fusion
        self.modal_weight = nn.Parameter(torch.Tensor([0.5, 0.5]))
        self.softmax = nn.Softmax(dim=0)
        
        # Initialize modal adjacency
        self.mm_adj = self.init_modal_adj()
        
    def init_modal_adj(self):
        dataset_path = os.path.abspath(self.config['data_path'] + self.config['dataset'])
        mm_adj_file = os.path.join(dataset_path, f'mm_adj_mmgat_{self.knn_k}.pt')
        
        if os.path.exists(mm_adj_file):
            return torch.load(mm_adj_file)
            
        if self.v_feat is not None:
            image_adj = self.build_modal_graph(self.image_embedding.weight)
            mm_adj = image_adj
        if self.t_feat is not None:
            text_adj = self.build_modal_graph(self.text_embedding.weight)
            mm_adj = text_adj
        if self.v_feat is not None and self.t_feat is not None:
            weight = self.softmax(self.modal_weight)
            mm_adj = weight[0] * image_adj + weight[1] * text_adj
            
        torch.save(mm_adj, mm_adj_file)
        return mm_adj.to(self.device)
        
    def build_modal_graph(self, embeddings):
        sim_matrix = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2)
        topk_values, _ = torch.topk(sim_matrix, self.knn_k, dim=1)
        threshold = topk_values[:, -1].view(-1, 1)
        adj = (sim_matrix >= threshold).float()
        return compute_normalized_laplacian(adj)
        
    def get_norm_adj_mat(self):
        A = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        A._update(data_dict)
        
        sumArr = (A > 0).sum(axis=1)
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        L = sp.coo_matrix(L)
        indices = torch.LongTensor(np.array([L.row, L.col]))
        values = torch.FloatTensor(L.data)
        return torch.sparse.FloatTensor(indices, values, torch.Size((self.n_nodes, self.n_nodes)))

    def get_edge_info(self):
        rows = torch.from_numpy(self.interaction_matrix.row)
        cols = torch.from_numpy(self.interaction_matrix.col)
        edges = torch.stack([rows, cols]).type(torch.LongTensor)
        values = torch.ones(edges.size(1))
        return edges, values
        
    def forward(self, adj):
        # Multimodal Feature Learning
        if self.v_feat is not None:
            image_feats = self.image_trs(self.image_embedding.weight)
        if self.t_feat is not None:
            text_feats = self.text_trs(self.text_embedding.weight)
            
        if self.v_feat is not None and self.t_feat is not None:
            weight = self.softmax(self.modal_weight)
            item_feats = weight[0] * image_feats + weight[1] * text_feats
        else:
            item_feats = image_feats if self.v_feat is not None else text_feats
            
        # Graph Attention on Item Features
        h = item_feats
        for gat_layer in self.gat_layers:
            h = gat_layer(h, self.mm_adj)
            
        # User-Item Graph Learning
        ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
        all_embeddings = [ego_embeddings]
        
        for _ in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(adj, ego_embeddings)
            all_embeddings.append(ego_embeddings)
            
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        
        # Combine item embeddings with multimodal features
        i_g_embeddings = i_g_embeddings + F.normalize(h, p=2, dim=1)
        
        return u_g_embeddings, i_g_embeddings
        
    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        
        u_embeddings, i_embeddings = self.forward(self.norm_adj)
        
        u_embeddings = u_embeddings[users]
        pos_embeddings = i_embeddings[pos_items]
        neg_embeddings = i_embeddings[neg_items]
        
        # BPR Loss
        pos_scores = torch.sum(torch.mul(u_embeddings, pos_embeddings), dim=1)
        neg_scores = torch.sum(torch.mul(u_embeddings, neg_embeddings), dim=1)
        
        mf_loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores)))
        
        # Regularization Loss
        reg_loss = self.reg_weight * (
            torch.norm(u_embeddings) ** 2 +
            torch.norm(pos_embeddings) ** 2 +
            torch.norm(neg_embeddings) ** 2
        ) / (2 * len(users))
        
        return mf_loss + reg_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        
        u_embeddings, i_embeddings = self.forward(self.norm_adj)
        u_embeddings = u_embeddings[user]
        
        scores = torch.matmul(u_embeddings, i_embeddings.transpose(0, 1))
        return scores