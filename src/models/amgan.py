# coding: utf-8
# New Hybrid Model for Multi-modal Recommendation
# Combining aspects of DRAGON, FREEDOM, BM3, and SLMRec to create an advanced model
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from torch.nn.functional import cosine_similarity
from torch_geometric.nn import GATConv, TransformerConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, degree
from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss, L2Loss
from sklearn.cluster import KMeans
from torch_scatter import scatter

class HybridRecModel(GeneralRecommender):
    def __init__(self, config, dataset):
        super(HybridRecModel, self).__init__(config, dataset)
        
        # Configuration and hyperparameters
        self.embedding_dim = config["embedding_size"]
        self.feat_embed_dim = config["feat_embed_dim"]
        self.knn_k = config["knn_k"]
        self.lambda_coeff = config["lambda_coeff"]
        self.reg_weight = config["reg_weight"]
        self.n_layers = config["n_layers"]
        self.dropout_rate = config["dropout"]
        
        self.n_users, self.n_items = dataset.n_users, dataset.n_items
        self.n_nodes = self.n_users + self.n_items

        # Graph adjacency matrix and normalization
        interaction_matrix = dataset.inter_matrix(form="coo").astype(np.float32)
        self.norm_adj = self.get_norm_adj_mat(interaction_matrix).to(self.device)
        self.mm_adj = self.build_multimodal_graph(dataset)
        
        # Embedding layers for users, items, and multi-modal features
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)
        
        if dataset.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(dataset.v_feat, freeze=False)
            self.image_trs = nn.Linear(dataset.v_feat.shape[1], self.feat_embed_dim)
            nn.init.xavier_uniform_(self.image_trs.weight)
        if dataset.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(dataset.t_feat, freeze=False)
            self.text_trs = nn.Linear(dataset.t_feat.shape[1], self.feat_embed_dim)
            nn.init.xavier_uniform_(self.text_trs.weight)
        
        # Defining the multi-modal fusion transformer
        self.multi_modal_transformer = TransformerConv(self.feat_embed_dim, self.embedding_dim, heads=2, concat=False)

        # Defining GAT for propagation
        self.gat = GATConv(self.embedding_dim, self.embedding_dim, heads=2, concat=False, dropout=self.dropout_rate)

        # Predictor for self-supervised learning
        self.predictor = nn.Linear(self.embedding_dim, self.embedding_dim)
        nn.init.xavier_normal_(self.predictor.weight)
        
        self.bpr_loss_fn = BPRLoss()

    def get_norm_adj_mat(self, interaction_matrix):
        """
        Create a normalized adjacency matrix.
        """
        A = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        inter_M = interaction_matrix
        inter_M_t = interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        A._update(data_dict)
        sumArr = (A > 0).sum(axis=1)
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        L = sp.coo_matrix(L)
        row, col = L.row, L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        return torch.sparse.FloatTensor(i, data, torch.Size((self.n_nodes, self.n_nodes)))

    def build_multimodal_graph(self, dataset):
        """
        Create the multi-modal adjacency matrix based on visual and textual features.
        """
        if dataset.v_feat is not None:
            indices, image_adj = self.get_knn_adj_mat(dataset.v_feat)
            mm_adj = image_adj
        if dataset.t_feat is not None:
            indices, text_adj = self.get_knn_adj_mat(dataset.t_feat)
            mm_adj = text_adj if mm_adj is None else 0.5 * image_adj + 0.5 * text_adj
        return mm_adj

    def get_knn_adj_mat(self, mm_embeddings):
        """
        Create KNN adjacency matrix for multi-modal features.
        """
        context_norm = mm_embeddings.div(torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True))
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        _, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
        adj_size = sim.size()
        indices0 = torch.arange(knn_ind.shape[0]).to(self.device).unsqueeze(1).expand(-1, self.knn_k)
        indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)
        return indices, self.compute_normalized_laplacian(indices, adj_size)

    def compute_normalized_laplacian(self, indices, adj_size):
        """
        Normalize the Laplacian for graph construction.
        """
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return torch.sparse.FloatTensor(indices, values, adj_size)

    def forward(self, users, items):
        user_emb = self.user_embedding(users)
        item_emb = self.item_id_embedding(items)
        item_features = self.multi_modal_transformer(item_emb, self.mm_adj)
        item_features = F.normalize(item_features, p=2, dim=-1)
        user_rep = self.gat(user_emb, self.norm_adj)
        return user_rep, item_features

    def calculate_loss(self, interaction):
        users, pos_items, neg_items = interaction[0], interaction[1], interaction[2]
        u_g_embeddings, i_g_embeddings = self.forward(users, pos_items)
        u_g_neg_embeddings, i_g_neg_embeddings = self.forward(users, neg_items)
        loss_bpr = self.bpr_loss_fn(u_g_embeddings, i_g_embeddings, u_g_neg_embeddings)
        return loss_bpr

    def full_sort_predict(self, interaction):
        user = interaction[0]
        u_g_embeddings, i_g_embeddings = self.forward(user, torch.arange(self.n_items).to(self.device))
        scores = torch.matmul(u_g_embeddings, i_g_embeddings.t())
        return scores

    def get_embedding(self, users, pos_items, neg_items):
        user_emb, item_emb = self.forward(users, pos_items)
        return user_emb, item_emb
