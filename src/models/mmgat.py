import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from common.abstract_recommender import GeneralRecommender

class MMGAT(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MMGAT, self).__init__(config, dataset)

        # Model dimensions
        self.embedding_dim = config["embedding_size"]
        self.feat_embed_dim = config["feat_embed_dim"]
        self.n_layers = config["n_mm_layers"]
        self.dropout = config["dropout"]
        self.reg_weight = config["reg_weight"]
        self.knn_k = config["knn_k"]
        self.batch_size = config["train_batch_size"]
        self.temperature = 0.2

        # Initialize embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        # Initialize modal encoders
        if self.v_feat is not None:
            self.image_encoder = nn.Sequential(
                nn.Linear(self.v_feat.shape[1], self.feat_embed_dim),
                nn.LayerNorm(self.feat_embed_dim),
                nn.GELU(),
                nn.Dropout(self.dropout)
            )

        if self.t_feat is not None:
            self.text_encoder = nn.Sequential(
                nn.Linear(self.t_feat.shape[1], self.feat_embed_dim),
                nn.LayerNorm(self.feat_embed_dim),
                nn.GELU(),
                nn.Dropout(self.dropout)
            )

        # Modal fusion for multi-modal case
        if self.v_feat is not None and self.t_feat is not None:
            self.modal_fusion = nn.Sequential(
                nn.Linear(self.feat_embed_dim * 2, self.feat_embed_dim),
                nn.LayerNorm(self.feat_embed_dim),
                nn.GELU(),
                nn.Dropout(self.dropout)
            )
            self.modal_weight = nn.Parameter(torch.ones(2))

        # Graph layers
        self.gat_layers = nn.ModuleList([
            GATConv(self.embedding_dim, self.embedding_dim // 8, heads=8, dropout=self.dropout)
            for _ in range(self.n_layers)
        ])

        # Build interaction matrix and edges
        self.norm_adj = self.get_norm_adj_mat().to(self.device)
        self.edge_index = self.build_edges().to(self.device)
        self.mm_edge_index = None
        self.build_modal_graph()

    def get_norm_adj_mat(self):
        # Get interaction matrix
        inter_mat = self.dataset.inter_matrix(form='coo').astype(np.float32)
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        
        # Build adjacency matrix
        adj_mat = adj_mat.tolil()
        R = inter_mat.tolil()
        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()

        # Normalize adjacency matrix
        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum + 1e-7, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        norm_adj = d_mat.dot(adj_mat).dot(d_mat)

        # Convert to PyTorch sparse tensor
        norm_adj = norm_adj.tocoo()
        values = norm_adj.data
        indices = np.vstack((norm_adj.row, norm_adj.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = torch.Size(norm_adj.shape)
        return torch.sparse.FloatTensor(i, v, shape)

    def build_edges(self):
        inter_mat = self.dataset.inter_matrix(form='coo').astype(np.float32)
        rows = torch.from_numpy(inter_mat.row)
        cols = torch.from_numpy(inter_mat.col) + self.n_users
        
        edge_index = torch.stack([
            torch.cat([rows, cols]),
            torch.cat([cols, rows])
        ])
        
        return edge_index

    def build_modal_graph(self):
        if not (self.v_feat is not None or self.t_feat is not None):
            return

        features = None
        if self.v_feat is not None and self.t_feat is not None:
            v_emb = F.normalize(self.image_encoder(self.v_feat), p=2, dim=1)
            t_emb = F.normalize(self.text_encoder(self.t_feat), p=2, dim=1)
            weights = F.softmax(self.modal_weight, dim=0)
            features = weights[0] * v_emb + weights[1] * t_emb
        else:
            feat = self.v_feat if self.v_feat is not None else self.t_feat
            encoder = self.image_encoder if self.v_feat is not None else self.text_encoder
            features = F.normalize(encoder(feat), p=2, dim=1)

        # Build KNN graph in batches
        n_items = features.size(0)
        indices_list = []
        values_list = []

        for i in range(0, n_items, self.batch_size):
            batch_feat = features[i:min(i+self.batch_size, n_items)]
            sim = torch.matmul(batch_feat, features.t())
            topk_values, topk_indices = sim.topk(k=min(self.knn_k, n_items), dim=1)
            
            batch_indices = torch.arange(i, min(i+self.batch_size, n_items), 
                                       device=self.device).view(-1, 1)
            batch_indices = batch_indices.expand(-1, topk_indices.size(1))
            
            indices_list.append(torch.stack([batch_indices.reshape(-1), 
                                           topk_indices.reshape(-1)]))
            values_list.append(F.normalize(topk_values.reshape(-1), p=1, dim=0))

        self.mm_edge_index = torch.cat(indices_list, dim=1)
        self.mm_edge_weight = torch.cat(values_list)

    def forward(self):
        # Process multimodal features
        modal_emb = None
        if self.v_feat is not None or self.t_feat is not None:
            if self.v_feat is not None and self.t_feat is not None:
                v_emb = self.image_encoder(self.v_feat)
                t_emb = self.text_encoder(self.t_feat)
                modal_emb = self.modal_fusion(torch.cat([v_emb, t_emb], dim=1))
            else:
                feat = self.v_feat if self.v_feat is not None else self.t_feat
                encoder = self.image_encoder if self.v_feat is not None else self.text_encoder
                modal_emb = encoder(feat)
            modal_emb = F.normalize(modal_emb, p=2, dim=1)

        # Process graph structure
        x = torch.cat([self.user_embedding.weight, self.item_embedding.weight])
        h = x

        for layer in self.gat_layers:
            h = layer(h, self.edge_index)
            h = F.normalize(h, p=2, dim=1)

        user_emb, item_emb = torch.split(h, [self.n_users, self.n_items])

        if modal_emb is not None:
            item_emb = item_emb + modal_emb

        return user_emb, item_emb

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        user_emb, item_emb = self.forward()

        u_emb = user_emb[users]
        pos_emb = item_emb[pos_items]
        neg_emb = item_emb[neg_items]

        # BPR loss with temperature scaling
        pos_scores = torch.sum(u_emb * pos_emb, dim=1) / self.temperature
        neg_scores = torch.sum(u_emb * neg_emb, dim=1) / self.temperature

        bpr_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))

        # L2 regularization
        reg_loss = self.reg_weight * (
            torch.norm(u_emb) +
            torch.norm(pos_emb) +
            torch.norm(neg_emb)
        )

        return bpr_loss + reg_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_emb, item_emb = self.forward()
        scores = torch.matmul(user_emb[user], item_emb.transpose(0, 1))
        return scores