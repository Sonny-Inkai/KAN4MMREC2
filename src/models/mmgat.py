# coding: utf-8
import os
import random
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GATv2Conv
from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss, L2Loss
from utils.utils import build_sim, compute_normalized_laplacian

class CrossModalGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=4):
        super(CrossModalGATLayer, self).__init__()
        self.gat1 = GATv2Conv(in_dim, out_dim // num_heads, heads=num_heads, concat=True)
        self.gat2 = GATv2Conv(out_dim, out_dim, heads=1, concat=False)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(0.2)
        self.activation = nn.PReLU()
        
    def forward(self, x, edge_index):
        residual = x
        x = self.gat1(x, edge_index)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.gat2(x, edge_index)
        x = x + residual
        x = self.norm(x)
        return x

class MMGAT(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MMGAT, self).__init__(config, dataset)
        
        # Basic configurations
        self.embedding_dim = config["embedding_size"]
        self.feat_embed_dim = config["feat_embed_dim"]
        self.n_layers = config["n_mm_layers"]
        self.n_heads = 4
        self.dropout = config["dropout"]
        self.knn_k = config["knn_k"]
        self.reg_weight = config["reg_weight"]
        self.mm_image_weight = config["mm_image_weight"]
        self.temperature = 0.2
        self.n_nodes = self.n_users + self.n_items
        self.alpha = 0.2  # Weight for contrastive loss
        
        # Load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.norm_adj = self.get_norm_adj_mat(dataset.inter_matrix(form='coo').astype(np.float32)).to(self.device)
        
        # Initialize embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        # Modal feature processing
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_projection = nn.Sequential(
                nn.Linear(self.v_feat.shape[1], self.feat_embed_dim),
                nn.LayerNorm(self.feat_embed_dim),
                nn.PReLU(),
                nn.Dropout(0.2),
                nn.Linear(self.feat_embed_dim, self.feat_embed_dim)
            )
            
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_projection = nn.Sequential(
                nn.Linear(self.t_feat.shape[1], self.feat_embed_dim),
                nn.LayerNorm(self.feat_embed_dim),
                nn.PReLU(),
                nn.Dropout(0.2),
                nn.Linear(self.feat_embed_dim, self.feat_embed_dim)
            )

        # Cross-modal fusion network
        self.modal_fusion = nn.ModuleList([
            CrossModalGATLayer(self.feat_embed_dim, self.feat_embed_dim, self.n_heads)
            for _ in range(self.n_layers)
        ])
        
        # Modal attention
        self.modal_attention = nn.Sequential(
            nn.Linear(self.feat_embed_dim * 2, self.feat_embed_dim),
            nn.LayerNorm(self.feat_embed_dim),
            nn.PReLU(),
            nn.Linear(self.feat_embed_dim, 2),
            nn.Softmax(dim=-1)
        )
        
        # User-Item interaction layers
        self.ui_layers = nn.ModuleList([
            GATv2Conv(self.embedding_dim, self.embedding_dim // self.n_heads, 
                     heads=self.n_heads, concat=True)
            for _ in range(self.n_layers)
        ])
        
        self.final_projection = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
            nn.PReLU()
        )
        
        # Initialize modal adjacency
        self._init_modal_adj()
        
    def _init_modal_adj(self):
        if self.v_feat is not None:
            image_feats = self.image_projection(self.image_embedding.weight)
            indices, image_adj = self.get_knn_adj_mat(image_feats)
            self.mm_adj = image_adj
            
        if self.t_feat is not None:
            text_feats = self.text_projection(self.text_embedding.weight)
            indices, text_adj = self.get_knn_adj_mat(text_feats)
            self.mm_adj = text_adj
            
        if self.v_feat is not None and self.t_feat is not None:
            self.mm_adj = self.mm_image_weight * image_adj + (1.0 - self.mm_image_weight) * text_adj
        
        self.mm_edge_index = self.mm_adj._indices()

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
        # add epsilon to avoid Devide by zero Warning
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

        return torch.sparse.FloatTensor(
            i, data, torch.Size((self.n_nodes, self.n_nodes))
        )
        
    def get_knn_adj_mat(self, embeddings):
        sim = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2)
        topk_values, topk_indices = torch.topk(sim, self.knn_k, dim=1)
        mask = torch.zeros_like(sim).scatter_(1, topk_indices, 1)
        adj = mask * sim
        
        indices = torch.nonzero(adj).t()
        values = adj[indices[0], indices[1]]
        return indices, self.compute_normalized_laplacian(indices, adj.size(), values)
        
    def compute_normalized_laplacian(self, indices, size, values):
        adj = torch.sparse.FloatTensor(indices, values, size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * values * cols_inv_sqrt
        return torch.sparse.FloatTensor(indices, values, size)

    def contrastive_loss(self, h1, h2):
        h1 = F.normalize(h1, p=2, dim=-1)
        h2 = F.normalize(h2, p=2, dim=-1)
        pos_score = (h1 * h2).sum(dim=-1)
        neg_score = h1 @ h2.t()
        neg_score = torch.logsumexp(neg_score / self.temperature, dim=-1)
        return (-pos_score / self.temperature + neg_score).mean()
        
    def forward(self, adj):
        # Process modal features
        if self.v_feat is not None:
            image_feats = self.image_projection(self.image_embedding.weight)
        if self.t_feat is not None:
            text_feats = self.text_projection(self.text_embedding.weight)
            
        # Cross-modal fusion
        if self.v_feat is not None and self.t_feat is not None:
            concat_feats = torch.cat([image_feats, text_feats], dim=-1)
            attention_weights = self.modal_attention(concat_feats)
            fused_feats = attention_weights[:, 0].unsqueeze(-1) * image_feats + \
                         attention_weights[:, 1].unsqueeze(-1) * text_feats
        else:
            fused_feats = image_feats if self.v_feat is not None else text_feats
            
        # Apply cross-modal GAT layers
        x = fused_feats
        for gat_layer in self.modal_fusion:
            x = gat_layer(x, self.mm_edge_index)
            
        # User-Item graph learning
        ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
        all_embeddings = [ego_embeddings]
        
        x_ui = ego_embeddings
        edge_index = adj._indices()
        for layer in self.ui_layers:
            x_ui = layer(x_ui, edge_index)
            all_embeddings.append(x_ui)
            
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        
        # Combine user-item embeddings with modal features
        enhanced_items = self.final_projection(torch.cat([i_g_embeddings, x], dim=-1))
        
        return u_g_embeddings, enhanced_items
        
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
        
        # Contrastive Loss
        if self.v_feat is not None and self.t_feat is not None:
            image_feats = self.image_projection(self.image_embedding.weight)[pos_items]
            text_feats = self.text_projection(self.text_embedding.weight)[pos_items]
            contra_loss = self.contrastive_loss(image_feats, text_feats)
        else:
            contra_loss = 0.0
        
        # Regularization
        reg_loss = self.reg_weight * (
            torch.norm(u_embeddings) +
            torch.norm(pos_embeddings) +
            torch.norm(neg_embeddings)
        )
        
        return mf_loss + reg_loss + self.alpha * contra_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        u_embeddings, i_embeddings = self.forward(self.norm_adj)
        u_embeddings = u_embeddings[user]
        scores = torch.matmul(u_embeddings, i_embeddings.transpose(0, 1))
        return scores