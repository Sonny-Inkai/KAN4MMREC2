# coding: utf-8
import os
import random
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss

class ModalTower(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.transform = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
    def forward(self, x):
        return self.transform(x)

class MMGAT(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MMGAT, self).__init__(config, dataset)
        
        self.embedding_dim = config["embedding_size"]
        self.feat_embed_dim = config["feat_embed_dim"]
        self.hidden_dim = self.feat_embed_dim * 2
        self.n_layers = config["n_mm_layers"]
        self.knn_k = config["knn_k"]
        self.reg_weight = config["reg_weight"]
        self.dropout = config["dropout"]
        self.temperature = 1.0  # Increased temperature for more stable contrast
        
        # User Tower
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.user_tower = ModalTower(self.embedding_dim, self.hidden_dim, self.embedding_dim)
        
        # Modal Towers
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_tower = ModalTower(self.v_feat.shape[1], self.hidden_dim, self.embedding_dim)
            
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_tower = ModalTower(self.t_feat.shape[1], self.hidden_dim, self.embedding_dim)
            
        # Modal Fusion with residual
        self.fusion = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
            nn.GELU(),
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim)
        )
        
        # Initialize Graph Structure
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.build_graph_structure()
        self.to(self.device)
        
    def build_graph_structure(self):
        # User-Item Graph
        A = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        A._update(data_dict)
        
        # Symmetric normalization
        rowsum = np.array(A.sum(axis=1))
        d_inv = np.power(rowsum + 1e-7, -0.5).flatten()
        d_mat = sp.diags(d_inv)
        norm_adj = d_mat.dot(A).dot(d_mat)
        
        norm_adj = norm_adj.tocoo()
        indices = np.vstack((norm_adj.row, norm_adj.col))
        indices = torch.LongTensor(indices).to(self.device)
        values = torch.FloatTensor(norm_adj.data).to(self.device)
        self.norm_adj = torch.sparse_coo_tensor(indices, values, norm_adj.shape)
        
        # Modal similarity graphs with memory-efficient computation
        if self.v_feat is not None:
            self.image_adj = self.build_knn_graph(self.v_feat)
        if self.t_feat is not None:
            self.text_adj = self.build_knn_graph(self.t_feat)

    def build_knn_graph(self, features):
        # Compute KNN graph in batches to save memory
        batch_size = 1024
        n_batches = (features.size(0) + batch_size - 1) // batch_size
        adj_indices = []
        adj_values = []
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, features.size(0))
            batch_features = features[start_idx:end_idx]
            
            # Compute similarities
            norm_features = F.normalize(features, p=2, dim=1)
            batch_norm = F.normalize(batch_features, p=2, dim=1)
            sim = torch.mm(batch_norm, norm_features.t())
            
            # Get top-k
            topk_values, topk_indices = sim.topk(k=self.knn_k, dim=1)
            rows = torch.arange(start_idx, end_idx, device=self.device).repeat_interleave(self.knn_k)
            cols = topk_indices.reshape(-1)
            values = topk_values.reshape(-1)
            
            adj_indices.append(torch.stack([rows, cols]))
            adj_values.append(values)
            
        adj_indices = torch.cat(adj_indices, dim=1)
        adj_values = torch.cat(adj_values)
        
        # Normalize adjacency matrix
        adj = torch.sparse_coo_tensor(
            adj_indices, adj_values,
            (features.size(0), features.size(0))
        ).coalesce()
        
        # Symmetric normalization
        row_sum = torch.sparse.sum(adj, dim=1).to_dense()
        d_inv_sqrt = torch.pow(row_sum + 1e-7, -0.5)
        
        values = adj_values * d_inv_sqrt[adj_indices[0]] * d_inv_sqrt[adj_indices[1]]
        return torch.sparse_coo_tensor(adj_indices, values, adj.size()).to(self.device)

    def forward(self):
        # User Tower with graph convolution
        user_emb = self.user_tower(self.user_embedding.weight)
        
        # Modal Towers
        image_emb = self.image_tower(self.image_embedding.weight) if self.v_feat is not None else None
        text_emb = self.text_tower(self.text_embedding.weight) if self.t_feat is not None else None
        
        # Modal feature enhancement
        if image_emb is not None:
            image_emb = torch.sparse.mm(self.image_adj, image_emb)
        if text_emb is not None:
            text_emb = torch.sparse.mm(self.text_adj, text_emb)
        
        # Adaptive modal fusion with residual
        if image_emb is not None and text_emb is not None:
            modal_cat = torch.cat([image_emb, text_emb], dim=1)
            fused_emb = self.fusion(modal_cat)
            item_emb = fused_emb + 0.5 * (image_emb + text_emb)  # Residual connection
        else:
            item_emb = image_emb if image_emb is not None else text_emb
        
        # Graph convolution
        all_emb = torch.cat([user_emb, item_emb], dim=0)
        embs = [all_emb]
        
        # Multiple layers of graph convolution with residual connections
        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(self.norm_adj, all_emb)
            embs.append(all_emb)
        
        # Residual fusion of all layers
        all_emb = torch.stack(embs, dim=1)
        all_emb = torch.mean(all_emb, dim=1)
        
        user_emb, item_emb = torch.split(all_emb, [self.n_users, self.n_items])
        return user_emb, item_emb, image_emb, text_emb

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        
        user_emb, item_emb, img_emb, txt_emb = self.forward()
        
        user_e = user_emb[users]
        pos_e = item_emb[pos_items]
        neg_e = item_emb[neg_items]
        
        # BPR loss with margin
        pos_scores = torch.sum(user_e * pos_e, dim=1)
        neg_scores = torch.sum(user_e * neg_e, dim=1)
        margin = 1.0
        bpr_loss = torch.mean(F.softplus(neg_scores - pos_scores + margin))
        
        # InfoNCE loss for modalities
        modal_loss = 0.0
        if img_emb is not None and txt_emb is not None:
            pos_sim = torch.sum(F.normalize(img_emb[pos_items], dim=1) * 
                              F.normalize(txt_emb[pos_items], dim=1), dim=1)
            # Negative samples from batch
            neg_sim = torch.matmul(F.normalize(img_emb[pos_items], dim=1),
                                 F.normalize(txt_emb[pos_items], dim=1).t())
            modal_loss = -torch.mean(pos_sim / self.temperature - 
                                   torch.logsumexp(neg_sim / self.temperature, dim=1))
        
        # L2 regularization
        l2_loss = self.reg_weight * (
            torch.norm(user_e) +
            torch.norm(pos_e) +
            torch.norm(neg_e)
        )
        
        return bpr_loss + 0.1 * modal_loss + l2_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_emb, item_emb, _, _ = self.forward()
        score = torch.matmul(user_emb[user], item_emb.transpose(0, 1))
        return score