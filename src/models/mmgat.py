# coding: utf-8
import os
import random
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss, L2Loss

class LightGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(LightGATLayer, self).__init__()
        self.gat = GATv2Conv(in_dim, out_dim, heads=1, concat=False, dropout=0.1)
        self.norm = nn.LayerNorm(out_dim)
        
    def forward(self, x, edge_index):
        x = self.gat(x, edge_index)
        x = self.norm(x)
        return x

class MMGAT(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MMGAT, self).__init__(config, dataset)
        
        self.embedding_dim = config["embedding_size"]
        self.feat_embed_dim = config["feat_embed_dim"]
        self.n_layers = config["n_mm_layers"]
        self.dropout = config["dropout"]
        self.knn_k = config["knn_k"]
        self.reg_weight = config["reg_weight"]
        self.mm_image_weight = config["mm_image_weight"]
        self.batch_size = 512  # Batch size for similarity computation
        self.n_nodes = self.n_users + self.n_items
        
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.norm_adj = self.get_norm_adj_mat().to(self.device)
        
        # Embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_id_embedding.weight, std=0.1)
        
        # Modal feature processing
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)
            
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)

        # GAT layers
        self.gat_layers = nn.ModuleList([
            LightGATLayer(self.feat_embed_dim, self.feat_embed_dim)
            for _ in range(self.n_layers)
        ])
        
        # Projection layer
        self.projection = nn.Sequential(
            nn.Linear(self.feat_embed_dim * 2, self.feat_embed_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
        # Move to device
        self.to(self.device)
        self._init_modal_adj()

    def compute_similarity_batched(self, embeddings):
        n_items = embeddings.size(0)
        indices_list = []
        values_list = []
        
        for i in range(0, n_items, self.batch_size):
            end_idx = min(i + self.batch_size, n_items)
            batch_embeddings = embeddings[i:end_idx]
            
            # Compute similarity for current batch
            sim = F.cosine_similarity(
                batch_embeddings.unsqueeze(1),
                embeddings.unsqueeze(0),
                dim=2
            )
            
            # Get top-k for current batch
            topk_values, topk_indices = torch.topk(sim, self.knn_k, dim=1)
            
            # Create indices for current batch
            batch_size = end_idx - i
            row_indices = torch.arange(i, end_idx).view(-1, 1).expand(-1, self.knn_k)
            
            # Add to lists
            indices_list.append(torch.stack([row_indices.flatten(), topk_indices.flatten()]))
            values_list.append(topk_values.flatten())
            
            # Clear memory
            del sim, topk_values, topk_indices
            torch.cuda.empty_cache()
        
        # Combine all batches
        indices = torch.cat(indices_list, dim=1)
        values = torch.cat(values_list)
        
        return indices, values

    def _init_modal_adj(self):
        if self.v_feat is not None:
            image_feats = self.image_trs(self.image_embedding.weight)
            v_indices, v_values = self.compute_similarity_batched(image_feats)
            self.mm_adj = self.get_sparse_adj(v_indices, v_values, image_feats.size(0))
            
        if self.t_feat is not None:
            text_feats = self.text_trs(self.text_embedding.weight)
            t_indices, t_values = self.compute_similarity_batched(text_feats)
            text_adj = self.get_sparse_adj(t_indices, t_values, text_feats.size(0))
            
            if self.v_feat is not None:
                self.mm_adj = self.mm_image_weight * self.mm_adj + (1 - self.mm_image_weight) * text_adj
            else:
                self.mm_adj = text_adj
                
        self.mm_edge_index = self.mm_adj._indices()
        
    def get_sparse_adj(self, indices, values, size):
        adj = torch.sparse_coo_tensor(indices, values, (size, size))
        adj = self.normalize_adj(adj)
        return adj.to(self.device)
        
    def normalize_adj(self, adj):
        row_sum = torch.sparse.sum(adj, dim=1).to_dense()
        d_inv_sqrt = torch.pow(row_sum + 1e-7, -0.5)
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        return torch.sparse.mm(torch.sparse.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)

    def get_norm_adj_mat(self):
        A = sp.dok_matrix((self.n_nodes, self.n_nodes), dtype=np.float32)
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
        return torch.sparse_coo_tensor(
            torch.LongTensor([L.row, L.col]),
            torch.FloatTensor(L.data),
            torch.Size([self.n_nodes, self.n_nodes])
        )
        
    def forward(self, adj):
        # Process modal features
        if self.v_feat is not None:
            image_feats = self.image_trs(self.image_embedding.weight)
        if self.t_feat is not None:
            text_feats = self.text_trs(self.text_embedding.weight)
            
        # Modal fusion
        if self.v_feat is not None and self.t_feat is not None:
            modal_feats = torch.cat([image_feats, text_feats], dim=1)
            fused_feats = self.projection(modal_feats)
        else:
            fused_feats = image_feats if self.v_feat is not None else text_feats
        
        # GAT layers
        x = fused_feats
        for layer in self.gat_layers:
            x = layer(x, self.mm_edge_index)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
        # User-Item interaction
        ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
        all_embeddings = [ego_embeddings]
        
        # Light graph convolution
        for _ in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(adj, ego_embeddings)
            all_embeddings.append(ego_embeddings)
            
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        i_g_embeddings = i_g_embeddings + x
        
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
        pos_scores = torch.sum(u_embeddings * pos_embeddings, dim=1)
        neg_scores = torch.sum(u_embeddings * neg_embeddings, dim=1)
        mf_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        
        # Regularization
        reg_loss = self.reg_weight * (
            torch.norm(u_embeddings) +
            torch.norm(pos_embeddings) +
            torch.norm(neg_embeddings)
        )
        
        return mf_loss + reg_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        u_embeddings, i_embeddings = self.forward(self.norm_adj)
        u_embeddings = u_embeddings[user]
        scores = torch.matmul(u_embeddings, i_embeddings.transpose(0, 1))
        return scores