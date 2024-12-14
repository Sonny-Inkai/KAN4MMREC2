# coding: utf-8
import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from common.abstract_recommender import GeneralRecommender

class AdaptiveEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.main_path = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        self.residual_path = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        self.gating = nn.Sequential(
            nn.Linear(output_dim * 2, 2),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x):
        main = self.main_path(x)
        res = self.residual_path(x)
        gate = self.gating(torch.cat([main, res], dim=-1))
        output = gate[:, 0:1] * main + gate[:, 1:2] * res
        return output

class DynamicAggregator(nn.Module):
    def __init__(self, dim, momentum=0.99):
        super().__init__()
        self.momentum = momentum
        self.register_buffer('running_mean', torch.zeros(dim))
        self.register_buffer('running_var', torch.ones(dim))
        
    def forward(self, x, adj):
        if self.training:
            mean = x.mean(0)
            var = x.var(0)
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            mean = self.running_mean
            var = self.running_var
            
        x = (x - mean) / (var + 1e-5).sqrt()
        return torch.mm(adj, x)

class MMGAT(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MMGAT, self).__init__(config, dataset)
        
        self.embedding_dim = config["embedding_size"]
        self.feat_embed_dim = config["feat_embed_dim"]
        self.hidden_dim = self.feat_embed_dim * 2
        self.n_layers = config["n_mm_layers"]
        self.reg_weight = config["reg_weight"]
        self.dropout = config["dropout"]
        self.knn_k = config["knn_k"]
        
        # User/Item embeddings with auxiliary tasks
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        self.user_bias = nn.Embedding(self.n_users, 1)
        self.item_bias = nn.Embedding(self.n_items, 1)
        
        # Initialize with small values
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)
        
        # Modal encoders with adaptive paths
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_encoder = AdaptiveEncoder(
                self.v_feat.shape[1], self.hidden_dim, self.feat_embed_dim
            )
            
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_encoder = AdaptiveEncoder(
                self.t_feat.shape[1], self.hidden_dim, self.feat_embed_dim
            )
        
        # Dynamic graph layers
        self.graph_layers = nn.ModuleList([
            DynamicAggregator(self.embedding_dim)
            for _ in range(self.n_layers)
        ])
        
        # Modal fusion with attention
        if self.v_feat is not None and self.t_feat is not None:
            self.modal_attention = nn.MultiheadAttention(
                self.feat_embed_dim, 4, dropout=self.dropout, batch_first=True
            )
            self.modal_fusion = nn.Sequential(
                nn.Linear(self.feat_embed_dim * 2, self.feat_embed_dim),
                nn.LayerNorm(self.feat_embed_dim),
                nn.GELU(),
                nn.Dropout(self.dropout)
            )
        
        # Load data
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.norm_adj = None
        self.modal_adj = None
        self.build_graphs()
        
        self.to(self.device)
        
    def build_graphs(self):
        # Build user-item graph
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.interaction_matrix.tolil()
        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()
        
        # Normalize adjacency matrix
        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum + 1e-9, -0.5).flatten()
        d_mat = sp.diags(d_inv)
        norm_adj = d_mat.dot(adj_mat).dot(d_mat)
        
        norm_adj = norm_adj.tocoo()
        indices = np.vstack((norm_adj.row, norm_adj.col))
        i = torch.LongTensor(indices).to(self.device)
        v = torch.FloatTensor(norm_adj.data).to(self.device)
        self.norm_adj = torch.sparse_coo_tensor(i, v, norm_adj.shape)
        
        # Build modal graph
        if self.v_feat is not None:
            v_feat = F.normalize(self.v_feat, p=2, dim=1)
            self.modal_adj = self.build_modal_graph(v_feat)
        
        if self.t_feat is not None:
            t_feat = F.normalize(self.t_feat, p=2, dim=1)
            if self.modal_adj is None:
                self.modal_adj = self.build_modal_graph(t_feat)
            else:
                t_adj = self.build_modal_graph(t_feat)
                self.modal_adj = 0.5 * (self.modal_adj + t_adj)
                
    def build_modal_graph(self, features):
        sim = torch.mm(features, features.t())
        values, indices = sim.topk(k=self.knn_k, dim=1)
        rows = torch.arange(features.size(0), device=self.device)
        rows = rows.view(-1, 1).expand_as(indices)
        
        adj = torch.zeros_like(sim)
        adj[rows.reshape(-1), indices.reshape(-1)] = values.reshape(-1)
        adj = (adj + adj.t()) / 2
        
        # Normalize
        deg = torch.sum(adj, dim=1)
        deg_inv_sqrt = torch.pow(deg + 1e-12, -0.5)
        adj = deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)
        return adj.to(self.device)

    def forward(self):
        # Modal processing
        img_feat = txt_feat = None
        
        if self.v_feat is not None:
            img_feat = self.image_encoder(self.image_embedding.weight)
            
        if self.t_feat is not None:
            txt_feat = self.text_encoder(self.text_embedding.weight)
        
        # Modal fusion
        if img_feat is not None and txt_feat is not None:
            # Cross attention
            attn_out, _ = self.modal_attention(
                img_feat.unsqueeze(0), txt_feat.unsqueeze(0), txt_feat.unsqueeze(0)
            )
            attn_feat = attn_out.squeeze(0)
            
            # Concatenative fusion
            modal_feat = self.modal_fusion(torch.cat([img_feat, attn_feat], dim=1))
        else:
            modal_feat = img_feat if img_feat is not None else txt_feat
            
        # Graph propagation
        x = torch.cat([self.user_embedding.weight, self.item_embedding.weight])
        all_embs = [x]
        
        # Dynamic graph convolution
        for layer in self.graph_layers:
            x = layer(x, self.norm_adj)
            if modal_feat is not None:
                item_part = x[self.n_users:]
                item_part = item_part + layer(modal_feat, self.modal_adj)
                x = torch.cat([x[:self.n_users], item_part])
            all_embs.append(x)
            
        x = torch.stack(all_embs).mean(dim=0)
        user_emb, item_emb = torch.split(x, [self.n_users, self.n_items])
        
        return user_emb, item_emb, img_feat, txt_feat

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        
        user_emb, item_emb, img_feat, txt_feat = self.forward()
        
        u_e = user_emb[users]
        pos_e = item_emb[pos_items]
        neg_e = item_emb[neg_items]
        
        # Main recommendation loss
        u_b = self.user_bias(users).squeeze()
        pos_b = self.item_bias(pos_items).squeeze()
        neg_b = self.item_bias(neg_items).squeeze()
        
        pos_scores = torch.sum(u_e * pos_e, dim=1) + u_b + pos_b
        neg_scores = torch.sum(u_e * neg_e, dim=1) + u_b + neg_b
        rec_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        
        # Multi-task learning with modal alignment
        modal_loss = 0.0
        if img_feat is not None and txt_feat is not None:
            i_feat = F.normalize(img_feat[pos_items], dim=1)
            t_feat = F.normalize(txt_feat[pos_items], dim=1)
            modal_loss = 1 - F.cosine_similarity(i_feat, t_feat).mean()
        
        # L2 regularization with decay
        reg_loss = self.reg_weight * (
            torch.norm(u_e) +
            torch.norm(pos_e) +
            torch.norm(neg_e)
        )
        
        return rec_loss + 0.1 * modal_loss + reg_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_emb, item_emb, _, _ = self.forward()
        
        scores = torch.matmul(user_emb[user], item_emb.transpose(0, 1))
        #scores = scores + self.user_bias(user).squeeze() + self.item_bias.weight.squeeze()
        return scores