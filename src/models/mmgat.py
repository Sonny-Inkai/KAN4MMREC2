# coding: utf-8
import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from common.abstract_recommender import GeneralRecommender

class MMGAT(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MMGAT, self).__init__(config, dataset)
        
        self.embedding_dim = config["embedding_size"]
        self.feat_embed_dim = config["feat_embed_dim"]
        self.n_layers = config["n_mm_layers"]
        self.reg_weight = config["reg_weight"]
        self.dropout = config["dropout"]
        self.knn_k = config["knn_k"]
        
        # Initialize embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)
        
        # Initialize modal projections
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_proj = nn.Sequential(
                nn.Linear(self.v_feat.shape[1], self.feat_embed_dim),
                nn.GELU()
            )
        
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_proj = nn.Sequential(
                nn.Linear(self.t_feat.shape[1], self.feat_embed_dim),
                nn.GELU()
            )
        
        # Light fusion network
        self.modal_gating = nn.Sequential(
            nn.Linear(self.feat_embed_dim * 2, 2),
            nn.Softmax(dim=-1)
        )
        
        # Load interaction data and build graph structure
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.edge_index = self.build_edge_index().to(self.device)
        self.edge_norm = self.compute_edge_norm(self.edge_index).to(self.device)
        
        # Build modal graph structure
        self.mm_edge_index = None
        self.mm_edge_norm = None
        self.build_modal_graph()
        
        self.to(self.device)
    
    def build_edge_index(self):
        rows = self.interaction_matrix.row
        cols = self.interaction_matrix.col + self.n_users
        
        edge_index = torch.tensor(np.vstack([
            np.concatenate([rows, cols, cols, rows]),
            np.concatenate([cols, rows, rows, cols])
        ]), dtype=torch.long)
        
        return edge_index

    def compute_edge_norm(self, edge_index):
        row, col = edge_index[0], edge_index[1]
        deg = torch.bincount(row, minlength=self.n_users + self.n_items).float()
        deg_inv_sqrt = torch.pow(deg + 1e-12, -0.5)
        
        return deg_inv_sqrt[row] * deg_inv_sqrt[col]

    def build_modal_graph(self):
        if self.v_feat is None and self.t_feat is None:
            return
            
        features = []
        if self.v_feat is not None:
            v_feat = self.image_proj(self.image_embedding.weight.detach())
            features.append(v_feat)
        
        if self.t_feat is not None:
            t_feat = self.text_proj(self.text_embedding.weight.detach())
            features.append(t_feat)
            
        if len(features) == 2:
            features = torch.stack(features).mean(dim=0)
        else:
            features = features[0]
            
        features = F.normalize(features, p=2, dim=1)
        sim = torch.mm(features, features.t())
        values, indices = sim.topk(k=self.knn_k, dim=1)
        
        rows = torch.arange(features.size(0), device=self.device).view(-1, 1).expand(-1, self.knn_k)
        self.mm_edge_index = torch.stack([rows.reshape(-1), indices.reshape(-1)]).long()
        self.mm_edge_norm = self.compute_edge_norm(self.mm_edge_index)

    def light_graph_conv(self, x, edge_index, edge_norm):
        # Light graph convolution with edge dropout
        if self.training:
            mask = torch.rand(edge_norm.size(0), device=self.device) > self.dropout
            edge_index = edge_index[:, mask]
            edge_norm = edge_norm[mask]
            
        row, col = edge_index[0], edge_index[1]
        out = torch.zeros_like(x)
        out.index_add_(0, row, x[col] * edge_norm.view(-1, 1))
        
        return out

    def forward(self):
        # Process modalities
        modal_feat = None
        if self.v_feat is not None and self.t_feat is not None:
            img_feat = self.image_proj(self.image_embedding.weight)
            txt_feat = self.text_proj(self.text_embedding.weight)
            
            # Light gating mechanism
            gate = self.modal_gating(torch.cat([img_feat, txt_feat], dim=-1))
            modal_feat = gate[:, 0:1] * img_feat + gate[:, 1:2] * txt_feat
        elif self.v_feat is not None:
            modal_feat = self.image_proj(self.image_embedding.weight)
        elif self.t_feat is not None:
            modal_feat = self.text_proj(self.text_embedding.weight)
        
        # Graph convolution for items
        if modal_feat is not None and self.mm_edge_index is not None:
            modal_feat = self.light_graph_conv(modal_feat, self.mm_edge_index, self.mm_edge_norm)
            
        # User-item graph convolution
        all_emb = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        embs = [all_emb]
        
        for _ in range(self.n_layers):
            all_emb = self.light_graph_conv(all_emb, self.edge_index, self.edge_norm)
            embs.append(all_emb)
            
        all_emb = torch.stack(embs, dim=1).mean(dim=1)
        
        users_emb, items_emb = torch.split(all_emb, [self.n_users, self.n_items])
        items_emb = items_emb + modal_feat if modal_feat is not None else items_emb
        
        return users_emb, items_emb

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        
        users_emb, items_emb = self.forward()
        
        # Extract embeddings
        u_emb = users_emb[users]
        pos_emb = items_emb[pos_items]
        neg_emb = items_emb[neg_items]
        
        # Compute scores
        pos_scores = torch.sum(u_emb * pos_emb, dim=1)
        neg_scores = torch.sum(u_emb * neg_emb, dim=1)
        
        # BPR loss
        mf_loss = -torch.mean(torch.log(1e-10 + torch.sigmoid(pos_scores - neg_scores)))
        
        # Regularization
        reg_loss = self.reg_weight * (
            torch.norm(u_emb) +
            torch.norm(pos_emb) +
            torch.norm(neg_emb)
        )
        
        return mf_loss + reg_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        users_emb, items_emb = self.forward()
        scores = torch.matmul(users_emb[user], items_emb.transpose(0, 1))
        return scores