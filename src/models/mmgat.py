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
        self.predictor = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim)
        )
        
    def forward(self, x):
        h = self.transform(x)
        return h, self.predictor(h.detach())

class LightGraphConv(nn.Module):
    def __init__(self, edge_dropout=0.1):
        super().__init__()
        self.edge_dropout = edge_dropout
        
    def forward(self, x, adj):
        if self.training:
            mask = torch.rand(adj._values().size()) > self.edge_dropout
            adj = torch.sparse_coo_tensor(
                adj._indices()[:, mask],
                adj._values()[mask],
                adj.size()
            ).to(x.device)
        return torch.sparse.mm(adj, x)

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
        self.temperature = 0.2
        
        # User Tower
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.user_tower = nn.Sequential(
            nn.Linear(self.embedding_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.embedding_dim)
        )
        
        # Modal Towers
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_tower = ModalTower(self.v_feat.shape[1], self.hidden_dim, self.embedding_dim)
            self.image_graph = LightGraphConv(self.dropout)
            
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_tower = ModalTower(self.t_feat.shape[1], self.hidden_dim, self.embedding_dim)
            self.text_graph = LightGraphConv(self.dropout)
            
        # Fusion Layer
        self.fusion = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
            nn.GELU(),
            nn.Dropout(self.dropout)
        )
        
        # Graph Layers
        self.user_graph = LightGraphConv(self.dropout)
        
        # Initialize Graph Structure
        self.build_graph_structure()
        self.to(self.device)
        
    def build_graph_structure(self):
        # Build user-item interaction graph
        self.interaction_matrix = self.dataset.inter_matrix(form='coo').astype(np.float32)
        ui_adj = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        ui_adj = ui_adj.tolil()
        R = self.interaction_matrix.tolil()
        ui_adj[:self.n_users, self.n_users:] = R
        ui_adj[self.n_users:, :self.n_users] = R.T
        ui_adj = ui_adj.todok()
        
        # Normalize adjacency matrix
        rowsum = np.array(ui_adj.sum(axis=1))
        d_inv = np.power(rowsum + 1e-7, -0.5).flatten()
        d_mat = sp.diags(d_inv)
        norm_adj = d_mat.dot(ui_adj).dot(d_mat)
        
        # Convert to sparse tensor
        coo = norm_adj.tocoo()
        indices = torch.LongTensor([coo.row, coo.col])
        values = torch.FloatTensor(coo.data)
        self.norm_adj = torch.sparse_coo_tensor(indices, values, coo.shape).to(self.device)
        
        # Build modal graphs
        if self.v_feat is not None:
            self.image_adj = self.build_knn_graph(self.v_feat)
        if self.t_feat is not None:
            self.text_adj = self.build_knn_graph(self.t_feat)
            
    def build_knn_graph(self, features):
        sim = torch.mm(F.normalize(features, p=2, dim=1), F.normalize(features, p=2, dim=1).t())
        topk_values, topk_indices = torch.topk(sim, k=self.knn_k, dim=1)
        rows = torch.arange(features.size(0)).repeat_interleave(self.knn_k)
        adj = torch.sparse_coo_tensor(
            torch.stack([rows, topk_indices.reshape(-1)]),
            topk_values.reshape(-1),
            (features.size(0), features.size(0))
        ).to(self.device)
        return adj
        
    def forward(self):
        # Process user embeddings
        user_emb = self.user_tower(self.user_embedding.weight)
        user_enhanced = user_emb
        
        # Process modalities
        image_emb, image_pred, text_emb, text_pred = None, None, None, None
        
        if self.v_feat is not None:
            image_emb, image_pred = self.image_tower(self.image_embedding.weight)
            for _ in range(self.n_layers):
                image_emb = self.image_graph(image_emb, self.image_adj)
                
        if self.t_feat is not None:
            text_emb, text_pred = self.text_tower(self.text_embedding.weight)
            for _ in range(self.n_layers):
                text_emb = self.text_graph(text_emb, self.text_adj)
        
        # Fuse modalities
        if image_emb is not None and text_emb is not None:
            modal_emb = self.fusion(torch.cat([image_emb, text_emb], dim=1))
        else:
            modal_emb = image_emb if image_emb is not None else text_emb
            
        # Graph convolution
        for _ in range(self.n_layers):
            user_enhanced = self.user_graph(
                torch.cat([user_enhanced, modal_emb], dim=0),
                self.norm_adj
            )[:self.n_users]
            
        return (
            user_enhanced, modal_emb,
            image_emb, image_pred,
            text_emb, text_pred
        )
        
    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        
        user_emb, item_emb, img_emb, img_pred, txt_emb, txt_pred = self.forward()
        
        user_e = user_emb[users]
        pos_e = item_emb[pos_items]
        neg_e = item_emb[neg_items]
        
        # BPR loss
        pos_scores = torch.sum(user_e * pos_e, dim=1)
        neg_scores = torch.sum(user_e * neg_e, dim=1)
        bpr_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        
        # Contrastive loss
        modal_loss = 0.0
        if img_emb is not None and txt_emb is not None:
            i_emb = F.normalize(img_emb[pos_items], dim=1)
            t_emb = F.normalize(txt_emb[pos_items], dim=1)
            i_pred = F.normalize(img_pred[pos_items], dim=1)
            t_pred = F.normalize(txt_pred[pos_items], dim=1)
            
            modal_loss = -(
                torch.mean(F.cosine_similarity(i_emb, t_pred.detach())) +
                torch.mean(F.cosine_similarity(t_emb, i_pred.detach()))
            ) / (2 * self.temperature)
        
        # L2 regularization
        reg_loss = self.reg_weight * (
            torch.norm(user_e) +
            torch.norm(pos_e) +
            torch.norm(neg_e)
        )
        
        return bpr_loss + 0.2 * modal_loss + reg_loss
        
    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_emb, item_emb, _, _, _, _ = self.forward()
        scores = torch.matmul(user_emb[user], item_emb.transpose(0, 1))
        return scores