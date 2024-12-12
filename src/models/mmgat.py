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
            nn.Linear(self.hidden_dim, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim)
        )
        
        # Modal Towers
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_tower = ModalTower(self.v_feat.shape[1], self.hidden_dim, self.embedding_dim)
            
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_tower = ModalTower(self.t_feat.shape[1], self.hidden_dim, self.embedding_dim)
            
        # Fusion Layer with gating
        self.fusion_gate = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, 2),
            nn.Softmax(dim=1)
        )
            
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.to(self.device)
        self.build_graph_structure()

    def build_graph_structure(self):
        # Build user-item interaction graph
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
        indices = np.vstack((coo.row, coo.col))
        i = torch.LongTensor(indices).to(self.device)
        v = torch.FloatTensor(coo.data).to(self.device)
        self.norm_adj = torch.sparse_coo_tensor(i, v, coo.shape)
        
        # Build modal graphs
        if self.v_feat is not None:
            self.image_adj = self.build_knn_graph(self.v_feat)
        if self.t_feat is not None:
            self.text_adj = self.build_knn_graph(self.t_feat)

    def build_knn_graph(self, features):
        features = F.normalize(features, dim=1)
        sim = torch.mm(features, features.t())
        topk_values, topk_indices = torch.topk(sim, k=self.knn_k, dim=1)
        rows = torch.arange(features.size(0), device=self.device).repeat_interleave(self.knn_k)
        cols = topk_indices.reshape(-1)
        i = torch.stack([rows, cols])
        v = torch.ones_like(cols, dtype=torch.float32)
        adj = torch.sparse_coo_tensor(i, v, (features.size(0), features.size(0)))
        return adj

    def graph_conv(self, x, adj):
        return torch.sparse.mm(adj, x)

    def forward(self):
        # User Tower
        user_emb = self.user_tower(self.user_embedding.weight)
        
        # Image Tower
        image_emb, image_pred = None, None
        if self.v_feat is not None:
            image_emb, image_pred = self.image_tower(self.image_embedding.weight)
            image_emb = self.graph_conv(image_emb, self.image_adj)
            
        # Text Tower
        text_emb, text_pred = None, None
        if self.t_feat is not None:
            text_emb, text_pred = self.text_tower(self.text_embedding.weight)
            text_emb = self.graph_conv(text_emb, self.text_adj)
        
        # Fuse modalities with gating
        if image_emb is not None and text_emb is not None:
            concat_feat = torch.cat([image_emb, text_emb], dim=1)
            gate_weights = self.fusion_gate(concat_feat)
            modal_emb = gate_weights[:, 0].unsqueeze(1) * image_emb + \
                       gate_weights[:, 1].unsqueeze(1) * text_emb
        else:
            modal_emb = image_emb if image_emb is not None else text_emb

        # Graph convolution with user-item interactions
        all_emb = torch.cat([user_emb, modal_emb], dim=0)
        embs = [all_emb]
        
        for _ in range(self.n_layers):
            all_emb = self.graph_conv(all_emb, self.norm_adj)
            embs.append(all_emb)
            
        all_emb = torch.stack(embs, dim=1).mean(dim=1)
        user_emb, item_emb = torch.split(all_emb, [self.n_users, self.n_items])
        
        return user_emb, item_emb, image_emb, image_pred, text_emb, text_pred

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        
        user_emb, item_emb, img_emb, img_pred, txt_emb, txt_pred = self.forward()
        
        # Get embeddings
        u_e = user_emb[users]
        pos_e = item_emb[pos_items]
        neg_e = item_emb[neg_items]
        
        # BPR loss
        pos_scores = torch.sum(u_e * pos_e, dim=1)
        neg_scores = torch.sum(u_e * neg_e, dim=1)
        bpr_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        
        # Modal contrastive loss
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
            torch.norm(u_e) +
            torch.norm(pos_e) +
            torch.norm(neg_e)
        )
        
        return bpr_loss + 0.2 * modal_loss + reg_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_emb, item_emb, _, _, _, _ = self.forward()
        scores = torch.matmul(user_emb[user], item_emb.transpose(0, 1))
        return scores