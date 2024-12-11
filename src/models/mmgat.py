# coding: utf-8
import os
import random
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss, L2Loss

class StableGATConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.gat = GATConv(in_channels, out_channels // 4, heads=4, dropout=0.1)
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(0.1)
        self.projection = nn.Linear(in_channels, out_channels)
        
    def forward(self, x, edge_index):
        identity = self.projection(x)
        out = self.gat(x, edge_index)
        out = self.dropout(out)
        out = self.norm(out + identity)
        return F.gelu(out)

class ModalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embed_dim, n_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.gat_layers = nn.ModuleList([
            StableGATConv(hidden_dim, hidden_dim)
            for _ in range(n_layers)
        ])
        self.output_proj = nn.Linear(hidden_dim, embed_dim)
        
    def forward(self, x, edge_index):
        x = self.norm(self.input_proj(x))
        residual = x
        
        for layer in self.gat_layers:
            x = layer(x, edge_index)
            x = 0.2 * x + 0.8 * residual
            residual = x
            
        return self.output_proj(x)

class MMGAT(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MMGAT, self).__init__(config, dataset)
        
        self.embedding_dim = config["embedding_size"]
        self.feat_embed_dim = config["feat_embed_dim"]
        self.hidden_dim = self.feat_embed_dim * 2
        self.n_layers = config["n_mm_layers"]
        self.n_ui_layers = 2
        self.reg_weight = config["reg_weight"]
        self.knn_k = config["knn_k"]
        self.dropout = config["dropout"]
        
        self.n_nodes = self.n_users + self.n_items
        
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_normal_(self.user_embedding.weight)
        nn.init.xavier_normal_(self.item_id_embedding.weight)

        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_encoder = ModalEncoder(self.v_feat.shape[1], self.hidden_dim, self.feat_embed_dim)
            
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_encoder = ModalEncoder(self.t_feat.shape[1], self.hidden_dim, self.feat_embed_dim)

        self.modal_fusion = nn.Sequential(
            nn.Linear(self.feat_embed_dim * 2, self.feat_embed_dim),
            nn.LayerNorm(self.feat_embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.feat_embed_dim, self.feat_embed_dim),
            nn.LayerNorm(self.feat_embed_dim)
        )
        
        self.graph_layers = nn.ModuleList([
            StableGATConv(self.embedding_dim, self.embedding_dim)
            for _ in range(self.n_ui_layers)
        ])
        
        self.to(self.device)
        self.reset_parameters()
        self.build_graph()

    def reset_parameters(self):
        def _reset(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.apply(_reset)

    def build_graph(self):
        # Build UI graph
        ui_mat = sp.dok_matrix((self.n_nodes, self.n_nodes), dtype=np.float32)
        ui_mat = ui_mat.tolil()
        R = self.interaction_matrix.tolil()
        ui_mat[:self.n_users, self.n_users:] = R
        ui_mat[self.n_users:, :self.n_users] = R.T
        ui_mat = ui_mat.todok()
        
        rowsum = np.array(ui_mat.sum(axis=1))
        d_inv = np.power(rowsum + 1e-6, -0.5).flatten()
        d_mat = sp.diags(d_inv)
        norm_adj = d_mat.dot(ui_mat).dot(d_mat)
        
        coo = norm_adj.tocoo()
        indices = np.vstack((coo.row, coo.col))
        self.edge_index = torch.LongTensor(indices).to(self.device)
        
        # Build modal graphs
        if self.v_feat is not None:
            self.image_edge_index = self.build_knn_graph(self.v_feat)
        if self.t_feat is not None:
            self.text_edge_index = self.build_knn_graph(self.t_feat)

    def build_knn_graph(self, features):
        sim = torch.mm(F.normalize(features, dim=1), F.normalize(features, dim=1).t())
        _, indices = sim.topk(k=self.knn_k, dim=1)
        rows = torch.arange(features.size(0), device=self.device).repeat_interleave(self.knn_k)
        cols = indices.reshape(-1)
        edge_index = torch.stack([rows, cols])
        return edge_index

    def forward(self):
        image_emb, text_emb = None, None
        
        if self.v_feat is not None:
            image_emb = self.image_encoder(self.image_embedding.weight, self.image_edge_index)
            
        if self.t_feat is not None:
            text_emb = self.text_encoder(self.text_embedding.weight, self.text_edge_index)
        
        if image_emb is not None and text_emb is not None:
            modal_emb = self.modal_fusion(torch.cat([image_emb, text_emb], dim=1))
        else:
            modal_emb = image_emb if image_emb is not None else text_emb

        ego_embeddings = torch.cat([self.user_embedding.weight, self.item_id_embedding.weight], dim=0)
        all_embeddings = [ego_embeddings]
        x = ego_embeddings

        for layer in self.graph_layers:
            x = layer(x, self.edge_index)
            all_embeddings.append(x)

        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        
        user_embeddings, item_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items])
        
        item_embeddings = item_embeddings + 0.2 * modal_emb
        
        return user_embeddings, item_embeddings, image_emb, text_emb

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        user_embeddings, item_embeddings, image_emb, text_emb = self.forward()

        u_embeddings = user_embeddings[users]
        pos_embeddings = item_embeddings[pos_items]
        neg_embeddings = item_embeddings[neg_items]

        # BPR Loss
        pos_scores = torch.sum(u_embeddings * pos_embeddings, dim=1)
        neg_scores = torch.sum(u_embeddings * neg_embeddings, dim=1)
        rec_loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-9))

        # Modal Contrastive Loss
        modal_loss = 0.0
        if image_emb is not None and text_emb is not None:
            i_emb = F.normalize(image_emb[pos_items], dim=1)
            t_emb = F.normalize(text_emb[pos_items], dim=1)
            modal_loss = 1 - torch.mean(F.cosine_similarity(i_emb, t_emb, dim=1))

        # L2 Regularization
        l2_loss = self.reg_weight * (
            torch.norm(u_embeddings) +
            torch.norm(pos_embeddings) +
            torch.norm(neg_embeddings)
        )

        loss = rec_loss + 0.1 * modal_loss + l2_loss
        return torch.clamp(loss, max=1e4)

    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_embeddings, item_embeddings, _, _ = self.forward()

        scores = torch.matmul(user_embeddings[user], item_embeddings.transpose(0, 1))
        return scores