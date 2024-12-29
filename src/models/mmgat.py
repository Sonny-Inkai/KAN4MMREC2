# coding: utf-8
import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from common.abstract_recommender import GeneralRecommender
from utils.utils import build_sim, compute_normalized_laplacian
from torch_scatter import scatter_mean, scatter_softmax
from torch_geometric.utils import add_self_loops
# 0.0260
class ModalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
    def forward(self, x):
        return F.normalize(self.encoder(x), p=2, dim=1)

class GraphAttentionLayer(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.gat = GATConv(
            dim, 
            dim // 4,
            heads=4, 
            dropout=dropout,
            add_self_loops=True,
            edge_dim=1
        )
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()
        
    def forward(self, x, edge_index, edge_attr=None):
        out = self.gat(x, edge_index, edge_attr)
        out = self.act(out)
        out = self.dropout(out)
        return self.norm(x + out)

class MMGAT(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MMGAT, self).__init__(config, dataset)
        
        self.embedding_dim = config["embedding_size"]
        self.feat_embed_dim = config["feat_embed_dim"]
        self.hidden_dim = self.feat_embed_dim * 2
        self.n_layers = config["n_mm_layers"]
        self.dropout = config["dropout"]
        self.reg_weight = config["reg_weight"]
        self.knn_k = config["knn_k"]
        
        # User-Item embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        
        # Modal encoders
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_proj = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)
            self.image_encoder = ModalEncoder(self.feat_embed_dim, self.hidden_dim, self.feat_embed_dim, self.dropout)
            
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_proj = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)
            self.text_encoder = ModalEncoder(self.feat_embed_dim, self.hidden_dim, self.feat_embed_dim, self.dropout)
            
        # Graph layers
        self.mm_layers = nn.ModuleList([
            GraphAttentionLayer(self.feat_embed_dim, self.dropout)
            for _ in range(self.n_layers)
        ])
        
        self.ui_layers = nn.ModuleList([
            GraphAttentionLayer(self.embedding_dim, self.dropout)
            for _ in range(self.n_layers)
        ])
        
        # Modal fusion
        if self.v_feat is not None and self.t_feat is not None:
            self.fusion = nn.Sequential(
                nn.Linear(self.feat_embed_dim * 2, self.feat_embed_dim * 2),
                nn.LayerNorm(self.feat_embed_dim * 2),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.feat_embed_dim * 2, self.feat_embed_dim * 2)
            )
            self.fusion_gate = nn.Linear(self.feat_embed_dim * 2, 2)
        
        # Load data
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.edge_index = self.build_edges()
        self.mm_edge_index = None
        self.build_modal_graph()
        
        self.to(self.device)
        
        # Add degree-sensitive dropout
        self.degree_ratio = config.get("degree_ratio", 0.8)
        self.masked_adj = None
        self.edge_values = None
        
        # Add learnable temperature parameter
        self.tau = nn.Parameter(torch.FloatTensor([1.0]))
        
        # Add edge weights for GAT
        if self.v_feat is not None:
            self.image_edge_weights = nn.Parameter(torch.ones(self.mm_edge_index.size(1), 1))
        if self.t_feat is not None:
            self.text_edge_weights = nn.Parameter(torch.ones(self.mm_edge_index.size(1), 1))

    def build_edges(self):
        rows = self.interaction_matrix.row
        cols = self.interaction_matrix.col + self.n_users
        
        edge_index = torch.tensor(np.vstack([
            np.concatenate([rows, cols]),
            np.concatenate([cols, rows])
        ]), dtype=torch.long).to(self.device)
        
        return edge_index

    def build_modal_graph(self):
        if self.v_feat is None and self.t_feat is None:
            return
            
        if self.v_feat is not None:
            feats = F.normalize(self.v_feat, p=2, dim=1)
            sim = torch.mm(feats, feats.t())
            sim = sim / self.tau
            
        if self.t_feat is not None:
            feats = F.normalize(self.t_feat, p=2, dim=1)
            sim = torch.mm(feats, feats.t())
            sim = sim / self.tau
        
        values, indices = sim.topk(k=self.knn_k, dim=1)
        rows = torch.arange(feats.size(0), device=self.device).view(-1, 1).expand_as(indices)
        
        edge_index = torch.stack([
            torch.cat([rows.reshape(-1), indices.reshape(-1)]),
            torch.cat([indices.reshape(-1), rows.reshape(-1)])
        ])
        
        # Add self-loops to modal graph
        edge_index, _ = add_self_loops(edge_index)
        self.mm_edge_index = edge_index.to(self.device)

    def pre_epoch_processing(self):
        if self.dropout <= 0.0:
            self.masked_adj = self.edge_index
            return
            
        # Degree-sensitive edge pruning like FREEDOM
        degree_len = int(self.edge_values.size(0) * (1.0 - self.dropout))
        degree_idx = torch.multinomial(self.edge_values, degree_len)
        keep_indices = self.edge_index[:, degree_idx]
        
        # Normalize adjacency matrix
        keep_values = self._normalize_adj_m(keep_indices)
        self.masked_adj = keep_indices
        self.edge_values = keep_values

    def _normalize_adj_m(self, indices):
        adj = torch.sparse.FloatTensor(
            indices, 
            torch.ones_like(indices[0]), 
            torch.Size([self.n_users + self.n_items, self.n_users + self.n_items])
        ).to(self.device)
        
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return values

    def forward(self):
        # Process modalities with improved attention
        img_emb = txt_emb = None
        
        if self.v_feat is not None:
            img_feat = self.image_proj(self.image_embedding.weight)
            img_emb = self.image_encoder(img_feat) + img_feat
            for layer in self.mm_layers:
                img_emb = layer(img_emb, self.mm_edge_index, self.image_edge_weights)
                
        if self.t_feat is not None:
            txt_feat = self.text_proj(self.text_embedding.weight)
            txt_emb = self.text_encoder(txt_feat) + txt_feat
            for layer in self.mm_layers:
                txt_emb = layer(txt_emb, self.mm_edge_index, self.text_edge_weights)
        
        # Enhanced modal fusion with cross-attention
        if img_emb is not None and txt_emb is not None:
            concat_emb = torch.cat([img_emb, txt_emb], dim=1)
            fused_emb = self.fusion(concat_emb)
            gates = F.softmax(self.fusion_gate(concat_emb) / self.tau, dim=-1)
            modal_emb = gates[:, 0].unsqueeze(1) * img_emb + gates[:, 1].unsqueeze(1) * txt_emb
            
            # Add cross-modal attention
            cross_attn = torch.matmul(img_emb, txt_emb.transpose(0, 1))
            cross_attn = F.softmax(cross_attn / self.tau, dim=-1)
            modal_emb = modal_emb + torch.matmul(cross_attn, txt_emb)
        else:
            modal_emb = img_emb if img_emb is not None else txt_emb

        # Process user-item graph with degree-sensitive dropout
        x = torch.cat([self.user_embedding.weight, self.item_embedding.weight])
        all_embs = [x]
        
        edge_index = self.masked_adj if self.training else self.edge_index
        
        for layer in self.ui_layers:
            x = layer(x, edge_index)
            all_embs.append(x)
            
        x = torch.stack(all_embs, dim=1)
        x = x.mean(dim=1)
        
        user_emb, item_emb = torch.split(x, [self.n_users, self.n_items])
        
        # Enhanced fusion of ID and modal embeddings
        if modal_emb is not None:
            item_emb = item_emb + F.normalize(modal_emb, p=2, dim=1)
            
        return user_emb, item_emb, img_emb, txt_emb

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        
        user_emb, item_emb, img_emb, txt_emb = self.forward()
        
        u_emb = user_emb[users]
        pos_emb = item_emb[pos_items]
        neg_emb = item_emb[neg_items]
        
        # BPR loss
        pos_scores = torch.sum(u_emb * pos_emb, dim=1)
        neg_scores = torch.sum(u_emb * neg_emb, dim=1)
        mf_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        
        # Enhanced modal contrastive loss with hard negative mining
        modal_loss = 0.0
        if img_emb is not None and txt_emb is not None:
            # Compute all pairwise similarities
            sim_matrix = torch.matmul(img_emb, txt_emb.transpose(0, 1)) / self.tau
            
            # Positive pairs
            pos_sim = torch.diagonal(sim_matrix[pos_items])
            
            # Hard negative mining
            neg_sim = sim_matrix[pos_items]
            neg_sim.fill_diagonal_(-float('inf'))
            hardest_negs = neg_sim.max(dim=1)[0]
            
            modal_loss = -torch.mean(F.logsigmoid(pos_sim - hardest_negs))
        
        # Temperature regularization
        temp_loss = 0.1 * torch.abs(self.tau - 1.0)
        
        # Regularization loss
        reg_loss = self.reg_weight * (
            torch.norm(u_emb) +
            torch.norm(pos_emb) +
            torch.norm(neg_emb)
        )
        
        # Add L2 regularization for modal embeddings
        if img_emb is not None:
            reg_loss += self.reg_weight * torch.norm(img_emb)
        if txt_emb is not None:
            reg_loss += self.reg_weight * torch.norm(txt_emb)
            
        # Adjust loss weights
        return mf_loss + 0.2 * modal_loss + reg_loss + temp_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_emb, item_emb, _, _ = self.forward()
        scores = torch.matmul(user_emb[user], item_emb.transpose(0, 1))
        return scores