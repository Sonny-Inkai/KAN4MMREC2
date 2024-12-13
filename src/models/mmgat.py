# coding: utf-8
import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from common.abstract_recommender import GeneralRecommender

class ModalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        self.predictor = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim)
        )
        
    def forward(self, x):
        h = F.normalize(self.encoder(x), p=2, dim=1)
        z = F.normalize(self.predictor(h.detach()), p=2, dim=1)
        return h, z

class GraphAttentionAggregator(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.gat = GATConv(dim, dim // heads, heads=heads, dropout=dropout)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim)
        )
        
    def forward(self, x, edge_index):
        # Multi-head attention with residual
        h = self.gat(x, edge_index)
        h = self.dropout(h)
        h = self.norm(x + h)
        
        # FFN with residual
        out = self.ffn(h)
        out = self.dropout(out)
        out = self.norm(h + out)
        return out

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
        self.temperature = 0.2
        
        # Enhanced embeddings with better initialization
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        self.init_weights()
        
        # Modal processors
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_encoder = ModalEncoder(self.v_feat.shape[1], self.hidden_dim, self.feat_embed_dim, self.dropout)
            
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_encoder = ModalEncoder(self.t_feat.shape[1], self.hidden_dim, self.feat_embed_dim, self.dropout)
            
        # Graph message passing layers
        self.mm_layers = nn.ModuleList([
            GraphAttentionAggregator(self.feat_embed_dim)
            for _ in range(self.n_layers)
        ])
        
        self.ui_layers = nn.ModuleList([
            GraphAttentionAggregator(self.embedding_dim)
            for _ in range(self.n_layers)
        ])
        
        # Advanced modal fusion
        if self.v_feat is not None and self.t_feat is not None:
            self.fusion = nn.Sequential(
                nn.Linear(self.feat_embed_dim * 2, self.feat_embed_dim),
                nn.LayerNorm(self.feat_embed_dim),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.feat_embed_dim, self.feat_embed_dim)
            )
            self.fusion_gate = nn.Sequential(
                nn.Linear(self.feat_embed_dim * 2, 2),
                nn.Softmax(dim=-1)
            )
        
        # Graph structure
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.edge_index = self.build_edges()
        self.mm_edge_index = None
        self.build_modal_graph()
        
        self.to(self.device)
        
    def init_weights(self):
        nn.init.xavier_uniform_(self.user_embedding.weight, gain=1/2)
        nn.init.xavier_uniform_(self.item_embedding.weight, gain=1/2)

    def build_edges(self):
        rows = self.interaction_matrix.row
        cols = self.interaction_matrix.col + self.n_users
        
        edge_index = torch.tensor(np.vstack([
            np.concatenate([rows, cols, cols, rows]),
            np.concatenate([cols, rows, rows, cols])
        ]), dtype=torch.long).to(self.device)
        
        return edge_index

    def build_modal_graph(self):
        if not (self.v_feat is not None or self.t_feat is not None):
            return
            
        feat = None
        if self.v_feat is not None:
            feat = F.normalize(self.v_feat, p=2, dim=1)
        if self.t_feat is not None:
            feat = F.normalize(self.t_feat, p=2, dim=1)
            
        sim = torch.mm(feat, feat.t())
        values, indices = sim.topk(k=self.knn_k, dim=1)
        rows = torch.arange(feat.size(0), device=self.device).view(-1, 1).expand_as(indices)
        
        self.mm_edge_index = torch.stack([
            torch.cat([rows.reshape(-1), indices.reshape(-1)]),
            torch.cat([indices.reshape(-1), rows.reshape(-1)])
        ]).to(self.device)

    def message_dropout(self, x):
        if not self.training:
            return x
        mask = torch.bernoulli((1 - self.dropout) * torch.ones_like(x))
        return x * mask / (1 - self.dropout)

    def forward(self):
        # Process modalities with enhanced feature extraction
        img_emb = txt_emb = None
        img_pred = txt_pred = None
        
        if self.v_feat is not None:
            img_emb, img_pred = self.image_encoder(self.image_embedding.weight)
            for layer in self.mm_layers:
                img_emb = layer(img_emb, self.mm_edge_index)
                
        if self.t_feat is not None:
            txt_emb, txt_pred = self.text_encoder(self.text_embedding.weight)
            for layer in self.mm_layers:
                txt_emb = layer(txt_emb, self.mm_edge_index)
                
        # Advanced modal fusion with gating
        if img_emb is not None and txt_emb is not None:
            concat_emb = torch.cat([img_emb, txt_emb], dim=1)
            gate_weights = self.fusion_gate(concat_emb)
            modal_emb = self.fusion(concat_emb)
            modal_emb = gate_weights[:, 0:1] * img_emb + gate_weights[:, 1:2] * txt_emb + modal_emb
            modal_emb = F.normalize(modal_emb, p=2, dim=1)
        else:
            modal_emb = img_emb if img_emb is not None else txt_emb
        
        # Enhanced graph convolution
        x = torch.cat([self.user_embedding.weight, self.item_embedding.weight])
        all_embs = [x]
        
        for layer in self.ui_layers:
            x = self.message_dropout(x)
            x = layer(x, self.edge_index)
            all_embs.append(F.normalize(x, p=2, dim=1))
            
        x = torch.stack(all_embs, dim=1)
        x = torch.mean(x, dim=1)
        
        user_emb, item_emb = torch.split(x, [self.n_users, self.n_items])
        
        if modal_emb is not None:
            item_emb = item_emb + modal_emb
            
        return user_emb, item_emb, (img_emb, txt_emb), (img_pred, txt_pred)

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        
        user_emb, item_emb, (img_emb, txt_emb), (img_pred, txt_pred) = self.forward()
        
        u_e = user_emb[users]
        pos_e = item_emb[pos_items]
        neg_e = item_emb[neg_items]
        
        # Enhanced BPR loss with temperature scaling
        pos_scores = torch.sum(u_e * pos_e, dim=1) / self.temperature
        neg_scores = torch.sum(u_e * neg_e, dim=1) / self.temperature
        mf_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        
        # Advanced contrastive loss for modalities
        modal_loss = 0.0
        if img_emb is not None and txt_emb is not None:
            i_emb = img_emb[pos_items]
            t_emb = txt_emb[pos_items]
            i_pred = img_pred[pos_items]
            t_pred = txt_pred[pos_items]
            
            pos_sim = torch.sum(i_emb * t_emb, dim=1) / self.temperature
            neg_sim = torch.sum(i_emb * txt_emb[neg_items], dim=1) / self.temperature
            
            modal_loss = -(
                torch.mean(F.logsigmoid(pos_sim - neg_sim)) +
                torch.mean(F.cosine_similarity(i_emb, t_pred)) +
                torch.mean(F.cosine_similarity(t_emb, i_pred))
            ) / 3
            
        # L2 regularization with momentum
        reg_loss = self.reg_weight * (
            torch.norm(u_e) +
            torch.norm(pos_e) +
            torch.norm(neg_e)
        )
        
        return mf_loss + 0.2 * modal_loss + reg_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_emb, item_emb, _, _ = self.forward()
        scores = torch.matmul(user_emb[user], item_emb.transpose(0, 1))
        return scores