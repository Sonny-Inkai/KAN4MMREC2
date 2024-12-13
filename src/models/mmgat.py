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
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        self.predictor = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        feat = self.encoder(x)
        return feat, self.predictor(feat.detach())

class GraphAttentionEncoder(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attention = nn.ModuleList([
            GATConv(dim, dim // num_heads, heads=num_heads, dropout=dropout)
            for _ in range(2)
        ])
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, 2),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x, edge_index):
        out1 = self.attention[0](x, edge_index)
        out2 = self.attention[1](x, edge_index)
        
        gate = self.gate(torch.cat([out1, out2], dim=-1))
        out = gate[:, 0:1] * out1 + gate[:, 1:2] * out2
        return self.norm(x + self.dropout(out))

class MMGAT(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MMGAT, self).__init__(config, dataset)
        
        self.embedding_dim = config["embedding_size"]
        self.feat_embed_dim = config["feat_embed_dim"]
        self.hidden_dim = self.feat_embed_dim * 2
        self.n_layers = config["n_mm_layers"]
        self.n_heads = 4
        self.dropout = config["dropout"]
        self.reg_weight = config["reg_weight"]
        
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_normal_(self.user_embedding.weight, gain=0.1)
        nn.init.xavier_normal_(self.item_embedding.weight, gain=0.1)
        
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_encoder = ModalEncoder(
                self.v_feat.shape[1], 
                self.hidden_dim, 
                self.feat_embed_dim,
                self.dropout
            )
            
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_encoder = ModalEncoder(
                self.t_feat.shape[1], 
                self.hidden_dim, 
                self.feat_embed_dim,
                self.dropout
            )
        
        self.graph_encoders = nn.ModuleList([
            GraphAttentionEncoder(self.embedding_dim, self.n_heads, self.dropout)
            for _ in range(self.n_layers)
        ])
        
        if self.v_feat is not None or self.t_feat is not None:
            self.mm_encoders = nn.ModuleList([
                GraphAttentionEncoder(self.feat_embed_dim, self.n_heads, self.dropout)
                for _ in range(self.n_layers)
            ])
            
            self.fusion = nn.Sequential(
                nn.Linear(self.feat_embed_dim * 2, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim, self.feat_embed_dim)
            )
            
            self.gate = nn.Sequential(
                nn.Linear(self.embedding_dim * 2, 2),
                nn.Softmax(dim=-1)
            )
        
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.edge_index = self.build_edges()
        self.mm_edge_index = None
        self.build_modal_graph()
        
        self.to(self.device)

    def build_edges(self):
        rows = self.interaction_matrix.row
        cols = self.interaction_matrix.col + self.n_users
        
        edge_index = torch.tensor(np.vstack([
            np.concatenate([rows, cols, cols, rows]),
            np.concatenate([cols, rows, rows, cols])
        ]), dtype=torch.long).to(self.device)
        
        return edge_index

    def build_modal_graph(self):
        if self.v_feat is None and self.t_feat is None:
            return
            
        feat = None
        if self.v_feat is not None:
            feat = self.v_feat
        if self.t_feat is not None:
            feat = self.t_feat
            
        sim = torch.matmul(F.normalize(feat, dim=1), F.normalize(feat, dim=1).t())
        values, indices = torch.topk(sim, k=20, dim=1)
        
        rows = torch.arange(feat.size(0), device=self.device).repeat_interleave(20)
        cols = indices.reshape(-1)
        
        self.mm_edge_index = torch.stack([
            torch.cat([rows, cols]),
            torch.cat([cols, rows])
        ]).to(self.device)

    def forward(self):
        modal_feats = []
        modal_preds = []
        
        if self.v_feat is not None:
            img_feat, img_pred = self.image_encoder(self.image_embedding.weight)
            for encoder in self.mm_encoders:
                img_feat = encoder(img_feat, self.mm_edge_index)
            modal_feats.append(img_feat)
            modal_preds.append(img_pred)
            
        if self.t_feat is not None:
            txt_feat, txt_pred = self.text_encoder(self.text_embedding.weight)
            for encoder in self.mm_encoders:
                txt_feat = encoder(txt_feat, self.mm_edge_index)
            modal_feats.append(txt_feat)
            modal_preds.append(txt_pred)
            
        if len(modal_feats) == 2:
            modal_emb = self.fusion(torch.cat(modal_feats, dim=1))
        elif len(modal_feats) == 1:
            modal_emb = modal_feats[0]
        else:
            modal_emb = None
            
        x = torch.cat([self.user_embedding.weight, self.item_embedding.weight])
        all_embs = [x]
        
        for encoder in self.graph_encoders:
            x = encoder(x, self.edge_index)
            all_embs.append(x)
            
        x = torch.stack(all_embs, dim=1).mean(dim=1)
        user_emb, item_emb = torch.split(x, [self.n_users, self.n_items])
        
        if modal_emb is not None:
            gate = self.gate(torch.cat([item_emb, modal_emb], dim=1))
            item_emb = gate[:, 0:1] * item_emb + gate[:, 1:2] * modal_emb
            
        return (
            user_emb,
            item_emb,
            modal_feats if modal_feats else [None, None],
            modal_preds if modal_preds else [None, None]
        )

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        
        user_emb, item_emb, (img_feat, txt_feat), (img_pred, txt_pred) = self.forward()
        
        u_e = user_emb[users]
        pos_e = item_emb[pos_items]
        neg_e = item_emb[neg_items]
        
        # Main recommendation loss
        pos_scores = torch.sum(u_e * pos_e, dim=1)
        neg_scores = torch.sum(u_e * neg_e, dim=1)
        rec_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        
        # Contrastive loss between modalities
        modal_loss = 0.0
        if img_feat is not None and txt_feat is not None:
            if img_pred is not None and txt_pred is not None:
                # Cross prediction between modalities
                img_sim = F.cosine_similarity(img_feat[pos_items], txt_pred[pos_items])
                txt_sim = F.cosine_similarity(txt_feat[pos_items], img_pred[pos_items])
                modal_loss = -torch.mean(F.logsigmoid(img_sim)) - torch.mean(F.logsigmoid(txt_sim))
                
                # Negative samples
                img_neg = F.cosine_similarity(img_feat[pos_items], txt_pred[neg_items])
                txt_neg = F.cosine_similarity(txt_feat[pos_items], img_pred[neg_items])
                modal_loss += 0.1 * (
                    torch.mean(F.logsigmoid(img_neg)) + 
                    torch.mean(F.logsigmoid(txt_neg))
                )
        
        # L2 regularization
        reg_loss = self.reg_weight * (
            torch.norm(u_e) +
            torch.norm(pos_e) +
            torch.norm(neg_e)
        )
        
        return rec_loss + 0.2 * modal_loss + reg_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_emb, item_emb, _, _ = self.forward()
        scores = torch.matmul(user_emb[user], item_emb.transpose(0, 1))
        return scores