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
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim)
        )
        
    def forward(self, x):
        return F.normalize(self.encoder(x), p=2, dim=1)

class GraphAttentionLayer(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.gat = GATConv(dim, dim // 8, heads=8, dropout=dropout, concat=True)
        self.norm = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Linear(dim, dim)
        self.act = nn.GELU()
        
    def forward(self, x, edge_index):
        identity = self.skip(x)
        out = self.gat(x, edge_index)
        out = self.norm(out)
        out = self.act(out)
        out = self.dropout(out)
        return out + identity

class MMGAT(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MMGAT, self).__init__(config, dataset)
        
        self.embedding_dim = config["embedding_size"]
        self.feat_embed_dim = config["feat_embed_dim"]
        self.hidden_dim = self.feat_embed_dim * 4
        self.n_layers = config["n_mm_layers"]
        self.dropout = config["dropout"]
        self.reg_weight = config["reg_weight"]
        self.knn_k = config["knn_k"]
        
        # User-Item embeddings with better initialization
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        
        # Modal encoders
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_encoder = ModalEncoder(self.v_feat.shape[1], self.hidden_dim, self.feat_embed_dim, self.dropout)
            
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_encoder = ModalEncoder(self.t_feat.shape[1], self.hidden_dim, self.feat_embed_dim, self.dropout)
            
        # Multi-head Graph Attention layers
        self.mm_layers = nn.ModuleList([
            GraphAttentionLayer(self.feat_embed_dim, self.dropout)
            for _ in range(self.n_layers)
        ])
        
        self.ui_layers = nn.ModuleList([
            GraphAttentionLayer(self.embedding_dim, self.dropout)
            for _ in range(self.n_layers)
        ])
        
        # Enhanced modal fusion
        if self.v_feat is not None and self.t_feat is not None:
            self.fusion = nn.Sequential(
                nn.Linear(self.feat_embed_dim * 2, self.hidden_dim),
                nn.BatchNorm1d(self.hidden_dim),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim, self.feat_embed_dim),
                nn.BatchNorm1d(self.feat_embed_dim)
            )
            
            self.modal_weight = nn.Parameter(torch.ones(2))
            self.softmax = nn.Softmax(dim=0)
        
        # Load and process data
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.edge_index = self.build_edges()
        self.mm_edge_index = None
        self.build_modal_graph()
        
        self.to(self.device)

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
            
        if self.v_feat is not None and self.t_feat is not None:
            v_feats = F.normalize(self.v_feat, p=2, dim=1)
            t_feats = F.normalize(self.t_feat, p=2, dim=1)
            
            v_sim = torch.mm(v_feats, v_feats.t())
            t_sim = torch.mm(t_feats, t_feats.t())
            
            weights = self.softmax(self.modal_weight)
            sim = weights[0] * v_sim + weights[1] * t_sim
        else:
            feats = F.normalize(self.v_feat if self.v_feat is not None else self.t_feat, p=2, dim=1)
            sim = torch.mm(feats, feats.t())
        
        values, indices = sim.topk(k=self.knn_k, dim=1)
        rows = torch.arange(sim.size(0), device=self.device).view(-1, 1).expand_as(indices)
        
        self.mm_edge_index = torch.stack([
            torch.cat([rows.reshape(-1), indices.reshape(-1)]),
            torch.cat([indices.reshape(-1), rows.reshape(-1)])
        ]).to(self.device)

    def forward(self):
        # Process modalities with gradient scaling
        img_emb = txt_emb = None
        
        if self.v_feat is not None:
            img_emb = self.image_encoder(self.image_embedding.weight)
            for layer in self.mm_layers:
                img_emb = layer(img_emb, self.mm_edge_index)
                img_emb = F.normalize(img_emb, p=2, dim=1)
                
        if self.t_feat is not None:
            txt_emb = self.text_encoder(self.text_embedding.weight)
            for layer in self.mm_layers:
                txt_emb = layer(txt_emb, self.mm_edge_index)
                txt_emb = F.normalize(txt_emb, p=2, dim=1)
        
        # Dynamic fusion with learned weights
        if img_emb is not None and txt_emb is not None:
            weights = self.softmax(self.modal_weight)
            modal_emb = self.fusion(torch.cat([
                weights[0] * img_emb,
                weights[1] * txt_emb
            ], dim=1))
            modal_emb = F.normalize(modal_emb, p=2, dim=1)
        else:
            modal_emb = img_emb if img_emb is not None else txt_emb
        
        # Enhanced user-item graph processing
        x = torch.cat([self.user_embedding.weight, self.item_embedding.weight])
        all_embs = []
        
        for i, layer in enumerate(self.ui_layers):
            if self.training:
                x = F.dropout(x, p=self.dropout * (i + 1) / len(self.ui_layers))
            x = layer(x, self.edge_index)
            x = F.normalize(x, p=2, dim=1)
            all_embs.append(x)
            
        x = torch.stack(all_embs, dim=1)
        x = x * F.softmax(x, dim=1)
        x = x.sum(dim=1)
        
        user_emb, item_emb = torch.split(x, [self.n_users, self.n_items])
        
        # Adaptive modal integration
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
        
        # InfoNCE loss for user-item interactions
        pos_scores = torch.sum(u_emb * pos_emb, dim=1)
        neg_scores = torch.sum(u_emb * neg_emb, dim=1)
        
        pos_labels = torch.ones_like(pos_scores)
        neg_labels = torch.zeros_like(neg_scores)
        
        scores = torch.cat([pos_scores, neg_scores])
        labels = torch.cat([pos_labels, neg_labels])
        
        mf_loss = F.binary_cross_entropy_with_logits(scores, labels)
        
        # Enhanced modal contrastive loss
        modal_loss = 0.0
        if img_emb is not None and txt_emb is not None:
            temp = 0.07
            img_pos = img_emb[pos_items]
            txt_pos = txt_emb[pos_items]
            img_neg = img_emb[neg_items]
            txt_neg = txt_emb[neg_items]
            
            pos_sim = torch.exp(F.cosine_similarity(img_pos, txt_pos) / temp)
            neg_sim = torch.exp(F.cosine_similarity(img_pos, txt_neg) / temp)
            
            modal_loss = -torch.log(pos_sim / (pos_sim + neg_sim)).mean()
            
            # Cross-modal alignment
            i2t_sim = torch.exp(F.cosine_similarity(img_pos, txt_pos) / temp)
            t2i_sim = torch.exp(F.cosine_similarity(txt_pos, img_pos) / temp)
            
            modal_loss += -0.5 * (torch.log(i2t_sim).mean() + torch.log(t2i_sim).mean())
        
        # L2 regularization with weight decay
        l2_reg = self.reg_weight * (
            torch.norm(u_emb, p=2) +
            torch.norm(pos_emb, p=2) +
            torch.norm(neg_emb, p=2)
        )
        
        # Dynamic loss weighting
        if self.training:
            with torch.no_grad():
                mf_weight = torch.sigmoid(mf_loss.detach())
                modal_weight = torch.sigmoid(modal_loss.detach()) if modal_loss > 0 else torch.tensor(0.0)
                
            total_loss = (
                mf_weight * mf_loss + 
                0.5 * modal_weight * modal_loss + 
                l2_reg
            )
        else:
            total_loss = mf_loss + 0.5 * modal_loss + l2_reg
        
        return total_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_emb, item_emb, _, _ = self.forward()
        scores = torch.matmul(user_emb[user], item_emb.transpose(0, 1))
        return scores