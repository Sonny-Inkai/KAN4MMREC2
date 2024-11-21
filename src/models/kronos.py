# kronos_light.py

import os
import math
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss

class KRONOS(GeneralRecommender):
    def __init__(self, config, dataset):
        super(KRONOS, self).__init__(config, dataset)
        
        # Basic parameters
        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.n_layers = config['n_mm_layers']
        self.n_heads = config['n_heads']
        self.dropout = config['dropout']
        self.temp = config['temperature']
        self.reg_weight = config['reg_weight']
        
        self.n_nodes = self.n_users + self.n_items
        
        # Load interaction matrix
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.norm_adj = self.get_norm_adj_mat().to(self.device)
        
        # Initialize embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_normal_(self.user_embedding.weight)
        nn.init.xavier_normal_(self.item_embedding.weight)
        
        # Modality encoders
        if self.v_feat is not None:
            self.image_encoder = ModalityEncoder(
                input_dim=self.v_feat.shape[1],
                hidden_dim=self.feat_embed_dim
            )
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            
        if self.t_feat is not None:
            self.text_encoder = ModalityEncoder(
                input_dim=self.t_feat.shape[1],
                hidden_dim=self.feat_embed_dim
            )
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
        
        # Fusion layer
        self.fusion = LightweightFusion(self.feat_embed_dim)
        
        # Graph convolution layers
        self.gnn_layers = nn.ModuleList([
            LightweightGNN(self.feat_embed_dim) 
            for _ in range(self.n_layers)
        ])

    def get_norm_adj_mat(self):
        def _convert_sp_mat_to_sp_tensor(X):
            coo = X.tocoo()
            indices = torch.LongTensor([coo.row, coo.col])
            data = torch.FloatTensor(coo.data)
            return torch.sparse.FloatTensor(indices, data, torch.Size(coo.shape))
            
        adj_mat = sp.dok_matrix((self.n_nodes, self.n_nodes))
        adj_mat = adj_mat.tocsr() + sp.eye(adj_mat.shape[0])
        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0
        d_mat = sp.diags(d_inv)
        norm_adj = d_mat.dot(adj_mat).dot(d_mat)
        return _convert_sp_mat_to_sp_tensor(norm_adj)

    def forward(self):
        # Process modalities
        v_feat = t_feat = None
        if self.v_feat is not None:
            v_feat = self.image_encoder(self.image_embedding.weight)
        if self.t_feat is not None:
            t_feat = self.text_encoder(self.text_embedding.weight)
            
        # Fuse modalities
        if v_feat is not None and t_feat is not None:
            item_feat = self.fusion(v_feat, t_feat)
        else:
            item_feat = v_feat if v_feat is not None else t_feat
            
        # Graph convolution
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight + item_feat
        
        all_emb = torch.cat([user_emb, item_emb])
        embs = [all_emb]
        
        # Efficient graph convolution
        for gnn in self.gnn_layers:
            all_emb = gnn(all_emb, self.norm_adj)
            embs.append(all_emb)
            
        all_emb = torch.stack(embs, dim=1).mean(dim=1)
        user_emb, item_emb = torch.split(all_emb, [self.n_users, self.n_items])
        
        return user_emb, item_emb

    def calculate_loss(self, interaction):
        user = interaction[0]
        pos_item = interaction[1]
        neg_item = interaction[2]

        user_all_emb, item_all_emb = self.forward()
        
        u_embeddings = user_all_emb[user]
        pos_embeddings = item_all_emb[pos_item]
        neg_embeddings = item_all_emb[neg_item]

        # Calculate scores
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        
        # BPR loss
        bpr_loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8))
        
        # L2 regularization
        reg_loss = self.reg_weight * (
            torch.norm(u_embeddings) +
            torch.norm(pos_embeddings) +
            torch.norm(neg_embeddings)
        )
        
        return bpr_loss + reg_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        
        user_all_emb, item_all_emb = self.forward()
        u_embeddings = user_all_emb[user]
        
        scores = torch.matmul(u_embeddings, item_all_emb.t())
        return scores

class ModalityEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, x):
        return self.net(x)

class LightweightFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, 2),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, visual, textual):
        gate_input = torch.cat([visual, textual], dim=-1)
        weights = self.gate(gate_input)
        
        fused = weights[:, 0:1] * visual + weights[:, 1:] * textual
        return fused

class LightweightGNN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x, adj):
        # Simple but effective graph convolution
        h = torch.sparse.mm(adj, x)
        h = self.norm(h + x)  # Residual connection
        return h