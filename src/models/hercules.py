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

class HERCULES(GeneralRecommender):
    def __init__(self, config, dataset):
        super(HERCULES, self).__init__(config, dataset)
        
        # Basic parameters
        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.knn_k = config['knn_k']
        self.n_layers = config['n_mm_layers'] 
        self.n_ui_layers = config['n_ui_layers']
        self.dropout = config['dropout']
        self.reg_weight = config['reg_weight']
        self.contrast_weight = config['contrast_weight']
        self.temp = config['temperature']
        self.n_heads = config['n_heads']
        
        self.n_nodes = self.n_users + self.n_items
        
        # Initialize embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        # Get normalized adjacency matrix
        self.norm_adj = self.get_norm_adj_mat().to(self.device)
        
        # Multi-modal feature projectors 
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_encoder = EnhancedEncoder(
                input_dim=self.v_feat.shape[1],
                hidden_dim=self.feat_embed_dim,
                n_heads=self.n_heads
            )
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_encoder = EnhancedEncoder(
                input_dim=self.t_feat.shape[1],
                hidden_dim=self.feat_embed_dim,  
                n_heads=self.n_heads
            )

        # Cross-modal fusion 
        self.fusion_layer = ModalityFusion(
            dim=self.feat_embed_dim,
            n_heads=self.n_heads
        )

        # Graph structure learning
        self.graph_learner = AdaptiveGraphLearner(
            dim=self.feat_embed_dim,
            k=self.knn_k,
            temp=self.temp
        )

        # Message passing layers
        self.gnn_layers = nn.ModuleList([
            MessagePassingLayer(self.feat_embed_dim) 
            for _ in range(self.n_layers)
        ])
        
        # Contrastive predictor
        self.predictor = nn.Sequential(
            nn.Linear(self.feat_embed_dim, self.feat_embed_dim),
            nn.PReLU(),
            nn.Linear(self.feat_embed_dim, self.feat_embed_dim)
        )

    def get_norm_adj_mat(self):
        # Convert to PyTorch sparse tensor
        def _convert_sp_mat_to_sp_tensor(X):
            coo = X.tocoo()
            indices = torch.LongTensor([coo.row, coo.col])
            data = torch.FloatTensor(coo.data)
            return torch.sparse.FloatTensor(indices, data, torch.Size(coo.shape))

        # Build adjacency matrix
        adj_mat = sp.dok_matrix((self.n_nodes, self.n_nodes))
        adj_mat = adj_mat.tocsr() + sp.eye(adj_mat.shape[0])
        
        # Normalize
        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0
        d_mat = sp.diags(d_inv)
        norm_adj = d_mat.dot(adj_mat).dot(d_mat)
        
        return _convert_sp_mat_to_sp_tensor(norm_adj)

    def forward(self, dropout=False):
        # Process modalities
        v_feat = t_feat = None
        if self.v_feat is not None:
            v_feat = self.image_encoder(self.image_embedding.weight)
        if self.t_feat is not None:
            t_feat = self.text_encoder(self.text_embedding.weight)

        # Fuse modalities
        if v_feat is not None and t_feat is not None:
            mm_feat = self.fusion_layer(v_feat, t_feat)
        else:
            mm_feat = v_feat if v_feat is not None else t_feat

        # Apply dropout if in training
        if dropout and self.training:
            mm_feat = F.dropout(mm_feat, p=self.dropout)

        # Learn adaptive graph structure
        learned_graph = self.graph_learner(mm_feat)
        
        # Message passing on learned graph 
        item_emb = mm_feat
        for gnn in self.gnn_layers:
            item_emb = gnn(item_emb, learned_graph)

        # User-item interactions with LightGCN
        ego_embeddings = torch.cat([self.user_embedding.weight, item_emb], dim=0)
        all_embeddings = [ego_embeddings]
        
        for _ in range(self.n_ui_layers):
            ego_embeddings = torch.sparse.mm(self.norm_adj, ego_embeddings)
            if dropout and self.training:
                ego_embeddings = F.dropout(ego_embeddings, p=self.dropout)
            all_embeddings.append(ego_embeddings)
            
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        
        user_embeddings, item_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items])
        
        return user_embeddings, item_embeddings

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        # Get embeddings with and without dropout
        clean_user_emb, clean_item_emb = self.forward(dropout=False)
        aug_user_emb, aug_item_emb = self.forward(dropout=True)

        # BPR Loss
        u_embeddings = clean_user_emb[users]
        pos_embeddings = clean_item_emb[pos_items]
        neg_embeddings = clean_item_emb[neg_items]

        pos_scores = (u_embeddings * pos_embeddings).sum(dim=1)
        neg_scores = (u_embeddings * neg_embeddings).sum(dim=1)
        
        bpr_loss = -F.logsigmoid(pos_scores - neg_scores).mean()

        # Contrastive Loss
        u_embeddings_aug = aug_user_emb[users]
        pos_embeddings_aug = aug_item_emb[pos_items]

        u_embeddings_pred = self.predictor(u_embeddings)
        pos_embeddings_pred = self.predictor(pos_embeddings)

        pos_scores = (u_embeddings_pred * pos_embeddings_aug).sum(dim=1)
        neg_scores = (u_embeddings_pred * neg_embeddings).sum(dim=1)
        
        con_loss = -F.logsigmoid(pos_scores/self.temp - neg_scores/self.temp).mean()

        # L2 regularization
        l2_loss = self.reg_weight * (
            torch.norm(u_embeddings) +
            torch.norm(pos_embeddings) + 
            torch.norm(neg_embeddings)
        )

        loss = bpr_loss + self.contrast_weight * con_loss + l2_loss
        return loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_embeddings, item_embeddings = self.forward(dropout=False)
        u_embeddings = user_embeddings[user]
        scores = torch.matmul(u_embeddings, item_embeddings.t())
        return scores

class EnhancedEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(input_dim, n_heads)
        self.norm1 = nn.LayerNorm(input_dim) 
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim*4),
            nn.PReLU(),
            nn.Linear(hidden_dim*4, hidden_dim),
            nn.Dropout(0.1)
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        # Self attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        # FFN
        out = self.ffn(x)
        out = self.norm2(x + out)
        return out

class ModalityFusion(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, n_heads)
        self.gate = nn.Sequential(
            nn.Linear(dim*2, 2),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, visual, textual):
        # Cross attention
        fused, _ = self.attention(visual, textual, textual)
        
        # Gating
        gate_input = torch.cat([visual, textual], dim=-1)
        weights = self.gate(gate_input)
        
        out = weights[:,:1] * visual + weights[:,1:] * textual
        return out
        
class AdaptiveGraphLearner(nn.Module):
    def __init__(self, dim, k, temp=0.1):
        super().__init__()
        self.k = k
        self.temp = temp
        
    def forward(self, x):
        # Compute similarity
        sim = torch.matmul(x, x.transpose(0, 1)) 
        sim = sim / self.temp
        
        # Get top-k
        values, indices = torch.topk(sim, self.k, dim=-1)
        mask = torch.zeros_like(sim)
        mask.scatter_(-1, indices, 1)
        
        # Sparsify
        adj = sim * mask
        
        # Symmetrize
        adj = (adj + adj.transpose(0,1))/2
        return adj

class MessagePassingLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(dim, dim))
        nn.init.xavier_uniform_(self.weight)
        
    def forward(self, x, adj):
        support = torch.matmul(x, self.weight)
        out = torch.matmul(adj, support)
        return out + x  # Residual