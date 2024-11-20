# kronos.py

import os
import math
import random
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import remove_self_loops, add_self_loops
from einops import rearrange, repeat

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
        self.ssl_weight = config['ssl_weight']
        self.ssl_temp = config['ssl_temperature']
        self.proto_reg = config['proto_reg']
        self.n_prototypes = config['n_prototypes']
        
        self.n_nodes = self.n_users + self.n_items
        
        # Load interaction matrix
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.norm_adj = self.get_norm_adj_mat().to(self.device)
        
        # Initialize embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_normal_(self.user_embedding.weight)
        nn.init.xavier_normal_(self.item_embedding.weight)
        
        # Initialize prototypes
        self.prototypes = nn.Parameter(torch.randn(self.n_prototypes, self.embedding_dim))
        nn.init.xavier_normal_(self.prototypes)
        
        # Modality encoders
        if self.v_feat is not None:
            self.v_encoder = ModalityTransformer(
                input_dim=self.v_feat.shape[1],
                hidden_dim=self.feat_embed_dim,
                n_heads=self.n_heads,
                n_layers=self.n_layers,
                dropout=self.dropout
            )
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            
        if self.t_feat is not None:
            self.t_encoder = ModalityTransformer(
                input_dim=self.t_feat.shape[1],
                hidden_dim=self.feat_embed_dim,
                n_heads=self.n_heads,
                n_layers=self.n_layers,
                dropout=self.dropout
            )
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
        
        # Cross-modal fusion
        self.fusion = MultiModalFusion(
            dim=self.feat_embed_dim,
            n_heads=self.n_heads,
            dropout=self.dropout
        )
        
        # Graph transformer layers
        self.graph_transformer = GraphTransformer(
            dim=self.feat_embed_dim,
            depth=self.n_layers,
            heads=self.n_heads,
            dropout=self.dropout
        )
        
        # Interest extraction
        self.interest_extractor = InterestExtractor(
            dim=self.feat_embed_dim,
            n_interests=4,
            n_heads=self.n_heads
        )
        
        # Contrastive learner
        self.contrastive = HierarchicalContrastive(
            dim=self.feat_embed_dim,
            temp=self.ssl_temp
        )
        
        # Predictors
        self.global_predictor = Predictor(self.feat_embed_dim)
        self.local_predictor = Predictor(self.feat_embed_dim)

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

    def forward(self, users, items):
        # Process modalities
        v_feat = t_feat = None
        if self.v_feat is not None:
            v_feat = self.v_encoder(self.image_embedding.weight)
        if self.t_feat is not None:
            t_feat = self.t_encoder(self.text_embedding.weight)
            
        # Fuse modalities
        if v_feat is not None and t_feat is not None:
            item_feat = self.fusion(v_feat, t_feat)
        else:
            item_feat = v_feat if v_feat is not None else t_feat
            
        # Graph transformer processing
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight
        
        all_emb = torch.cat([user_emb, item_emb + item_feat])
        all_emb = self.graph_transformer(all_emb, self.norm_adj)
        
        user_emb, item_emb = torch.split(all_emb, [self.n_users, self.n_items])
        
        # Extract user interests
        user_interests = self.interest_extractor(user_emb[users])
        
        # Global and local predictions
        global_scores = self.global_predictor(user_emb[users], item_emb[items])
        local_scores = []
        for interest in user_interests:
            local_scores.append(self.local_predictor(interest, item_emb[items]))
        local_scores = torch.stack(local_scores, dim=1)
        
        # Combine predictions
        scores = global_scores.unsqueeze(1) + local_scores
        scores = scores.max(dim=1)[0]
        
        return scores, user_interests, item_emb[items]

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        
        # Forward pass
        pos_scores, user_interests, pos_items_emb = self.forward(users, pos_items)
        neg_scores, _, neg_items_emb = self.forward(users, neg_items)
        
        # BPR loss
        bpr_loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores)))
        
        # Contrastive loss
        ssl_loss = self.contrastive(user_interests, pos_items_emb, neg_items_emb)
        
        # Prototype regularization
        proto_loss = self.prototype_reg(user_interests)
        
        # Total loss
        loss = bpr_loss + self.ssl_weight * ssl_loss + self.proto_reg * proto_loss
        
        return loss

    def prototype_reg(self, embeddings):
        # Calculate distances to prototypes
        dists = torch.cdist(embeddings, self.prototypes)
        
        # Soft assignment
        Q = 1.0 / (1.0 + dists + 1e-7)
        Q = Q / Q.sum(dim=1, keepdim=True)
        
        # Target distribution
        P = torch.pow(Q, 2) / torch.sum(Q, dim=0, keepdim=True)
        P = P / P.sum(dim=1, keepdim=True)
        
        # KL divergence
        loss = F.kl_div(Q.log(), P.detach(), reduction='batchmean')
        return loss

    def full_sort_predict(self, interaction):
        users = interaction[0]
        scores, _, _ = self.forward(users, torch.arange(self.n_items).to(self.device))
        return scores
    
# Supporting modules for KRONOS

class ModalityTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_heads, n_layers, dropout):
        super().__init__()
        self.projection = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList([
            TransformerLayer(hidden_dim, n_heads, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        x = self.projection(x)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

class TransformerLayer(nn.Module):
    def __init__(self, dim, n_heads, dropout):
        super().__init__()
        self.attention = MultiHeadAttention(dim, n_heads, dropout)
        self.ffn = FeedForward(dim, dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
    def forward(self, x):
        x = self.norm1(x + self.attention(x))
        x = self.norm2(x + self.ffn(x))
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, n_heads, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        return x

class FeedForward(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)

class MultiModalFusion(nn.Module):
    def __init__(self, dim, n_heads, dropout):
        super().__init__()
        self.cross_attention = MultiHeadAttention(dim, n_heads, dropout)
        self.self_attention = MultiHeadAttention(dim, n_heads, dropout)
        self.ffn = FeedForward(dim, dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        
    def forward(self, visual, textual):
        # Cross-modal attention
        fused = self.norm1(visual + self.cross_attention(visual, textual, textual))
        # Self attention
        fused = self.norm2(fused + self.self_attention(fused))
        # Feed forward
        fused = self.norm3(fused + self.ffn(fused))
        return fused

class GraphTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, MultiHeadAttention(dim, heads, dropout)),
                PreNorm(dim, FeedForward(dim, dropout))
            ]))
        
    def forward(self, x, adj):
        for attn, ff in self.layers:
            x = attn(x, mask=adj) + x
            x = ff(x) + x
        return x

class InterestExtractor(nn.Module):
    def __init__(self, dim, n_interests, n_heads):
        super().__init__()
        self.n_interests = n_interests
        self.attention = MultiHeadAttention(dim, n_heads, dropout=0.1)
        self.interest_embeddings = nn.Parameter(torch.randn(n_interests, dim))
        nn.init.xavier_normal_(self.interest_embeddings)
        
    def forward(self, user_emb):
        B = user_emb.size(0)
        interests = repeat(self.interest_embeddings, 'n d -> b n d', b=B)
        interests = self.attention(interests, user_emb.unsqueeze(1))
        return interests

class HierarchicalContrastive(nn.Module):
    def __init__(self, dim, temp):
        super().__init__()
        self.temp = temp
        self.proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        
    def forward(self, user_interests, pos_items, neg_items):
        # Project all embeddings
        user_interests = self.proj(user_interests)
        pos_items = self.proj(pos_items)
        neg_items = self.proj(neg_items)
        
        # Normalize embeddings
        user_interests = F.normalize(user_interests, dim=-1)
        pos_items = F.normalize(pos_items, dim=-1)
        neg_items = F.normalize(neg_items, dim=-1)
        
        # Calculate positive and negative scores
        pos_scores = torch.einsum('bnd,bd->bn', user_interests, pos_items)
        neg_scores = torch.einsum('bnd,bd->bn', user_interests, neg_items)
        
        # Contrastive loss
        pos_scores = pos_scores / self.temp
        neg_scores = neg_scores / self.temp
        
        loss = -torch.log(
            pos_scores.exp().sum(dim=1) / 
            (pos_scores.exp().sum(dim=1) + neg_scores.exp().sum(dim=1))
        ).mean()
        
        return loss

class Predictor(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )
        
    def forward(self, users, items):
        return torch.sum(self.net(users) * items, dim=-1)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
        
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)