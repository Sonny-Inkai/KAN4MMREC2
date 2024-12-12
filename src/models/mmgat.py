import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np
import math
from typing import Optional

class MMGAT(nn.Module):
    def __init__(self, config, dataset):
        super(MMGAT, self).__init__()
        
        # Basic configuration
        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads'] if 'n_heads' in config else 4  # Default to 4 if not specified
        self.dropout = config['dropout']
        self.reg_weight = config['reg_weight']
        
        # Dataset information
        self.n_users = dataset.user_num
        self.n_items = dataset.item_num
        self.n_nodes = self.n_users + self.n_items
        
        # Load interaction matrix
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        
        # Initialize embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)
        
        # Feature processing
        self.v_feat = getattr(dataset, 'v_feat', None)
        self.t_feat = getattr(dataset, 't_feat', None)
        
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_transform = nn.Sequential(
                nn.Linear(self.v_feat.shape[1], self.feat_embed_dim),
                nn.LayerNorm(self.feat_embed_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout)
            )
            self.image_attention = MultiHeadAttention(
                dim=self.feat_embed_dim,
                n_heads=self.n_heads,
                dropout=self.dropout
            )
            
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_transform = nn.Sequential(
                nn.Linear(self.t_feat.shape[1], self.feat_embed_dim),
                nn.LayerNorm(self.feat_embed_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout)
            )
            self.text_attention = MultiHeadAttention(
                dim=self.feat_embed_dim,
                n_heads=self.n_heads,
                dropout=self.dropout
            )
        
        # Initialize adjacency matrix
        try:
            self.norm_adj = self.get_norm_adj_mat()
        except Exception as e:
            print(f"Warning: Error in adjacency matrix initialization: {str(e)}")
            # Provide fallback initialization
            self.norm_adj = self._initialize_fallback_adj()
        
        # Fusion layer
        self.fusion = ModalFusion(self.feat_embed_dim)
        
        # Output transformation
        self.output_layer = nn.Linear(self.feat_embed_dim, self.embedding_dim)

    def _initialize_fallback_adj(self):
        indices = torch.arange(self.n_nodes).repeat(2, 1)
        values = torch.ones(self.n_nodes)
        return torch.sparse_coo_tensor(
            indices, 
            values,
            (self.n_nodes, self.n_nodes)
        )

    def get_norm_adj_mat(self):
        adj_mat = sp.dok_matrix((self.n_nodes, self.n_nodes), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.interaction_matrix.tolil()
        
        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()
        
        adj_mat = adj_mat + sp.eye(adj_mat.shape[0])
        
        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.zeros_like(rowsum)
        d_inv[rowsum.nonzero()] = np.power(rowsum[rowsum.nonzero()], -0.5).flatten()
        d_mat_inv = sp.diags(d_inv.flatten())
        norm_adj = d_mat_inv.dot(adj_mat).dot(d_mat_inv)
        
        norm_adj = norm_adj.tocoo()
        indices = torch.tensor([norm_adj.row, norm_adj.col], dtype=torch.long)
        values = torch.tensor(norm_adj.data, dtype=torch.float)
        shape = torch.Size(norm_adj.shape)
        
        return torch.sparse_coo_tensor(indices, values, shape)

    def forward(self, users, items=None):
        item_features = []
        
        if self.v_feat is not None:
            image_feats = self.image_transform(self.image_embedding.weight)
            image_feats = self.image_attention(image_feats)
            item_features.append(image_feats)
            
        if self.t_feat is not None:
            text_feats = self.text_transform(self.text_embedding.weight)
            text_feats = self.text_attention(text_feats)
            item_features.append(text_feats)
        
        if len(item_features) > 1:
            item_features = self.fusion(item_features)
        elif len(item_features) == 1:
            item_features = item_features[0]
        else:
            item_features = self.item_id_embedding.weight
            
        item_embeddings = self.output_layer(item_features)
        final_item_embeddings = item_embeddings + self.item_id_embedding.weight
        user_embeddings = self.user_embedding.weight
        
        if items is not None:
            user_embeddings = user_embeddings[users]
            final_item_embeddings = final_item_embeddings[items]
            
        return user_embeddings, final_item_embeddings

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        
        user_embeddings, item_embeddings = self.forward(
            users, 
            torch.cat([pos_items, neg_items])
        )
        
        pos_embeddings, neg_embeddings = torch.split(
            item_embeddings, 
            [pos_items.shape[0], neg_items.shape[0]]
        )
        
        pos_scores = torch.sum(user_embeddings * pos_embeddings, dim=1)
        neg_scores = torch.sum(user_embeddings * neg_embeddings, dim=1)
        bpr_loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores)))
        
        reg_loss = self.reg_weight * (
            torch.norm(user_embeddings) +
            torch.norm(pos_embeddings) +
            torch.norm(neg_embeddings)
        )
        
        return bpr_loss + reg_loss

    def predict(self, interaction):
        users = interaction[0]
        items = interaction[1]
        
        user_embeddings, item_embeddings = self.forward(users, items)
        scores = torch.sum(user_embeddings * item_embeddings, dim=1)
        return scores

    def full_sort_predict(self, interaction):
        users = interaction[0]
        user_embeddings, item_embeddings = self.forward(users)
        scores = torch.matmul(user_embeddings, item_embeddings.transpose(0, 1))
        return scores

class MultiHeadAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert dim % n_heads == 0, f"dim ({dim}) must be divisible by n_heads ({n_heads})"
        
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: t.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2),
            qkv
        )
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(batch_size, seq_len, self.dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x

class ModalFusion(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 1)
        )
        
    def forward(self, features):
        weights = []
        for feat in features:
            weight = self.attention(feat)
            weights.append(weight)
            
        weights = torch.stack(weights, dim=1)
        weights = F.softmax(weights, dim=1)
        
        output = torch.zeros_like(features[0])
        for i, feat in enumerate(features):
            output += weights[:, i] * feat
            
        return output