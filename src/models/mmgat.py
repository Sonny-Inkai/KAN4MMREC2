# coding: utf-8
import os 
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from common.abstract_recommender import GeneralRecommender

class FeatureEncoder(nn.Module):
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
        
    def forward(self, x):
        return self.encoder(x)

class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.temperature
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output

class MultiModalAttention(nn.Module):
    def __init__(self, dim, n_heads=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        
        self.w_q = nn.Linear(dim, dim)
        self.w_k = nn.Linear(dim, dim)
        self.w_v = nn.Linear(dim, dim)
        self.fc = nn.Linear(dim, dim)
        
        self.attention = ScaledDotProductAttention(temperature=self.head_dim ** 0.5, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, q, k, v):
        batch_size = q.size(0)
        q = self.w_q(q).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        
        output = self.attention(q, k, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.dim)
        output = self.dropout(self.fc(output))
        output = self.norm(output)
        return output

class MMGAT(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MMGAT, self).__init__(config, dataset)
        
        self.embedding_dim = config["embedding_size"]
        self.feat_embed_dim = config["feat_embed_dim"]
        self.hidden_dim = self.feat_embed_dim * 2
        self.n_layers = config["n_mm_layers"]
        self.reg_weight = config["reg_weight"]
        self.dropout = config["dropout"]
        
        # Basic embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_normal_(self.user_embedding.weight, gain=1/2)
        nn.init.xavier_normal_(self.item_embedding.weight, gain=1/2)
        
        # Modal encoders
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_encoder = FeatureEncoder(self.v_feat.shape[1], self.hidden_dim, self.feat_embed_dim, self.dropout)
            
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_encoder = FeatureEncoder(self.t_feat.shape[1], self.hidden_dim, self.feat_embed_dim, self.dropout)
        
        # Multimodal attention
        self.modal_attn = MultiModalAttention(self.feat_embed_dim, n_heads=4, dropout=self.dropout)
        
        # Graph structure
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.adj_norm = self.build_adj_matrix().to(self.device)
        self.to(self.device)

    def build_adj_matrix(self):
        # Build normalized adjacency matrix
        adj = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj = adj.tolil()
        R = self.interaction_matrix.tolil()
        adj[:self.n_users, self.n_users:] = R
        adj[self.n_users:, :self.n_users] = R.T
        adj = adj.todok()
        
        # Symmetric normalization
        rowsum = np.array(adj.sum(axis=1))
        degree_inv = np.power(rowsum + 1e-9, -0.5).flatten()
        degree_mat_inv = sp.diags(degree_inv)
        norm_adj = degree_mat_inv.dot(adj).dot(degree_mat_inv)
        
        # Convert to sparse tensor
        norm_adj = norm_adj.tocoo()
        indices = torch.LongTensor([norm_adj.row, norm_adj.col])
        values = torch.FloatTensor(norm_adj.data)
        return torch.sparse_coo_tensor(indices, values, norm_adj.shape)

    def message_dropout(self, adj):
        if not self.training or self.dropout == 0:
            return adj
            
        mask = torch.bernoulli(torch.ones(adj._values().size()) * (1 - self.dropout)).to(self.device)
        values = adj._values() * mask
        return torch.sparse_coo_tensor(adj._indices(), values, adj.size())

    def forward(self):
        # Process modalities 
        img_feat = txt_feat = None
        modal_feat = None
        
        if self.v_feat is not None:
            img_feat = self.image_encoder(self.image_embedding.weight)
            
        if self.t_feat is not None:
            txt_feat = self.text_encoder(self.text_embedding.weight)
            
        # Multimodal fusion with attention
        if img_feat is not None and txt_feat is not None:
            # Cross-modal attention
            modal_feat = self.modal_attn(
                img_feat.unsqueeze(0),
                torch.stack([img_feat, txt_feat], dim=1),
                torch.stack([img_feat, txt_feat], dim=1)
            ).squeeze(0)
            modal_feat = F.normalize(modal_feat, p=2, dim=1)
        else:
            modal_feat = img_feat if img_feat is not None else txt_feat
            
        # Graph convolution with dropouts
        ego_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        all_embeddings = [F.normalize(ego_embeddings, p=2, dim=1)]
        
        adj_dropout = self.message_dropout(self.adj_norm)
        for _ in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(adj_dropout, ego_embeddings)
            ego_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
            all_embeddings.append(ego_embeddings)
            
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        
        # Split user-item embeddings
        user_embeddings, item_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items])
        
        # Enhance item embeddings with modal features
        item_embeddings = item_embeddings + modal_feat
        
        return user_embeddings, item_embeddings, (img_feat, txt_feat)

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        
        user_embeddings, item_embeddings, (img_feat, txt_feat) = self.forward()
        
        user_e = user_embeddings[users]
        pos_e = item_embeddings[pos_items]
        neg_e = item_embeddings[neg_items]
        
        # Recommendation loss
        pos_scores = torch.sum(user_e * pos_e, dim=1)
        neg_scores = torch.sum(user_e * neg_e, dim=1)
        rec_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        
        # Modal-specific loss
        modal_loss = 0.0
        if img_feat is not None and txt_feat is not None:
            img_e = F.normalize(img_feat[pos_items], p=2, dim=1)
            txt_e = F.normalize(txt_feat[pos_items], p=2, dim=1)
            modal_loss = 1 - torch.mean(F.cosine_similarity(img_e, txt_e))
        
        # L2 regularization
        l2_loss = self.reg_weight * (
            torch.norm(user_e) + 
            torch.norm(pos_e) + 
            torch.norm(neg_e)
        )
        
        return rec_loss + 0.1 * modal_loss + l2_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_embeddings, item_embeddings, _ = self.forward()
        scores = torch.matmul(user_embeddings[user], item_embeddings.transpose(0, 1))
        return scores