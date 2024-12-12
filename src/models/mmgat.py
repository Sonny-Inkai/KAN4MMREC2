# coding: utf-8
import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from common.abstract_recommender import GeneralRecommender

class ModalEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim, dropout=0.1):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.predictor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
    def forward(self, x):
        feat = self.projector(x)
        pred = self.predictor(feat.detach())
        return feat, pred

class GraphAttention(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x, adj):
        B, N, C = x.shape
        q = self.q_proj(x).reshape(B, N, self.heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, N, self.heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, N, self.heads, self.head_dim).permute(0, 2, 1, 3)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.masked_fill(adj.unsqueeze(1) == 0, -1e9)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.o_proj(x)
        return self.norm(x)

class MMGAT(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MMGAT, self).__init__(config, dataset)
        
        # Basic settings
        self.n_users = dataset.user_num
        self.n_items = dataset.item_num
        self.n_nodes = self.n_users + self.n_items
        
        self.embedding_dim = config["embedding_size"]
        self.feat_embed_dim = config["feat_embed_dim"]
        self.n_layers = config["n_mm_layers"]
        self.reg_weight = config["reg_weight"]
        self.dropout = config["dropout"]
        self.temperature = 0.2
        
        # Initialize embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_normal_(self.user_embedding.weight)
        nn.init.xavier_normal_(self.item_embedding.weight)
        
        # Modal encoders
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_encoder = ModalEncoder(self.v_feat.shape[1], self.feat_embed_dim, self.dropout)
            
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_encoder = ModalEncoder(self.t_feat.shape[1], self.feat_embed_dim, self.dropout)
        
        # Graph attention layers
        self.gat_layers = nn.ModuleList([
            GraphAttention(self.embedding_dim, dropout=self.dropout)
            for _ in range(self.n_layers)
        ])
        
        # Load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        
        # Initialize device
        self.to(self.device)
        
        # Build graph structure
        self.norm_adj = self.build_norm_adj().to(self.device)
        
    def build_norm_adj(self):
        adj_mat = sp.dok_matrix((self.n_nodes, self.n_nodes), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.interaction_matrix.tolil()
        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()
        
        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum + 1e-7, -0.5).flatten()
        d_mat = sp.diags(d_inv)
        norm_adj = d_mat.dot(adj_mat).dot(d_mat)
        
        coo = norm_adj.tocoo()
        indices = np.vstack((coo.row, coo.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(coo.data)
        shape = torch.Size(coo.shape)
        return torch.sparse_coo_tensor(i, v, shape)

    def forward(self):
        # Process modalities
        img_feat, img_pred = None, None
        txt_feat, txt_pred = None, None
        
        if self.v_feat is not None:
            img_feat, img_pred = self.image_encoder(self.image_embedding.weight)
            
        if self.t_feat is not None:
            txt_feat, txt_pred = self.text_encoder(self.text_embedding.weight)
        
        # Combine modal features
        modal_feat = torch.zeros_like(self.item_embedding.weight)
        if img_feat is not None:
            modal_feat = modal_feat + img_feat
        if txt_feat is not None:
            modal_feat = modal_feat + txt_feat
        if img_feat is not None and txt_feat is not None:
            modal_feat = modal_feat / 2
        
        # Graph attention with residual connections
        ego_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0).unsqueeze(0)
        all_embeddings = []
        
        for layer in self.gat_layers:
            ego_embeddings = layer(ego_embeddings, self.norm_adj)
            all_embeddings.append(ego_embeddings)
        
        all_embeddings = torch.stack(all_embeddings, dim=1).mean(dim=1).squeeze(0)
        user_emb, item_emb = torch.split(all_embeddings, [self.n_users, self.n_items])
        
        item_emb = item_emb + modal_feat
        
        return user_emb, item_emb, (img_feat, img_pred), (txt_feat, txt_pred)

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        
        user_emb, item_emb, (img_feat, img_pred), (txt_feat, txt_pred) = self.forward()
        
        # BPR loss
        u_e = user_emb[users]
        pos_e = item_emb[pos_items]
        neg_e = item_emb[neg_items]
        
        pos_scores = torch.sum(u_e * pos_e, dim=1)
        neg_scores = torch.sum(u_e * neg_e, dim=1)
        rec_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        
        # Modal contrastive loss
        modal_loss = 0.0
        if img_feat is not None and txt_feat is not None:
            pos_sim = torch.sum(F.normalize(img_feat[pos_items], dim=1) * F.normalize(txt_feat[pos_items], dim=1), dim=1)
            neg_sim = torch.sum(F.normalize(img_feat[pos_items], dim=1) * F.normalize(txt_feat[neg_items], dim=1), dim=1)
            modal_loss = -torch.mean(F.logsigmoid((pos_sim - neg_sim) / self.temperature))

        # Multi-task regularization
        reg_loss = self.reg_weight * (torch.norm(u_e) + torch.norm(pos_e) + torch.norm(neg_e))
        
        # Item feature consistency loss
        feat_loss = 0.0
        if img_feat is not None:
            feat_loss += -torch.mean(F.logsigmoid(torch.sum(F.normalize(pos_e, dim=1) * F.normalize(img_feat[pos_items], dim=1), dim=1)))
        if txt_feat is not None:
            feat_loss += -torch.mean(F.logsigmoid(torch.sum(F.normalize(pos_e, dim=1) * F.normalize(txt_feat[pos_items], dim=1), dim=1)))
            
        return rec_loss + 0.2 * modal_loss + 0.1 * feat_loss + reg_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_emb, item_emb, _, _ = self.forward()
        scores = torch.matmul(user_emb[user], item_emb.transpose(0, 1))
        return scores