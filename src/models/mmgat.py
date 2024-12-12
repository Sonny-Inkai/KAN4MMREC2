# coding: utf-8
import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from common.abstract_recommender import GeneralRecommender

class CrossModalAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.scale = dim ** 0.5
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x1, x2):
        q = self.q(x1)
        k = self.k(x2)
        v = self.v(x2)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn = F.softmax(attn, dim=-1)
        x = torch.matmul(attn, v)
        x = self.proj(x)
        return self.norm(x + x1)

class GraphAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.att = nn.Parameter(torch.empty(size=(1, dim)))
        nn.init.xavier_normal_(self.att.data)
        
    def forward(self, x, adj):
        x = self.linear(x)
        f_1 = torch.matmul(x, self.att.transpose(0, 1))
        f_2 = torch.matmul(x, self.att.transpose(0, 1)).transpose(0, 1)
        
        s = f_1 + f_2.transpose(0, 1)
        attention = F.softmax(F.leaky_relu(s), dim=1)
        attention = torch.sparse.mm(adj, attention * x)
        return x + attention

class MMGAT(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MMGAT, self).__init__(config, dataset)
        
        self.embedding_dim = config["embedding_size"]
        self.feat_embed_dim = config["feat_embed_dim"]
        self.n_layers = config["n_mm_layers"]
        self.temperature = 0.2
        
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_proj = nn.Linear(self.v_feat.shape[1], self.embedding_dim)
            
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_proj = nn.Linear(self.t_feat.shape[1], self.embedding_dim)
        
        self.cross_attention = CrossModalAttention(self.embedding_dim)
        self.user_attention = GraphAttention(self.embedding_dim)
        self.modal_attention = GraphAttention(self.embedding_dim)
        
        self.fusion = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
            nn.GELU()
        )
        
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.build_graphs()
        self.to(self.device)
    
    def build_graphs(self):
        A = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        R = self.interaction_matrix.tolil()
        A[:self.n_users, self.n_users:] = R
        A[self.n_users:, :self.n_users] = R.T
        A = A.todok()
        
        rowsum = np.array(A.sum(axis=1))
        d_inv = np.power(rowsum + 1e-9, -0.5).flatten()
        d_mat = sp.diags(d_inv)
        norm_adj = d_mat.dot(A).dot(d_mat)
        
        coo = norm_adj.tocoo()
        values = torch.FloatTensor(coo.data)
        indices = np.vstack((coo.row, coo.col))
        i = torch.LongTensor(indices)
        self.norm_adj = torch.sparse_coo_tensor(i, values, coo.shape).to(self.device)
        
        if self.v_feat is not None:
            self.image_graph = self.build_modal_graph(self.v_feat)
        if self.t_feat is not None:
            self.text_graph = self.build_modal_graph(self.t_feat)
    
    def build_modal_graph(self, features):
        sim = F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=2)
        values, indices = sim.topk(k=64, dim=-1)
        rows = torch.arange(features.size(0), device=features.device).repeat_interleave(64)
        cols = indices.reshape(-1)
        i = torch.stack([rows, cols])
        v = values.reshape(-1)
        adj = torch.sparse_coo_tensor(i, v, sim.shape)
        return adj.to(self.device)

    def forward(self):
        image_feat = None
        text_feat = None
        
        if self.v_feat is not None:
            image_feat = self.image_proj(self.image_embedding.weight)
            image_feat = self.modal_attention(image_feat, self.image_graph)
            
        if self.t_feat is not None:
            text_feat = self.text_proj(self.text_embedding.weight)
            text_feat = self.modal_attention(text_feat, self.text_graph)
        
        if image_feat is not None and text_feat is not None:
            img_enhanced = self.cross_attention(image_feat, text_feat)
            txt_enhanced = self.cross_attention(text_feat, image_feat)
            modal_feat = self.fusion(torch.cat([img_enhanced, txt_enhanced], dim=1))
        else:
            modal_feat = image_feat if image_feat is not None else text_feat
        
        user_emb = self.user_embedding.weight
        all_emb = torch.cat([user_emb, modal_feat], dim=0)
        embs = [all_emb]
        
        for _ in range(self.n_layers):
            all_emb = self.user_attention(all_emb, self.norm_adj)
            embs.append(all_emb)
        
        all_emb = torch.stack(embs, dim=1).mean(dim=1)
        user_emb, item_emb = torch.split(all_emb, [self.n_users, self.n_items])
        
        return user_emb, item_emb, image_feat, text_feat

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        
        user_emb, item_emb, img_emb, txt_emb = self.forward()
        
        u_e = user_emb[users]
        pos_e = item_emb[pos_items]
        neg_e = item_emb[neg_items]
        
        pos_scores = torch.sum(u_e * pos_e, dim=1)
        neg_scores = torch.sum(u_e * neg_e, dim=1)
        
        rec_loss = torch.mean(F.softplus(neg_scores - pos_scores))
        
        modal_loss = 0.0
        if img_emb is not None and txt_emb is not None:
            i_emb = F.normalize(img_emb[pos_items], dim=1)
            t_emb = F.normalize(txt_emb[pos_items], dim=1)
            modal_loss = 1 - F.cosine_similarity(i_emb, t_emb, dim=1).mean()
        
        return rec_loss + 0.1 * modal_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_emb, item_emb, _, _ = self.forward()
        score = torch.matmul(user_emb[user], item_emb.transpose(0, 1))
        return score