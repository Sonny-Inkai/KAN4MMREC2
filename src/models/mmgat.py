# coding: utf-8
import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from common.abstract_recommender import GeneralRecommender

class ModalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)

class CrossModalFusion(nn.Module):
    def __init__(self, dim, n_heads=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x1, x2):
        out1, _ = self.attention(x1, x2, x2)
        out2, _ = self.attention(x2, x1, x1)
        return self.norm(out1 + x1), self.norm(out2 + x2)

class MMGAT(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MMGAT, self).__init__(config, dataset)
        
        self.n_users = dataset.user_num
        self.n_items = dataset.item_num
        self.embedding_dim = config["embedding_size"]
        self.feat_embed_dim = config["feat_embed_dim"]
        self.n_layers = config["n_mm_layers"]
        self.reg_weight = config["reg_weight"]
        self.dropout = config["dropout"]
        self.temperature = 0.2
        
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_encoder = ModalEncoder(self.v_feat.shape[1], self.feat_embed_dim, self.dropout)
            
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_encoder = ModalEncoder(self.t_feat.shape[1], self.feat_embed_dim, self.dropout)
            
        self.modal_fusion = CrossModalFusion(self.feat_embed_dim)
        
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.norm_adj = self.get_norm_adj_mat()
        self.to(self.device)

    def get_norm_adj_mat(self):
        A = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        for key, value in data_dict.items():
            A[key] = value
        
        rowsum = np.array(A.sum(axis=1))
        d_inv = np.power(rowsum + 1e-7, -0.5).flatten()
        D = sp.diags(d_inv)
        L = D * A * D
        L = sp.coo_matrix(L)
        
        indices = np.vstack((L.row, L.col))
        i = torch.LongTensor(indices).to(self.device)
        v = torch.FloatTensor(L.data).to(self.device)
        shape = torch.Size(L.shape)
        return torch.sparse_coo_tensor(i, v, shape)

    def forward(self):
        img_feat = None
        txt_feat = None
        
        if self.v_feat is not None:
            img_feat = self.image_encoder(self.image_embedding.weight)
            
        if self.t_feat is not None:
            txt_feat = self.text_encoder(self.text_embedding.weight)
            
        if img_feat is not None and txt_feat is not None:
            img_feat, txt_feat = self.modal_fusion(img_feat.unsqueeze(0), txt_feat.unsqueeze(0))
            modal_feat = (img_feat.squeeze(0) + txt_feat.squeeze(0)) / 2
        else:
            modal_feat = img_feat if img_feat is not None else txt_feat
            
        ego_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        all_embeddings = [ego_embeddings]
        
        for _ in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.norm_adj, ego_embeddings)
            all_embeddings.append(ego_embeddings)
            
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1)
        
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items])
        i_g_embeddings = i_g_embeddings + modal_feat
        
        return u_g_embeddings, i_g_embeddings, img_feat, txt_feat

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        
        u_embeddings, i_embeddings, img_feat, txt_feat = self.forward()
        
        u_e = u_embeddings[users]
        pos_e = i_embeddings[pos_items]
        neg_e = i_embeddings[neg_items]
        
        # BPR loss
        pos_scores = torch.sum(u_e * pos_e, dim=1)
        neg_scores = torch.sum(u_e * neg_e, dim=1)
        bpr_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        
        # Modal alignment loss
        modal_loss = 0.0
        if img_feat is not None and txt_feat is not None:
            pos_sim = torch.sum(F.normalize(img_feat, dim=1) * F.normalize(txt_feat, dim=1), dim=1)
            modal_loss = -torch.mean(F.logsigmoid(pos_sim / self.temperature))
        
        # Feature matching loss
        feat_loss = 0.0
        if img_feat is not None:
            feat_loss += F.mse_loss(F.normalize(i_embeddings, dim=1), F.normalize(img_feat, dim=1))
        if txt_feat is not None:
            feat_loss += F.mse_loss(F.normalize(i_embeddings, dim=1), F.normalize(txt_feat, dim=1))
        
        # L2 regularization
        reg_loss = self.reg_weight * (
            torch.norm(u_e) + 
            torch.norm(pos_e) + 
            torch.norm(neg_e)
        )
        
        return bpr_loss + 0.1 * modal_loss + 0.1 * feat_loss + reg_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        u_embeddings, i_embeddings, _, _ = self.forward()
        scores = torch.matmul(u_embeddings[user], i_embeddings.transpose(0, 1))
        return scores