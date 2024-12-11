# coding: utf-8
import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from common.abstract_recommender import GeneralRecommender

class LightAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        attn = F.softmax(self.proj(x), dim=-1)
        return self.norm(x + torch.matmul(attn, x))

class MMGAT(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MMGAT, self).__init__(config, dataset)
        
        self.embedding_dim = config["embedding_size"]
        self.feat_embed_dim = config["feat_embed_dim"]
        self.n_layers = config["n_mm_layers"]
        self.reg_weight = config["reg_weight"]
        self.cl_weight = config["cl_weight"]
        self.dropout = config["dropout"]
        self.lambda_coeff = config["lambda_coeff"]
        self.temperature = 0.2
        
        # Basic embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_normal_(self.user_embedding.weight)
        nn.init.xavier_normal_(self.item_id_embedding.weight)
        
        # Modal processing
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_transform = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)
            self.image_attn = LightAttention(self.feat_embed_dim)
            
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_transform = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)
            self.text_attn = LightAttention(self.feat_embed_dim)
        
        self.fusion = nn.Sequential(
            nn.Linear(self.feat_embed_dim * 2, self.feat_embed_dim),
            nn.LayerNorm(self.feat_embed_dim),
            nn.GELU()
        )
        
        # Graph components
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.norm_adj = self.get_norm_adj_mat().to(self.device)
        self.to(self.device)

    def get_norm_adj_mat(self):
        A = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        for key, value in data_dict.items():
            A[key] = value
        
        sumArr = (A > 0).sum(axis=1)
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        L = sp.coo_matrix(L)
        indices = torch.LongTensor([L.row, L.col])
        values = torch.FloatTensor(L.data)
        shape = torch.Size(L.shape)
        return torch.sparse_coo_tensor(indices, values, shape)

    def forward(self):
        # Process modalities with bootstrap
        if self.v_feat is not None:
            img_feat = F.dropout(self.image_transform(self.image_embedding.weight), p=self.dropout, training=self.training)
            img_feat = self.image_attn(img_feat)
            
        if self.t_feat is not None:
            txt_feat = F.dropout(self.text_transform(self.text_embedding.weight), p=self.dropout, training=self.training)
            txt_feat = self.text_attn(txt_feat)
        
        # Modal fusion
        if self.v_feat is not None and self.t_feat is not None:
            modal_feat = self.fusion(torch.cat([img_feat, txt_feat], dim=1))
        else:
            modal_feat = img_feat if self.v_feat is not None else txt_feat
        
        # Graph convolution
        ego_embeddings = torch.cat([self.user_embedding.weight, self.item_id_embedding.weight], dim=0)
        all_embeddings = [ego_embeddings]
        
        for _ in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.norm_adj, ego_embeddings)
            all_embeddings.append(ego_embeddings)
            
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1)
        
        user_embeddings, item_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items])
        item_embeddings = item_embeddings + modal_feat
        
        return user_embeddings, item_embeddings, img_feat, txt_feat

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        
        user_embeddings, item_embeddings, img_feat, txt_feat = self.forward()
        
        u_embeddings = user_embeddings[users]
        pos_embeddings = item_embeddings[pos_items]
        neg_embeddings = item_embeddings[neg_items]
        
        # BPR Loss
        pos_scores = torch.sum(u_embeddings * pos_embeddings, dim=1)
        neg_scores = torch.sum(u_embeddings * neg_embeddings, dim=1)
        bpr_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        
        # Contrastive Loss
        cl_loss = 0.0
        if img_feat is not None and txt_feat is not None:
            img_pos = F.normalize(img_feat[pos_items], dim=1)
            txt_pos = F.normalize(txt_feat[pos_items], dim=1)
            cl_loss = -torch.mean(F.cosine_similarity(img_pos, txt_pos)) / self.temperature
            
        # Regularization
        reg_loss = self.reg_weight * (
            torch.norm(u_embeddings) +
            torch.norm(pos_embeddings) +
            torch.norm(neg_embeddings)
        )
        
        return bpr_loss + self.cl_weight * cl_loss + reg_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_embeddings, item_embeddings, _, _ = self.forward()
        scores = torch.matmul(user_embeddings[user], item_embeddings.transpose(0, 1))
        return scores