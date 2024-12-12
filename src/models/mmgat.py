# coding: utf-8
import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from common.abstract_recommender import GeneralRecommender
from torch.nn.functional import normalize

class MMGAT(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MMGAT, self).__init__(config, dataset)
        
        self.n_users = dataset.user_num
        self.n_items = dataset.item_num
        self.embedding_dim = config["embedding_size"]
        self.feat_embed_dim = config["feat_embed_dim"]
        self.n_layers = config["n_mm_layers"]
        self.reg_weight = config["reg_weight"]
        
        # User embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight, gain=1)
        
        # Item ID embeddings
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.item_embedding.weight, gain=1)
        
        # Modal towers
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_proj = nn.Sequential(
                nn.Linear(self.v_feat.shape[1], self.feat_embed_dim),
                nn.LayerNorm(self.feat_embed_dim),
                nn.GELU()
            )
            
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_proj = nn.Sequential(
                nn.Linear(self.t_feat.shape[1], self.feat_embed_dim),
                nn.LayerNorm(self.feat_embed_dim),
                nn.GELU()
            )
            
        # Multimodal fusion
        self.fusion = nn.Sequential(
            nn.Linear(self.feat_embed_dim * 2, self.feat_embed_dim),
            nn.LayerNorm(self.feat_embed_dim),
            nn.GELU()
        )
        
        # Graph structure
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.norm_adj = self.build_adj_matrix()
        self.to(self.device)

    def build_adj_matrix(self):
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.interaction_matrix.tolil()
        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()
        
        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum + 1e-9, -0.5).flatten()
        d_mat = sp.diags(d_inv)
        norm_adj = d_mat.dot(adj_mat).dot(d_mat)
        
        coo = norm_adj.tocoo()
        indices = np.vstack((coo.row, coo.col))
        i = torch.LongTensor(indices).to(self.device)
        v = torch.FloatTensor(coo.data).to(self.device)
        return torch.sparse_coo_tensor(i, v, coo.shape)

    def forward(self):
        # Process modal features
        image_feat = None
        text_feat = None
        
        if self.v_feat is not None:
            image_feat = normalize(self.image_proj(self.image_embedding.weight), p=2, dim=1)
            
        if self.t_feat is not None:
            text_feat = normalize(self.text_proj(self.text_embedding.weight), p=2, dim=1)
            
        # Modal fusion
        if image_feat is not None and text_feat is not None:
            modal_feat = self.fusion(torch.cat([image_feat, text_feat], dim=1))
        else:
            modal_feat = image_feat if image_feat is not None else text_feat
            
        modal_feat = normalize(modal_feat, p=2, dim=1)
            
        # Graph convolution
        ego_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        ego_embeddings = normalize(ego_embeddings, p=2, dim=1)
        all_embeddings = [ego_embeddings]
        
        for _ in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.norm_adj, ego_embeddings)
            ego_embeddings = normalize(ego_embeddings, p=2, dim=1)
            all_embeddings.append(ego_embeddings)
            
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        
        user_embeddings, item_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items])
        
        # Combine ID embeddings with modal features
        item_embeddings = item_embeddings + 0.1 * modal_feat
        item_embeddings = normalize(item_embeddings, p=2, dim=1)
        
        return user_embeddings, item_embeddings, image_feat, text_feat

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        
        # Get embeddings
        user_emb, item_emb, img_emb, txt_emb = self.forward()
        
        user_e = user_emb[users]
        pos_e = item_emb[pos_items]
        neg_e = item_emb[neg_items]
        
        # BPR loss with scaled cosine similarity
        pos_scores = torch.sum(user_e * pos_e, dim=1) * 0.5 + 0.5  # Scale to [0,1]
        neg_scores = torch.sum(user_e * neg_e, dim=1) * 0.5 + 0.5
        bpr_loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-6))
        
        # Modal contrastive loss
        modal_loss = 0.0
        if img_emb is not None and txt_emb is not None:
            pos_sim = torch.sum(img_emb[pos_items] * txt_emb[pos_items], dim=1) * 0.5 + 0.5
            modal_loss = -torch.mean(torch.log(pos_sim + 1e-6))
            
        # L2 regularization (scaled)
        reg_loss = self.reg_weight * (
            torch.norm(user_e, p=2) + 
            torch.norm(pos_e, p=2) + 
            torch.norm(neg_e, p=2)
        ) * 0.1
        
        return bpr_loss + 0.1 * modal_loss + reg_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_emb, item_emb, _, _ = self.forward()
        scores = torch.matmul(user_emb[user], item_emb.transpose(0, 1))
        return scores