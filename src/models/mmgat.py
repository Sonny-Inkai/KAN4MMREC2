# coding: utf-8
import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from common.abstract_recommender import GeneralRecommender

class AttentionLayer(nn.Module):
    def __init__(self, in_dim, dropout=0.2):
        super().__init__()
        self.att_w = nn.Parameter(torch.zeros(size=(1, in_dim)))
        nn.init.xavier_uniform_(self.att_w.data)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, query, key, value, mask=None):
        energy = self.leaky_relu(torch.matmul(query, key.transpose(-2, -1)))
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e9)
        attention = self.dropout(self.softmax(energy))
        return torch.matmul(attention, value)

class ModalFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attention = AttentionLayer(dim)
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, 2),
            nn.Softmax(dim=-1)
        )
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x1, x2):
        cross_x1 = self.attention(x1, x2, x2)
        cross_x2 = self.attention(x2, x1, x1)
        weights = self.gate(torch.cat([cross_x1, cross_x2], dim=-1))
        fused = weights[:, 0:1] * cross_x1 + weights[:, 1:2] * cross_x2
        return self.norm(fused)

class MMGAT(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MMGAT, self).__init__(config, dataset)
        
        self.embedding_dim = config["embedding_size"]
        self.feat_embed_dim = config["feat_embed_dim"]
        self.n_layers = config["n_mm_layers"]
        self.reg_weight = config["reg_weight"]
        self.dropout = config["dropout"]
        self.temperature = 0.07
        
        # Embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        
        # Modal encoders
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_encoder = nn.Sequential(
                nn.Linear(self.v_feat.shape[1], self.feat_embed_dim),
                nn.LayerNorm(self.feat_embed_dim),
                nn.GELU(),
                nn.Dropout(self.dropout)
            )
            
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_encoder = nn.Sequential(
                nn.Linear(self.t_feat.shape[1], self.feat_embed_dim),
                nn.LayerNorm(self.feat_embed_dim),
                nn.GELU(),
                nn.Dropout(self.dropout)
            )
        
        # Cross-modal attention
        self.modal_fusion = ModalFusion(self.feat_embed_dim)
        
        # Graph attention layers
        self.attention_layers = nn.ModuleList([
            AttentionLayer(self.embedding_dim) 
            for _ in range(self.n_layers)
        ])
        
        # Predictors for InfoNCE loss
        self.modal_predictor = nn.Sequential(
            nn.Linear(self.feat_embed_dim, self.feat_embed_dim),
            nn.GELU(),
            nn.Linear(self.feat_embed_dim, self.feat_embed_dim)
        )
        
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.build_graph()
        self.to(self.device)

    def build_graph(self):
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
        
        # Convert to PyTorch sparse tensor
        coo = norm_adj.tocoo()
        indices = np.vstack((coo.row, coo.col))
        i = torch.LongTensor(indices).to(self.device)
        v = torch.FloatTensor(coo.data).to(self.device)
        self.norm_adj = torch.sparse_coo_tensor(i, v, coo.shape)

    def forward(self):
        # Process modalities
        image_feat = None
        text_feat = None
        
        if self.v_feat is not None:
            image_feat = self.image_encoder(self.image_embedding.weight)
            
        if self.t_feat is not None:
            text_feat = self.text_encoder(self.text_embedding.weight)
        
        # Modal fusion with attention
        if image_feat is not None and text_feat is not None:
            modal_feat = self.modal_fusion(image_feat, text_feat)
        else:
            modal_feat = image_feat if image_feat is not None else text_feat
            
        # Graph attention with residual connections
        ego_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        all_embeddings = [ego_embeddings]
        
        for layer in self.attention_layers:
            neighbors = torch.sparse.mm(self.norm_adj, ego_embeddings)
            ego_embeddings = layer(ego_embeddings, neighbors, neighbors)
            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
            all_embeddings.append(norm_embeddings)
            
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1)
        user_emb, item_emb = torch.split(all_embeddings, [self.n_users, self.n_items])
        
        # Add modal information to item embeddings
        if modal_feat is not None:
            item_emb = item_emb + modal_feat
            
        return user_emb, item_emb, image_feat, text_feat

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        
        user_emb, item_emb, img_emb, txt_emb = self.forward()
        
        u_e = user_emb[users]
        pos_e = item_emb[pos_items]
        neg_e = item_emb[neg_items]
        
        # InfoNCE loss for recommendation
        pos_scores = torch.sum(u_e * pos_e, dim=1)
        neg_scores = torch.sum(u_e * neg_e, dim=1)
        rec_loss = -torch.mean(
            F.logsigmoid(pos_scores / self.temperature - neg_scores / self.temperature)
        )
        
        # Modal contrastive loss
        modal_loss = 0.0
        if img_emb is not None and txt_emb is not None:
            img_pred = self.modal_predictor(img_emb[pos_items])
            txt_pred = self.modal_predictor(txt_emb[pos_items])
            
            pos_sim = torch.sum(F.normalize(img_pred, dim=1) * F.normalize(txt_emb[pos_items].detach(), dim=1), dim=1)
            neg_sim = torch.sum(F.normalize(img_pred, dim=1) * F.normalize(txt_emb[neg_items].detach(), dim=1), dim=1)
            modal_loss_1 = -torch.mean(F.logsigmoid(pos_sim / self.temperature - neg_sim / self.temperature))
            
            pos_sim = torch.sum(F.normalize(txt_pred, dim=1) * F.normalize(img_emb[pos_items].detach(), dim=1), dim=1)
            neg_sim = torch.sum(F.normalize(txt_pred, dim=1) * F.normalize(img_emb[neg_items].detach(), dim=1), dim=1)
            modal_loss_2 = -torch.mean(F.logsigmoid(pos_sim / self.temperature - neg_sim / self.temperature))
            
            modal_loss = (modal_loss_1 + modal_loss_2) / 2
        
        # Multi-task regularization
        reg_loss = self.reg_weight * (
            torch.norm(u_e) + torch.norm(pos_e) + torch.norm(neg_e)
        )
        
        return rec_loss + 0.1 * modal_loss + reg_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_emb, item_emb, _, _ = self.forward()
        scores = torch.matmul(user_emb[user], item_emb.transpose(0, 1))
        return scores