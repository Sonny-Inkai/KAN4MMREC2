# coding: utf-8

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss

class MMGAT(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MMGAT, self).__init__(config, dataset)
        
        # Hyperparameters
        self.embedding_dim = 64
        self.feat_embed_dim = 64
        self.num_heads = 4
        self.dropout = 0.3
        self.n_ui_layers = 2
        self.n_mm_layers = 2
        self.reg_weight = 1e-4
        self.temperature = 0.2
        self.knn_k = 10
        self.mirror_grad_weight = 0.2
        self.device = config['device']
        
        # Basic initialization
        self.n_nodes = self.n_users + self.n_items
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        # Flash attention components
        self.q_proj = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.k_proj = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.v_proj = nn.Linear(self.embedding_dim, self.embedding_dim)
        
        # Modality-specific components
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_transform = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)
            self.image_attn = nn.MultiheadAttention(self.feat_embed_dim, self.num_heads, dropout=self.dropout, batch_first=True)
        
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_transform = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)
            self.text_attn = nn.MultiheadAttention(self.feat_embed_dim, self.num_heads, dropout=self.dropout, batch_first=True)

        # Initialize adjacency matrices
        self.norm_adj = self.get_norm_adj_mat().to(self.device)
        self.mm_adj = self.init_mm_adj()
        
        # Projection layers
        self.modal_fusion = nn.Linear(self.feat_embed_dim * 2, self.embedding_dim)
        self.predictor = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.LeakyReLU(),
            nn.Linear(self.embedding_dim, self.embedding_dim)
        )

        self.inter_mat = dataset.inter_matrix(form='coo').astype(np.float32)

    def init_mm_adj(self):
        mm_adj = None
        if self.v_feat is not None:
            v_adj = self.build_knn_graph(self.image_embedding.weight)
            mm_adj = v_adj
        
        if self.t_feat is not None:
            t_adj = self.build_knn_graph(self.text_embedding.weight)
            if mm_adj is None:
                mm_adj = t_adj
            else:
                mm_adj = 0.5 * (mm_adj + t_adj)
        
        return mm_adj.to(self.device) if mm_adj is not None else None

    def build_knn_graph(self, embeddings):
        sim = F.normalize(embeddings, p=2, dim=1) @ F.normalize(embeddings, p=2, dim=1).t()
        topk_values, topk_indices = torch.topk(sim, k=self.knn_k, dim=1)
        mask = torch.zeros_like(sim).scatter_(1, topk_indices, 1.0)
        adj = mask * sim
        
        # Symmetric normalization
        degree = torch.sum(adj, dim=1)
        degree_inv_sqrt = torch.pow(degree + 1e-7, -0.5)
        degree_matrix_inv_sqrt = torch.diag(degree_inv_sqrt)
        norm_adj = degree_matrix_inv_sqrt @ adj @ degree_matrix_inv_sqrt
        
        return norm_adj

    def get_norm_adj_mat(self):
        adj_mat = sp.dok_matrix((self.n_nodes, self.n_nodes), dtype=np.float32)
        
        adj_mat[:self.n_users, self.n_users:] = self.inter_mat
        adj_mat[self.n_users:, :self.n_users] = self.inter_mat.T
        
        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum + 1e-7, -0.5).flatten()
        d_mat = sp.diags(d_inv)
        norm_adj = d_mat.dot(adj_mat).dot(d_mat)
        
        return self._convert_sp_mat_to_tensor(norm_adj)

    def _convert_sp_mat_to_tensor(self, matrix):
        matrix = matrix.tocoo()
        indices = torch.LongTensor([matrix.row, matrix.col])
        values = torch.FloatTensor(matrix.data)
        shape = torch.Size(matrix.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def flash_attention(self, query, key, value):
        # Scaled dot-product attention with flash attention optimization
        attn_weights = (query @ key.transpose(-2, -1)) / np.sqrt(query.size(-1))
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        return attn_weights @ value

    def forward(self):
        # Process multimodal features
        modal_emb = None
        if self.v_feat is not None:
            img_emb = self.image_transform(self.image_embedding.weight)
            img_emb, _ = self.image_attn(img_emb, img_emb, img_emb)
            modal_emb = img_emb
        
        if self.t_feat is not None:
            txt_emb = self.text_transform(self.text_embedding.weight)
            txt_emb, _ = self.text_attn(txt_emb, txt_emb, txt_emb)
            modal_emb = txt_emb if modal_emb is None else torch.cat([modal_emb, txt_emb], dim=-1)

        if modal_emb is not None:
            modal_emb = self.modal_fusion(modal_emb)
            
        # Graph convolution on user-item graph
        ego_embeddings = torch.cat([self.user_embedding.weight, self.item_id_embedding.weight], dim=0)
        all_embeddings = [ego_embeddings]
        
        for _ in range(self.n_ui_layers):
            ego_embeddings = torch.sparse.mm(self.norm_adj, ego_embeddings)
            all_embeddings.append(ego_embeddings)
            
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items])
        
        # Combine with modal embeddings if available
        if modal_emb is not None:
            i_g_embeddings = i_g_embeddings + modal_emb
            
        return u_g_embeddings, i_g_embeddings

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        # First forward pass
        u_embeddings, i_embeddings = self.forward()
        
        user_e = u_embeddings[users]
        pos_e = i_embeddings[pos_items]
        neg_e = i_embeddings[neg_items]
        
        # BPR loss
        pos_scores = torch.sum(user_e * pos_e, dim=1)
        neg_scores = torch.sum(user_e * neg_e, dim=1)
        bpr_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))

        # Contrastive loss
        pos_scores_norm = F.normalize(pos_scores.unsqueeze(-1), dim=1)
        neg_scores_norm = F.normalize(neg_scores.unsqueeze(-1), dim=1)
        contrastive_loss = -torch.log(
            torch.exp(pos_scores_norm/self.temperature) / 
            (torch.exp(pos_scores_norm/self.temperature) + torch.exp(neg_scores_norm/self.temperature))
        ).mean()

        # Mirror gradient regularization
        mirror_reg = torch.mean(torch.abs(user_e.grad if user_e.grad is not None else 0))
        
        # L2 regularization
        l2_loss = self.reg_weight * (
            torch.norm(user_e) +
            torch.norm(pos_e) +
            torch.norm(neg_e)
        )

        loss = bpr_loss + 0.1 * contrastive_loss + self.mirror_grad_weight * mirror_reg + l2_loss
        
        return loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        
        user_e, item_e = self.forward()
        user_e = user_e[user]
        
        scores = torch.matmul(user_e, item_e.transpose(0, 1))
        return scores