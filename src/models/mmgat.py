# coding: utf-8
import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from common.abstract_recommender import GeneralRecommender

class MMGAT(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MMGAT, self).__init__(config, dataset)
        
        self.embedding_dim = config["embedding_size"]
        self.feat_embed_dim = config["feat_embed_dim"]
        self.n_layers = config["n_mm_layers"]
        self.temperature = 0.2
        
        # Basic embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        
        # Modal projections
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_proj = nn.Sequential(
                nn.Linear(self.v_feat.shape[1], self.embedding_dim),
                nn.LayerNorm(self.embedding_dim),
                nn.GELU()
            )
            
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_proj = nn.Sequential(
                nn.Linear(self.t_feat.shape[1], self.embedding_dim),
                nn.LayerNorm(self.embedding_dim),
                nn.GELU()
            )
        
        # Attention weights
        self.modal_attention = nn.Linear(self.embedding_dim, 1)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim)
        )
        
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.norm_adj = self.build_graph()
        self.to(self.device)
    
    def build_graph(self):
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.interaction_matrix.tolil()
        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()
        
        # Normalize adjacency matrix
        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum + 1e-7, -0.5).flatten()
        d_mat = sp.diags(d_inv)
        norm_adj = d_mat.dot(adj_mat).dot(d_mat)
        
        # Convert to sparse tensor
        coo = norm_adj.tocoo()
        indices = np.vstack((coo.row, coo.col))
        i = torch.LongTensor(indices).to(self.device)
        v = torch.FloatTensor(coo.data).to(self.device)
        return torch.sparse_coo_tensor(i, v, coo.shape)

    def aggregate_modalities(self, modal1, modal2=None):
        if modal2 is None:
            return modal1
            
        attention = torch.softmax(
            torch.cat([
                self.modal_attention(modal1),
                self.modal_attention(modal2)
            ], dim=1),
            dim=1
        )
        
        return attention[:, 0:1] * modal1 + attention[:, 1:2] * modal2

    def forward(self):
        # Process modalities
        image_feat = self.image_proj(self.image_embedding.weight) if self.v_feat is not None else None
        text_feat = self.text_proj(self.text_embedding.weight) if self.t_feat is not None else None
        
        # Fuse modalities
        if image_feat is not None and text_feat is not None:
            modal_feat = self.fusion(torch.cat([image_feat, text_feat], dim=1))
        else:
            modal_feat = image_feat if image_feat is not None else text_feat
        
        # Graph convolution
        ego_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight + modal_feat], dim=0)
        all_embeddings = [ego_embeddings]
        
        for _ in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.norm_adj, ego_embeddings)
            all_embeddings.append(ego_embeddings)
        
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1)
        
        user_embeddings, item_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items])
        return user_embeddings, item_embeddings, image_feat, text_feat

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        
        user_emb, item_emb, img_emb, txt_emb = self.forward()
        
        # BPR Loss
        u_embeddings = user_emb[users]
        pos_embeddings = item_emb[pos_items]
        neg_embeddings = item_emb[neg_items]
        
        pos_scores = torch.sum(u_embeddings * pos_embeddings, dim=1)
        neg_scores = torch.sum(u_embeddings * neg_embeddings, dim=1)
        bpr_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        
        # Modal Contrastive Loss
        modal_loss = 0.0
        if img_emb is not None and txt_emb is not None:
            pos_sim = F.cosine_similarity(
                F.normalize(img_emb[pos_items], dim=1),
                F.normalize(txt_emb[pos_items], dim=1)
            )
            modal_loss = -torch.mean(F.logsigmoid(pos_sim / self.temperature))
        
        return bpr_loss + 0.1 * modal_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_embeddings, item_embeddings, _, _ = self.forward()
        scores = torch.matmul(user_embeddings[user], item_embeddings.transpose(0, 1))
        return scores