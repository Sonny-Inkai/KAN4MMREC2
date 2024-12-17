import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from torch.nn.functional import cosine_similarity

from common.abstract_recommender import GeneralRecommender

class FlashAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.scale = dim ** -0.5

    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        attention = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attention = F.softmax(attention, dim=-1)
        return torch.matmul(attention, v)

class MMGAT(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MMGAT, self).__init__(config, dataset)
        
        self.embedding_dim = 64 
        self.feat_embed_dim = 64
        self.num_heads = 4
        self.dropout = 0.2
        self.n_layers = 2
        self.knn_k = 10
        self.alpha = 0.2
        self.lambda_coeff = 0.5
        self.reg_weight = 1e-4
        
        self.n_nodes = self.n_users + self.n_items
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim).to(self.device)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim).to(self.device)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)
        
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat).to(self.device)
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim).to(self.device)
            self.image_attn = FlashAttention(self.feat_embed_dim).to(self.device)
            
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat).to(self.device)
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim).to(self.device)
            self.text_attn = FlashAttention(self.feat_embed_dim).to(self.device)
            
        self.modal_fusion = nn.Linear(self.feat_embed_dim * 2, self.feat_embed_dim).to(self.device)
        self.modal_weight = nn.Parameter(torch.Tensor([0.5, 0.5])).to(self.device)
        self.softmax = nn.Softmax(dim=0)
        
        self.norm_adj = self.get_norm_adj_mat().to(self.device)
        self.mm_adj = None
        self.build_multimodal_graph()

    def get_norm_adj_mat(self):
        A = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        
        for key, value in data_dict.items():
            A[key] = value
            
        D = np.array(np.sum(A, axis=1)).squeeze() + 1e-7
        D = np.power(D, -0.5)
        D = sp.diags(D)
        L = D.dot(A).dot(D)
        
        L = sp.coo_matrix(L)
        indices = torch.LongTensor(np.array([L.row, L.col]))
        values = torch.FloatTensor(L.data)
        
        return torch.sparse_coo_tensor(indices, values, (self.n_nodes, self.n_nodes))

    def build_multimodal_graph(self):
        if self.v_feat is not None:
            image_feats = self.image_trs(self.image_embedding.weight)
            image_norm = F.normalize(image_feats, p=2, dim=1)
            image_sim = torch.mm(image_norm, image_norm.t())
            image_adj = self.build_knn_graph(image_sim)
            self.mm_adj = image_adj.to(self.device)
            
        if self.t_feat is not None:
            text_feats = self.text_trs(self.text_embedding.weight)
            text_norm = F.normalize(text_feats, p=2, dim=1)
            text_sim = torch.mm(text_norm, text_norm.t())
            text_adj = self.build_knn_graph(text_sim)
            if self.mm_adj is None:
                self.mm_adj = text_adj.to(self.device)
            else:
                self.mm_adj = (self.mm_adj + text_adj.to(self.device))
            
        if self.mm_adj is not None:
            self.mm_adj = F.normalize(self.mm_adj, p=1, dim=1)

    def build_knn_graph(self, sim_matrix):
        values, indices = sim_matrix.topk(k=self.knn_k, dim=1)
        row_indices = torch.arange(sim_matrix.size(0), device=sim_matrix.device).view(-1, 1).expand_as(indices)
        adj = torch.zeros_like(sim_matrix)
        adj[row_indices.flatten(), indices.flatten()] = values.flatten()
        return adj

    def forward(self):
        mm_embeddings = None
        
        if self.v_feat is not None:
            image_feats = self.image_trs(self.image_embedding.weight)
            image_feats = self.image_attn(image_feats.unsqueeze(0)).squeeze(0)
            mm_embeddings = image_feats
            
        if self.t_feat is not None:
            text_feats = self.text_trs(self.text_embedding.weight)
            text_feats = self.text_attn(text_feats.unsqueeze(0)).squeeze(0)
            if mm_embeddings is None:
                mm_embeddings = text_feats
            else:
                mm_embeddings = self.modal_fusion(torch.cat([mm_embeddings, text_feats], dim=1))

        ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
        all_embeddings = [ego_embeddings]
        
        for _ in range(self.n_layers):
            side_embeddings = torch.sparse.mm(self.norm_adj, ego_embeddings)
            ego_embeddings = side_embeddings
            
            if mm_embeddings is not None:
                item_embeddings = ego_embeddings[self.n_users:]
                mm_enhanced = torch.matmul(self.mm_adj, mm_embeddings)
                item_embeddings = item_embeddings + self.lambda_coeff * mm_enhanced
                ego_embeddings = torch.cat([ego_embeddings[:self.n_users], item_embeddings], dim=0)
                
            all_embeddings.append(ego_embeddings)
            
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        return u_g_embeddings, i_g_embeddings

    def calculate_loss(self, interaction):
        users = interaction[0].to(self.device)
        pos_items = interaction[1].to(self.device)
        neg_items = interaction[2].to(self.device)

        u_embeddings, i_embeddings = self.forward()
        
        user_embeddings = u_embeddings[users]
        pos_embeddings = i_embeddings[pos_items]
        neg_embeddings = i_embeddings[neg_items]

        mf_loss = self.mirror_gradient_loss(users, pos_items, neg_items)
        
        modal_loss = 0.0
        if self.v_feat is not None and self.t_feat is not None:
            image_feats = self.image_trs(self.image_embedding.weight)
            text_feats = self.text_trs(self.text_embedding.weight)
            
            pos_image_feats = image_feats[pos_items]
            pos_text_feats = text_feats[pos_items]
            
            modal_sim = cosine_similarity(pos_image_feats, pos_text_feats, dim=1)
            modal_loss = -torch.mean(torch.log(torch.sigmoid(modal_sim)))

        reg_loss = self.reg_weight * (
            torch.norm(user_embeddings) +
            torch.norm(pos_embeddings) +
            torch.norm(neg_embeddings)
        )

        return mf_loss + 0.1 * modal_loss + reg_loss

    def mirror_gradient_loss(self, users, pos_items, neg_items, alpha=0.2):
        u_embeddings = self.user_embedding(users)
        pos_embeddings = self.item_id_embedding(pos_items)
        neg_embeddings = self.item_id_embedding(neg_items)
        
        pos_scores = torch.sum(u_embeddings * pos_embeddings, dim=1)
        neg_scores = torch.sum(u_embeddings * neg_embeddings, dim=1)
        
        base_loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores)))
        
        with torch.no_grad():
            u_grad = torch.autograd.grad(base_loss, u_embeddings, retain_graph=True)[0]
            mirror_direction = -alpha * u_grad
            
        mirror_embeddings = u_embeddings + mirror_direction
        mirror_pos_scores = torch.sum(mirror_embeddings * pos_embeddings, dim=1)
        mirror_neg_scores = torch.sum(mirror_embeddings * neg_embeddings, dim=1)
        
        mirror_loss = -torch.mean(torch.log(torch.sigmoid(mirror_pos_scores - mirror_neg_scores)))
        
        return base_loss + mirror_loss

    def full_sort_predict(self, interaction):
        user = interaction[0].to(self.device)
        
        u_embeddings, i_embeddings = self.forward()
        u_embeddings = u_embeddings[user]
        
        scores = torch.matmul(u_embeddings, i_embeddings.transpose(0, 1))
        return scores