import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from torch.nn.functional import cosine_similarity

from common.abstract_recommender import GeneralRecommender

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
            self.image_proj = nn.Linear(self.feat_embed_dim, self.feat_embed_dim).to(self.device)
            
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat).to(self.device)
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim).to(self.device)
            self.text_proj = nn.Linear(self.feat_embed_dim, self.feat_embed_dim).to(self.device)
            
        self.modal_fusion = nn.Linear(self.feat_embed_dim * 2, self.feat_embed_dim).to(self.device)
        self.modal_weight = nn.Parameter(torch.Tensor([0.5, 0.5])).to(self.device)
        
        self.norm_adj = self.get_norm_adj_mat().to(self.device)
        self.mm_adj = None
        self.build_multimodal_graph()

    def get_norm_adj_mat(self):
        A = sp.dok_matrix((self.n_nodes, self.n_nodes), dtype=np.float32)
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
            values, indices = image_sim.topk(k=self.knn_k, dim=1)
            row_indices = torch.arange(image_sim.size(0), device=self.device).view(-1, 1).expand_as(indices)
            image_adj = torch.zeros_like(image_sim)
            image_adj[row_indices.flatten(), indices.flatten()] = values.flatten()
            self.mm_adj = F.normalize(image_adj, p=1, dim=1)
            
        if self.t_feat is not None:
            text_feats = self.text_trs(self.text_embedding.weight)
            text_norm = F.normalize(text_feats, p=2, dim=1)
            text_sim = torch.mm(text_norm, text_norm.t())
            values, indices = text_sim.topk(k=self.knn_k, dim=1)
            row_indices = torch.arange(text_sim.size(0), device=self.device).view(-1, 1).expand_as(indices)
            text_adj = torch.zeros_like(text_sim)
            text_adj[row_indices.flatten(), indices.flatten()] = values.flatten()
            text_adj = F.normalize(text_adj, p=1, dim=1)
            
            if self.mm_adj is None:
                self.mm_adj = text_adj
            else:
                weights = F.softmax(self.modal_weight, dim=0)
                self.mm_adj = weights[0] * self.mm_adj + weights[1] * text_adj

    def forward(self):
        mm_embeddings = None
        
        if self.v_feat is not None:
            image_feats = self.image_trs(self.image_embedding.weight)
            image_feats = self.image_proj(image_feats)
            mm_embeddings = image_feats
            
        if self.t_feat is not None:
            text_feats = self.text_trs(self.text_embedding.weight)
            text_feats = self.text_proj(text_feats)
            if mm_embeddings is None:
                mm_embeddings = text_feats
            else:
                mm_embeddings = self.modal_fusion(torch.cat([mm_embeddings, text_feats], dim=1))

        ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
        all_embeddings = [ego_embeddings]
        
        for _ in range(self.n_layers):
            side_embeddings = torch.sparse.mm(self.norm_adj, ego_embeddings)
            
            if mm_embeddings is not None:
                item_embeddings = side_embeddings[self.n_users:]
                mm_enhanced = torch.matmul(self.mm_adj, mm_embeddings)
                item_embeddings = item_embeddings + self.lambda_coeff * mm_enhanced
                side_embeddings = torch.cat([side_embeddings[:self.n_users], item_embeddings], dim=0)
                
            ego_embeddings = side_embeddings
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
        
        # BPR loss with gradient direction adjustment
        pos_scores = torch.sum(user_embeddings * pos_embeddings, dim=1)
        neg_scores = torch.sum(user_embeddings * neg_embeddings, dim=1)
        
        mf_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        
        # Mirror gradient adjustment
        gradient_direction = torch.sign(pos_scores - neg_scores).detach()
        adjusted_pos_scores = pos_scores + self.alpha * gradient_direction
        adjusted_neg_scores = neg_scores - self.alpha * gradient_direction
        mirror_loss = -torch.mean(F.logsigmoid(adjusted_pos_scores - adjusted_neg_scores))
        
        # Modal contrastive loss
        modal_loss = 0.0
        if self.v_feat is not None and self.t_feat is not None:
            image_feats = self.image_trs(self.image_embedding.weight)
            text_feats = self.text_trs(self.text_embedding.weight)
            
            pos_image_feats = image_feats[pos_items]
            pos_text_feats = text_feats[pos_items]
            
            modal_sim = cosine_similarity(pos_image_feats, pos_text_feats, dim=1)
            modal_loss = -torch.mean(F.logsigmoid(modal_sim))

        reg_loss = self.reg_weight * (
            torch.norm(user_embeddings) +
            torch.norm(pos_embeddings) +
            torch.norm(neg_embeddings)
        )

        return mf_loss + mirror_loss + 0.1 * modal_loss + reg_loss

    def full_sort_predict(self, interaction):
        user = interaction[0].to(self.device)
        
        u_embeddings, i_embeddings = self.forward()
        u_embeddings = u_embeddings[user]
        
        scores = torch.matmul(u_embeddings, i_embeddings.transpose(0, 1))
        return scores