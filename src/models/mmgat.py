# coding: utf-8
import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from common.abstract_recommender import GeneralRecommender

class CrossAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim ** 0.5
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x1, x2):
        q = self.query(x1)
        k = self.key(x2)
        v = self.value(x2)
        
        attention = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)
        
        out = torch.matmul(attention, v)
        return self.norm(out + x1)

class MultimodalEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.feat_embed_dim = config["feat_embed_dim"]
        
        self.image_transform = nn.Sequential(
            nn.Linear(config["v_feat_dim"], self.feat_embed_dim),
            nn.LayerNorm(self.feat_embed_dim),
            nn.GELU()
        )
        
        self.text_transform = nn.Sequential(
            nn.Linear(config["t_feat_dim"], self.feat_embed_dim),
            nn.LayerNorm(self.feat_embed_dim),
            nn.GELU()
        )
        
        self.cross_attention = CrossAttention(self.feat_embed_dim)
        
    def forward(self, image_feat, text_feat):
        image_emb = self.image_transform(image_feat)
        text_emb = self.text_transform(text_feat)
        
        img_enhanced = self.cross_attention(image_emb, text_emb)
        txt_enhanced = self.cross_attention(text_emb, image_emb)
        
        return img_enhanced, txt_enhanced

class MMGAT(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MMGAT, self).__init__(config, dataset)
        
        self.n_users = dataset.user_num
        self.n_items = dataset.item_num
        self.embedding_dim = config["embedding_size"]
        self.feat_embed_dim = config["feat_embed_dim"]
        self.n_layers = config["n_mm_layers"]
        self.temperature = 0.2
        self.reg_weight = config["reg_weight"]
        self.dropout = config["dropout"]
        
        # Initialize embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        
        config["v_feat_dim"] = self.v_feat.shape[1] if self.v_feat is not None else 0
        config["t_feat_dim"] = self.t_feat.shape[1] if self.t_feat is not None else 0
        
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
        
        # Multimodal processing
        self.modal_encoder = MultimodalEncoder(config)
        
        # Build interaction graph
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.norm_adj = self.get_norm_adj_mat().to(self.device)
        
        self.to(self.device)
        
    def get_norm_adj_mat(self):
        A = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        for key, item in data_dict.items():
            A[key] = item
        
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
        # Process multimodal information
        if self.v_feat is not None and self.t_feat is not None:
            img_enhanced, txt_enhanced = self.modal_encoder(
                self.image_embedding.weight,
                self.text_embedding.weight
            )
            modal_embedding = (img_enhanced + txt_enhanced) / 2
        else:
            modal_embedding = None
        
        # Process user-item graph
        ego_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        all_embeddings = [ego_embeddings]
        
        for _ in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.norm_adj, ego_embeddings)
            all_embeddings.append(ego_embeddings)
            
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1)
        
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items])
        
        # Combine with modal information
        if modal_embedding is not None:
            i_g_embeddings = i_g_embeddings + modal_embedding
            
        return u_g_embeddings, i_g_embeddings, img_enhanced, txt_enhanced
        
    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        
        u_embeddings, i_embeddings, img_enhanced, txt_enhanced = self.forward()
        
        u_embeddings = u_embeddings[users]
        pos_embeddings = i_embeddings[pos_items]
        neg_embeddings = i_embeddings[neg_items]
        
        # BPR Loss
        pos_scores = torch.sum(u_embeddings * pos_embeddings, dim=1)
        neg_scores = torch.sum(u_embeddings * neg_embeddings, dim=1)
        bpr_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        
        # Contrastive Loss for modalities
        modal_loss = 0.0
        if img_enhanced is not None and txt_enhanced is not None:
            i_emb = F.normalize(img_enhanced[pos_items], dim=1)
            t_emb = F.normalize(txt_enhanced[pos_items], dim=1)
            modal_loss = -torch.mean(
                F.cosine_similarity(i_emb, t_emb, dim=1) / self.temperature
            )
        
        # L2 Regularization
        reg_loss = self.reg_weight * (
            torch.norm(u_embeddings) +
            torch.norm(pos_embeddings) +
            torch.norm(neg_embeddings)
        )
        
        return bpr_loss + 0.1 * modal_loss + reg_loss
        
    def full_sort_predict(self, interaction):
        user = interaction[0]
        u_embeddings, i_embeddings, _, _ = self.forward()
        scores = torch.matmul(u_embeddings[user], i_embeddings.transpose(0, 1))
        return scores