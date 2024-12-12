# coding: utf-8
import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from common.abstract_recommender import GeneralRecommender

class MultimodalAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.temperature = 0.07
        self.cross_attention = nn.MultiheadAttention(dim, 4, dropout=0.1, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.fuse = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )
        
    def forward(self, x1, x2):
        attn_output, _ = self.cross_attention(x1, x2, x2)
        fused = self.fuse(torch.cat([x1, attn_output], dim=-1))
        return self.norm(fused + x1)

class DynamicGraphConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gating = nn.Sequential(
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x, adj):
        h = torch.sparse.mm(adj, x)
        gate = self.gating(h)
        return gate * h + (1 - gate) * x

class MMGAT(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MMGAT, self).__init__(config, dataset)
        
        self.embedding_dim = config["embedding_size"]
        self.feat_embed_dim = config["feat_embed_dim"]
        self.n_layers = config["n_mm_layers"]
        self.reg_weight = config["reg_weight"]
        self.n_nodes = self.n_users + self.n_items
        self.temperature = 0.07
        
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_normal_(self.user_embedding.weight)
        nn.init.xavier_normal_(self.item_embedding.weight)
        
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_proj = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)
            
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_proj = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)
        
        self.modal_attention = MultimodalAttention(self.feat_embed_dim)
        self.graph_layers = nn.ModuleList([
            DynamicGraphConv(self.embedding_dim) for _ in range(self.n_layers)
        ])
        
        self.predictor = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.GELU(),
            nn.Linear(self.embedding_dim, self.embedding_dim)
        )
        
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.norm_adj = self.get_norm_adj_mat().to(self.device)
        self.to(self.device)

    def get_norm_adj_mat(self):
        A = sp.dok_matrix((self.n_nodes, self.n_nodes), dtype=np.float32)
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
        return torch.sparse_coo_tensor(indices, values, L.shape).to(self.device)

    def forward(self):
        if self.v_feat is not None and self.t_feat is not None:
            image_feat = self.image_proj(self.image_embedding.weight)
            text_feat = self.text_proj(self.text_embedding.weight)
            
            img_enhanced = self.modal_attention(image_feat, text_feat)
            txt_enhanced = self.modal_attention(text_feat, image_feat)
            modal_feat = (img_enhanced + txt_enhanced) / 2
        else:
            img_enhanced = self.image_proj(self.image_embedding.weight) if self.v_feat is not None else None
            txt_enhanced = self.text_proj(self.text_embedding.weight) if self.t_feat is not None else None
            modal_feat = img_enhanced if img_enhanced is not None else txt_enhanced

        ego_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        all_embeddings = [ego_embeddings]
        
        x = ego_embeddings
        for layer in self.graph_layers:
            x = layer(x, self.norm_adj)
            all_embeddings.append(x)
        
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1)
        
        user_embeddings, item_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items])
        
        if modal_feat is not None:
            item_embeddings = item_embeddings + self.predictor(modal_feat)
        
        return user_embeddings, item_embeddings, img_enhanced, txt_enhanced

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        
        user_embeddings, item_embeddings, img_emb, txt_emb = self.forward()
        
        u_embeddings = user_embeddings[users]
        pos_embeddings = item_embeddings[pos_items]
        neg_embeddings = item_embeddings[neg_items]
        
        pos_scores = torch.sum(u_embeddings * pos_embeddings, dim=1)
        neg_scores = torch.sum(u_embeddings * neg_embeddings, dim=1)
        bpr_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        
        contrastive_loss = 0.0
        if img_emb is not None and txt_emb is not None:
            img_feat = F.normalize(img_emb[pos_items], dim=1)
            txt_feat = F.normalize(txt_emb[pos_items], dim=1)
            pos_sim = F.cosine_similarity(img_feat, txt_feat)
            contrastive_loss = -torch.mean(F.logsigmoid(pos_sim / self.temperature))
        
        reg_loss = self.reg_weight * (
            torch.norm(u_embeddings) +
            torch.norm(pos_embeddings) +
            torch.norm(neg_embeddings)
        )
        
        return bpr_loss + 0.2 * contrastive_loss + reg_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_embeddings, item_embeddings, _, _ = self.forward()
        scores = torch.matmul(user_embeddings[user], item_embeddings.transpose(0, 1))
        return scores