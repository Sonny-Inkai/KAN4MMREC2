# coding: utf-8
import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from common.abstract_recommender import GeneralRecommender

class DisentangledEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_factors=4, dropout=0.1):
        super().__init__()
        self.n_factors = n_factors
        self.factor_dim = output_dim // n_factors
        
        # Input transformation
        self.input_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Factor-specific networks
        self.factor_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, self.factor_dim),
                nn.LayerNorm(self.factor_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ) for _ in range(n_factors)
        ])
        
        # Factor attention
        self.factor_attention = nn.Sequential(
            nn.Linear(hidden_dim, n_factors),
            nn.Softmax(dim=1)
        )
        
        # Predictors for contrastive learning
        self.predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.factor_dim, self.factor_dim),
                nn.GELU(),
                nn.Linear(self.factor_dim, self.factor_dim)
            ) for _ in range(n_factors)
        ])
        
    def forward(self, x):
        h = self.input_net(x)
        
        # Generate factor-specific representations
        factor_outputs = []
        factor_preds = []
        
        for i in range(self.n_factors):
            factor = self.factor_nets[i](h)
            pred = self.predictors[i](factor.detach())
            factor_outputs.append(factor)
            factor_preds.append(pred)
            
        # Compute attention weights
        attention = self.factor_attention(h)
        
        # Combine factors with attention
        factors = torch.stack(factor_outputs, dim=1)
        weighted_factors = torch.sum(factors * attention.unsqueeze(-1), dim=1)
        
        return weighted_factors, torch.cat(factor_outputs, dim=1), torch.cat(factor_preds, dim=1)

class IntentionAwareAttention(nn.Module):
    def __init__(self, dim, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)
        
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.Sigmoid()
        )
        
    def forward(self, q, k, v):
        B, L, D = q.shape
        
        # Multi-head attention
        q = self.q_proj(q).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(k).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(v).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, L, D)
        out = self.o_proj(out)
        
        # Gating mechanism
        gate = self.gate(torch.cat([q.mean(1).squeeze(1), out.mean(1)], dim=-1))
        out = gate.unsqueeze(1) * out
        
        return out

class MMGAT(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MMGAT, self).__init__(config, dataset)
        
        self.embedding_dim = config["embedding_size"]
        self.feat_embed_dim = config["feat_embed_dim"]
        self.hidden_dim = self.feat_embed_dim * 2
        self.n_layers = config["n_mm_layers"]
        self.reg_weight = config["reg_weight"]
        self.dropout = config["dropout"]
        self.knn_k = config["knn_k"]
        
        # Core embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_normal_(self.user_embedding.weight, gain=0.1)
        nn.init.xavier_normal_(self.item_embedding.weight, gain=0.1)
        
        # Disentangled modal encoders
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_encoder = DisentangledEncoder(
                self.v_feat.shape[1], self.hidden_dim, self.feat_embed_dim
            )
            
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_encoder = DisentangledEncoder(
                self.t_feat.shape[1], self.hidden_dim, self.feat_embed_dim
            )
            
        # Cross-modal attention
        self.modal_attention = IntentionAwareAttention(self.feat_embed_dim)
        
        # Modal fusion with residual
        self.fusion = nn.Sequential(
            nn.Linear(self.feat_embed_dim * 2, self.feat_embed_dim),
            nn.LayerNorm(self.feat_embed_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.feat_embed_dim, self.feat_embed_dim),
            nn.LayerNorm(self.feat_embed_dim)
        )
        
        # Load data
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.norm_adj = self.get_norm_adj_mat().to(self.device)
        self.modal_adj = None
        self.build_modal_graph()
        
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
        
        indices = np.vstack((L.row, L.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(L.data)
        return torch.sparse_coo_tensor(i, v, L.shape)
        
    def build_modal_graph(self):
        if self.v_feat is not None:
            v_feat = F.normalize(self.v_feat, p=2, dim=1)
            v_sim = torch.mm(v_feat, v_feat.t())
            self.modal_adj = self.compute_graph(v_sim)
            
        if self.t_feat is not None:
            t_feat = F.normalize(self.t_feat, p=2, dim=1)
            t_sim = torch.mm(t_feat, t_feat.t())
            if self.modal_adj is None:
                self.modal_adj = self.compute_graph(t_sim)
            else:
                t_adj = self.compute_graph(t_sim)
                self.modal_adj = 0.5 * (self.modal_adj + t_adj)

    def compute_graph(self, sim_matrix):
        values, indices = sim_matrix.topk(k=self.knn_k, dim=1)
        rows = torch.arange(sim_matrix.size(0)).view(-1, 1).expand_as(indices)
        adj = torch.zeros_like(sim_matrix)
        adj[rows.reshape(-1), indices.reshape(-1)] = values.reshape(-1)
        
        adj = (adj + adj.t()) / 2
        deg = torch.sum(adj, dim=1)
        deg_inv_sqrt = torch.pow(deg + 1e-12, -0.5)
        norm_adj = deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)
        return norm_adj.to(self.device)

    def forward(self):
        # Process modalities
        img_feat = img_full = img_pred = txt_feat = txt_full = txt_pred = None
        
        if self.v_feat is not None:
            img_feat, img_full, img_pred = self.image_encoder(self.image_embedding.weight)
            for _ in range(self.n_layers):
                img_feat = torch.mm(self.modal_adj, img_feat)
                
        if self.t_feat is not None:
            txt_feat, txt_full, txt_pred = self.text_encoder(self.text_embedding.weight)
            for _ in range(self.n_layers):
                txt_feat = torch.mm(self.modal_adj, txt_feat)
        
        # Cross-modal attention and fusion
        if img_feat is not None and txt_feat is not None:
            img_feat = self.modal_attention(img_feat.unsqueeze(0), txt_feat.unsqueeze(0), txt_feat.unsqueeze(0)).squeeze(0)
            txt_feat = self.modal_attention(txt_feat.unsqueeze(0), img_feat.unsqueeze(0), img_feat.unsqueeze(0)).squeeze(0)
            modal_feat = self.fusion(torch.cat([img_feat, txt_feat], dim=1))
        else:
            modal_feat = img_feat if img_feat is not None else txt_feat
            
        # Graph convolution
        ego_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight])
        all_embeddings = [ego_embeddings]
        
        for _ in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.norm_adj, ego_embeddings)
            all_embeddings.append(ego_embeddings)
            
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1)
        
        user_emb, item_emb = torch.split(all_embeddings, [self.n_users, self.n_items])
        
        if modal_feat is not None:
            item_emb = item_emb + modal_feat
            
        return (
            user_emb, item_emb,
            (img_feat, img_full, img_pred),
            (txt_feat, txt_full, txt_pred)
        )

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        
        user_emb, item_emb, img_features, txt_features = self.forward()
        
        u_e = user_emb[users]
        pos_e = item_emb[pos_items]
        neg_e = item_emb[neg_items]
        
        # BPR loss
        pos_scores = torch.sum(u_e * pos_e, dim=1)
        neg_scores = torch.sum(u_e * neg_e, dim=1)
        bpr_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        
        # Enhanced contrastive loss
        modal_loss = 0.0
        if img_features[0] is not None and txt_features[0] is not None:
            img_feat, img_full, img_pred = [x[pos_items] for x in img_features]
            txt_feat, txt_full, txt_pred = [x[pos_items] for x in txt_features]
            
            # Factor-level contrastive loss
            modal_loss = -(
                torch.mean(F.cosine_similarity(F.normalize(img_full, dim=1), F.normalize(txt_pred, dim=1))) +
                torch.mean(F.cosine_similarity(F.normalize(txt_full, dim=1), F.normalize(img_pred, dim=1)))
            ) / 2
            
            # Feature-level alignment
            feat_loss = -torch.mean(F.cosine_similarity(F.normalize(img_feat, dim=1), F.normalize(txt_feat, dim=1)))
            modal_loss = modal_loss + 0.5 * feat_loss
            
        # L2 regularization
        reg_loss = self.reg_weight * (
            torch.norm(u_e) +
            torch.norm(pos_e) +
            torch.norm(neg_e)
        )
        
        return bpr_loss + 0.2 * modal_loss + reg_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_emb, item_emb, _, _ = self.forward()
        scores = torch.matmul(user_emb[user], item_emb.transpose(0, 1))
        return scores