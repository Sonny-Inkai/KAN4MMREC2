import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np
import copy

import torch.nn.functional as F
from torch.nn.functional import cosine_similarity

from common.abstract_recommender import GeneralRecommender
from common.loss import EmbLoss


class ResidualTransformer(nn.Module):
    def __init__(self, dim, n_heads=8, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dim, dim)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.attn(x, x, x)
        x = residual + self.dropout(attn_out)
        residual = x
        x = self.norm2(x)
        x = residual + self.dropout(self.ffn(x))
        return x

class MMGAT(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MMGAT, self).__init__(config, dataset)
        
        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.n_layers = config['n_layers']
        self.dropout = config['dropout']
        self.reg_weight = config['reg_weight']
        self.n_heads = 8
        self.temperature = 0.07
        self.n_nodes = self.n_users + self.n_items
        
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.norm_adj = self.get_norm_adj_mat().to(self.device)
        
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)
        
        if self.v_feat is not None:
            self.v_feat = self.v_feat.to(self.device)
            self.image_encoder = nn.Sequential(
                nn.Linear(self.v_feat.shape[1], self.feat_embed_dim),
                nn.LayerNorm(self.feat_embed_dim),
                nn.GELU(),
                ResidualTransformer(self.feat_embed_dim, self.n_heads, self.dropout),
                nn.Linear(self.feat_embed_dim, self.feat_embed_dim)
            )
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            
        if self.t_feat is not None:
            self.t_feat = self.t_feat.to(self.device)
            self.text_encoder = nn.Sequential(
                nn.Linear(self.t_feat.shape[1], self.feat_embed_dim),
                nn.LayerNorm(self.feat_embed_dim),
                nn.GELU(),
                ResidualTransformer(self.feat_embed_dim, self.n_heads, self.dropout),
                nn.Linear(self.feat_embed_dim, self.feat_embed_dim)
            )
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)

        self.modal_fusion = nn.ModuleList([
            ResidualTransformer(self.feat_embed_dim, self.n_heads, self.dropout)
            for _ in range(2)
        ])
        
        self.online_encoder = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.BatchNorm1d(self.embedding_dim),
            nn.GELU(),
            nn.Linear(self.embedding_dim, self.embedding_dim)
        )
        
        self.target_encoder = copy.deepcopy(self.online_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False
            
        self.predictor = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.BatchNorm1d(self.embedding_dim),
            nn.GELU(),
            nn.Linear(self.embedding_dim, self.embedding_dim)
        )
        
        self.tau = 0.996
        self.register_buffer('queue', torch.randn(65536, self.embedding_dim))
        self.queue = F.normalize(self.queue, dim=1)
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))
        
        self.to(self.device)

    @torch.no_grad()
    def _momentum_update(self):
        for online_params, target_params in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            target_params.data = target_params.data * self.tau + online_params.data * (1. - self.tau)
            
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        if ptr + batch_size > len(self.queue):
            batch_size = len(self.queue) - ptr
            keys = keys[:batch_size]
        self.queue[ptr:ptr + batch_size] = keys
        ptr = (ptr + batch_size) % len(self.queue)
        self.queue_ptr[0] = ptr

    def forward(self):
        modal_feats = []
        if self.v_feat is not None:
            img_feat = self.image_encoder(self.image_embedding.weight)
            modal_feats.append(img_feat)
            
        if self.t_feat is not None:
            txt_feat = self.text_encoder(self.text_embedding.weight)
            modal_feats.append(txt_feat)
            
        if len(modal_feats) > 0:
            if len(modal_feats) > 1:
                modal_feats = torch.stack(modal_feats, dim=1)
                for layer in self.modal_fusion:
                    modal_feats = layer(modal_feats)
                fused_features = modal_feats.mean(dim=1)
            else:
                fused_features = modal_feats[0]
        else:
            fused_features = self.item_embedding.weight

        ego_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight])
        all_embeddings = [ego_embeddings]
        
        for _ in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.norm_adj, ego_embeddings)
            ego_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
            all_embeddings.append(ego_embeddings)
            
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items])
        
        i_g_embeddings = i_g_embeddings + fused_features
        
        return u_g_embeddings, i_g_embeddings

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        self._momentum_update()
        
        u_embeddings, i_embeddings = self.forward()
        
        u_online = self.online_encoder(u_embeddings)
        i_online = self.online_encoder(i_embeddings)
        
        with torch.no_grad():
            u_target = self.target_encoder(u_embeddings)
            i_target = self.target_encoder(i_embeddings)
        
        u_pred = self.predictor(u_online)
        i_pred = self.predictor(i_online)
        
        u_feat = u_pred[users]
        pos_feat = i_pred[pos_items]
        neg_feat = i_pred[neg_items]
        
        u_target = u_target[users]
        i_target = i_target[pos_items]
        
        pos_logits = torch.sum(u_feat * pos_feat, dim=1)
        neg_logits = torch.sum(u_feat * neg_feat, dim=1)
        
        pos_loss = F.binary_cross_entropy_with_logits(
            pos_logits,
            torch.ones_like(pos_logits),
            reduction='none'
        )
        neg_loss = F.binary_cross_entropy_with_logits(
            neg_logits,
            torch.zeros_like(neg_logits),
            reduction='none'
        )
        
        loss_ce = (pos_loss + neg_loss).mean()
        
        queue = self.queue.clone().detach()
        u_contra = torch.einsum('nc,ck->nk', [F.normalize(u_feat, dim=1), queue.T])
        i_contra = torch.einsum('nc,ck->nk', [F.normalize(pos_feat, dim=1), queue.T])
        
        contra_pos = torch.exp(torch.sum(F.normalize(u_feat, dim=1) * F.normalize(pos_feat, dim=1), dim=1) / self.temperature)
        contra_neg_u = torch.sum(torch.exp(u_contra / self.temperature), dim=1)
        contra_neg_i = torch.sum(torch.exp(i_contra / self.temperature), dim=1)
        
        loss_contra = -torch.log(contra_pos / (contra_neg_u + contra_neg_i + contra_pos)).mean()
        
        self._dequeue_and_enqueue(F.normalize(i_target.detach(), dim=1))
        
        reg_loss = self.reg_weight * (
            torch.norm(u_feat) / len(users) +
            torch.norm(pos_feat) / len(pos_items) +
            torch.norm(neg_feat) / len(neg_items)
        )
        
        return loss_ce + loss_contra + reg_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        
        u_embeddings, i_embeddings = self.forward()
        
        u_embeddings = self.predictor(self.online_encoder(u_embeddings))
        i_embeddings = self.predictor(self.online_encoder(i_embeddings))
        
        scores = torch.matmul(u_embeddings[user], i_embeddings.transpose(0, 1))
        return scores

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
        indices = torch.LongTensor(np.array([L.row, L.col]))
        values = torch.FloatTensor(L.data)
        return torch.sparse_coo_tensor(indices, values, torch.Size((self.n_nodes, self.n_nodes)))