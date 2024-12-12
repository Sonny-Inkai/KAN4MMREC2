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
        self.n_ui_layers = config["n_ui_layers"]
        self.reg_weight = config["reg_weight"]
        self.dropout = config["dropout"]
        self.build_item_graph = True
        self.knn_k = config["knn_k"]
        
        # Initialize embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_normal_(self.user_embedding.weight)
        nn.init.xavier_normal_(self.item_embedding.weight)
        
        # Modal projections
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_trs = nn.Sequential(
                nn.Linear(self.v_feat.shape[1], self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim, self.feat_embed_dim)
            )
            
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_trs = nn.Sequential(
                nn.Linear(self.t_feat.shape[1], self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim, self.feat_embed_dim)
            )
        
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.norm_adj = self.get_norm_adj_mat().to(self.device)
        self.mm_adj = None
        self.build_modal_graph()
        self.to(self.device)

    def build_modal_graph(self):
        if self.v_feat is not None and self.t_feat is not None:
            v_feat = F.normalize(self.image_trs(self.image_embedding.weight), p=2, dim=1)
            t_feat = F.normalize(self.text_trs(self.text_embedding.weight), p=2, dim=1)
            
            v_adj = self.build_knn_neighborhood(v_feat)
            t_adj = self.build_knn_neighborhood(t_feat)
            
            self.mm_adj = 0.5 * (v_adj + t_adj)
        elif self.v_feat is not None:
            v_feat = F.normalize(self.image_trs(self.image_embedding.weight), p=2, dim=1)
            self.mm_adj = self.build_knn_neighborhood(v_feat)
        else:
            t_feat = F.normalize(self.text_trs(self.text_embedding.weight), p=2, dim=1)
            self.mm_adj = self.build_knn_neighborhood(t_feat)

    def build_knn_neighborhood(self, features):
        sim = torch.mm(features, features.t())
        vals, cols = torch.topk(sim, k=self.knn_k, dim=1)
        rows = torch.arange(features.size(0), device=self.device)
        rows = rows.view(-1, 1).repeat(1, self.knn_k)
        indices = torch.stack([rows.flatten(), cols.flatten()])
        values = vals.flatten()
        adj = torch.sparse_coo_tensor(indices, values, sim.size())
        return self.compute_normalized_laplacian(adj)

    def compute_normalized_laplacian(self, adj):
        row_sum = 1e-7 + torch.sparse.sum(adj, dim=1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        values = adj._values()
        indices = adj._indices()
        norm_values = values * r_inv_sqrt[indices[0]] * r_inv_sqrt[indices[1]]
        return torch.sparse_coo_tensor(indices, norm_values, adj.size())

    def get_norm_adj_mat(self):
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.interaction_matrix.tolil()
        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()

        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum + 1e-7, -0.5).flatten()
        d_mat = sp.diags(d_inv)
        norm_adj = d_mat.dot(adj_mat).dot(d_mat)

        coo = norm_adj.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.FloatTensor(coo.data)
        return torch.sparse_coo_tensor(i, v, torch.Size(coo.shape))

    def message_dropout(self, x, adj):
        if not self.training or self.dropout == 0:
            return torch.sparse.mm(adj, x)
            
        mask = torch.bernoulli(torch.full_like(x, 1 - self.dropout))
        return torch.sparse.mm(adj, x * mask) / (1 - self.dropout)

    def forward(self):
        # Process modalities
        if self.v_feat is not None:
            image_feat = self.image_trs(self.image_embedding.weight)
            
        if self.t_feat is not None:
            text_feat = self.text_trs(self.text_embedding.weight)
            
        # Message passing on modality graph
        h = self.item_embedding.weight
        for _ in range(self.n_layers):
            h = self.message_dropout(h, self.mm_adj)
            h = F.normalize(h, p=2, dim=1)
            
        # Graph convolution with residual connections
        ego_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        all_embeddings = [ego_embeddings]
        
        for _ in range(self.n_ui_layers):
            ego_embeddings = self.message_dropout(ego_embeddings, self.norm_adj)
            all_embeddings.append(ego_embeddings)
            
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1)
        
        user_embeddings, item_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items])
        item_embeddings = item_embeddings + h
        
        return (
            user_embeddings,
            item_embeddings,
            image_feat if self.v_feat is not None else None,
            text_feat if self.t_feat is not None else None
        )

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        
        user_emb, item_emb, img_emb, txt_emb = self.forward()
        
        # Graph-based recommendation loss
        u_e = user_emb[users]
        pos_e = item_emb[pos_items]
        neg_e = item_emb[neg_items]
        
        pos_scores = torch.sum(u_e * pos_e, dim=1)
        neg_scores = torch.sum(u_e * neg_e, dim=1)
        rec_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        
        # Modal-aware recommendation loss
        modal_loss = 0.0
        if img_emb is not None:
            pos_img_scores = torch.sum(u_e * F.normalize(img_emb[pos_items], dim=1), dim=1)
            neg_img_scores = torch.sum(u_e * F.normalize(img_emb[neg_items], dim=1), dim=1)
            modal_loss += -torch.mean(F.logsigmoid(pos_img_scores - neg_img_scores))
            
        if txt_emb is not None:
            pos_txt_scores = torch.sum(u_e * F.normalize(txt_emb[pos_items], dim=1), dim=1)
            neg_txt_scores = torch.sum(u_e * F.normalize(txt_emb[neg_items], dim=1), dim=1)
            modal_loss += -torch.mean(F.logsigmoid(pos_txt_scores - neg_txt_scores))
            
        # L2 regularization
        reg_loss = self.reg_weight * (
            torch.norm(u_e, p=2) +
            torch.norm(pos_e, p=2) +
            torch.norm(neg_e, p=2)
        )
        
        # Cross-modal contrastive loss
        if img_emb is not None and txt_emb is not None:
            i_feat = F.normalize(img_emb[pos_items], dim=1)
            t_feat = F.normalize(txt_emb[pos_items], dim=1)
            contra_loss = -torch.mean(F.logsigmoid(torch.sum(i_feat * t_feat, dim=1)))
            modal_loss += 0.1 * contra_loss
        
        return rec_loss + 0.2 * modal_loss + reg_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_emb, item_emb, _, _ = self.forward()
        scores = torch.matmul(user_emb[user], item_emb.transpose(0, 1))
        return scores