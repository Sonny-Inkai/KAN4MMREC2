# coding: utf-8
import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss, L2Loss

class MMGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super(MMGATLayer, self).__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Linear(2 * out_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.norm = nn.LayerNorm(out_dim)
        
    def forward(self, x, adj):
        h = self.W(x)
        batch_size, N = h.size(0), h.size(1)
        
        a_input = torch.cat([h.repeat(1,1,N).view(batch_size, N * N, -1),
                           h.repeat(1,N,1)], dim=2).view(batch_size, N, N, 2 * h.size(2))
        
        e = self.leakyrelu(self.a(a_input).squeeze(3))
        attention = F.softmax(e, dim=2)
        attention = self.dropout(attention)
        
        h_prime = torch.matmul(attention, h)
        
        return self.norm(h_prime)

class MMGAT(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MMGAT, self).__init__(config, dataset)
        
        self.embedding_dim = config["embedding_size"]
        self.feat_embed_dim = config["feat_embed_dim"]
        self.n_layers = config["n_mm_layers"] 
        self.reg_weight = config["reg_weight"]
        self.knn_k = config["knn_k"]
        self.mm_image_weight = config["mm_image_weight"]
        self.lambda_coeff = config["lambda_coeff"]
        self.temperature = 0.2
        
        self.n_nodes = self.n_users + self.n_items
        
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)
        
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)
            self.image_gat_layers = nn.ModuleList([
                MMGATLayer(self.feat_embed_dim, self.feat_embed_dim) 
                for _ in range(self.n_layers)
            ])
            
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)
            self.text_gat_layers = nn.ModuleList([
                MMGATLayer(self.feat_embed_dim, self.feat_embed_dim)
                for _ in range(self.n_layers)
            ])
            
        self.fusion = nn.Sequential(
            nn.Linear(self.feat_embed_dim * 2, self.feat_embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.feat_embed_dim, self.feat_embed_dim),
            nn.LayerNorm(self.feat_embed_dim)
        )
        
        self.build_graph_structure()

    def build_graph_structure(self):
        adj_mat = sp.dok_matrix((self.n_nodes, self.n_nodes), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.interaction_matrix.tolil()
        
        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()
        
        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum + 1e-9, -0.5).flatten()
        d_mat = sp.diags(d_inv)
        norm_adj = d_mat.dot(adj_mat).dot(d_mat)
        
        coo = norm_adj.tocoo()
        values = coo.data
        indices = np.vstack((coo.row, coo.col))
        
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = torch.Size(coo.shape)
        
        self.norm_adj = torch.sparse.FloatTensor(i, v, shape).to(self.device)
        
        # Build modal graphs
        if self.v_feat is not None:
            self.image_adj = self.build_modal_graph(self.v_feat)
        if self.t_feat is not None:
            self.text_adj = self.build_modal_graph(self.t_feat)

    def build_modal_graph(self, features):
        sim_matrix = torch.mm(F.normalize(features, p=2, dim=1), 
                            F.normalize(features, p=2, dim=1).t())
        values, indices = sim_matrix.topk(k=self.knn_k, dim=-1)
        rows = torch.arange(features.size(0)).unsqueeze(1).expand_as(indices)
        adj = torch.zeros_like(sim_matrix)
        adj[rows.flatten(), indices.flatten()] = values.flatten()
        adj = 0.5 * (adj + adj.t())
        return adj.to(self.device)

    def forward(self):
        image_out, text_out = None, None
        
        if self.v_feat is not None:
            image_feats = self.image_trs(self.image_embedding.weight)
            image_out = image_feats.unsqueeze(0)
            for gat in self.image_gat_layers:
                image_out = image_out + gat(image_out, self.image_adj)
            image_out = image_out.squeeze(0)
            
        if self.t_feat is not None:
            text_feats = self.text_trs(self.text_embedding.weight)
            text_out = text_feats.unsqueeze(0)
            for gat in self.text_gat_layers:
                text_out = text_out + gat(text_out, self.text_adj)
            text_out = text_out.squeeze(0)
            
        if image_out is not None and text_out is not None:
            mm_embedding = self.fusion(torch.cat([image_out, text_out], dim=1))
        else:
            mm_embedding = image_out if image_out is not None else text_out

        # User-item graph convolution    
        ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
        embeddings_list = [ego_embeddings]
        
        for _ in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.norm_adj, ego_embeddings)
            embeddings_list.append(ego_embeddings)
            
        final_embeddings = torch.stack(embeddings_list, dim=1).mean(dim=1)
        user_embeddings, item_embeddings = torch.split(final_embeddings, [self.n_users, self.n_items])
        
        item_embeddings = item_embeddings + mm_embedding
        
        return user_embeddings, item_embeddings
        
    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        
        user_embeddings, item_embeddings = self.forward()
        
        user_e = user_embeddings[users]
        pos_e = item_embeddings[pos_items]  
        neg_e = item_embeddings[neg_items]
        
        # BPR loss
        pos_scores = torch.sum(user_e * pos_e, dim=1)
        neg_scores = torch.sum(user_e * neg_e, dim=1)
        bpr_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        
        # Modal contrast loss
        modal_loss = 0.0
        if self.v_feat is not None and self.t_feat is not None:
            image_feats = F.normalize(self.image_trs(self.image_embedding.weight[pos_items]), p=2, dim=1)
            text_feats = F.normalize(self.text_trs(self.text_embedding.weight[pos_items]), p=2, dim=1)
            modal_loss = -torch.mean(F.cosine_similarity(image_feats, text_feats, dim=1)) / self.temperature
            
        # Regularization
        reg_loss = self.reg_weight * (
            torch.norm(user_e, p=2) +
            torch.norm(pos_e, p=2) +
            torch.norm(neg_e, p=2)
        )
        
        return bpr_loss + 0.1 * modal_loss + reg_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        
        user_embeddings, item_embeddings = self.forward()
        u_embeddings = user_embeddings[user]
        
        scores = torch.matmul(u_embeddings, item_embeddings.transpose(0, 1))
        return scores