# coding: utf-8
import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss, L2Loss

class SparseMMGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super(SparseMMGATLayer, self).__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Parameter(torch.zeros(size=(1, 2*out_dim)))
        nn.init.xavier_normal_(self.a.data)
        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.norm = nn.LayerNorm(out_dim)
        
    def forward(self, h, edge_index):
        N = h.size(0)
        h = self.W(h)
        
        # Compute attention scores
        edge_h = torch.cat([h[edge_index[0]], h[edge_index[1]]], dim=1)
        edge_e = self.leakyrelu(torch.matmul(edge_h, self.a.T))
        
        attention = torch.sparse_softmax(edge_e, edge_index[0])
        attention = self.dropout(attention)
        
        # Apply attention
        h_prime = torch.zeros_like(h)
        h_prime.index_add_(0, edge_index[1], attention.unsqueeze(-1) * h[edge_index[0]])
        
        return self.norm(h + h_prime)

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
            self.image_gat = SparseMMGATLayer(self.feat_embed_dim, self.feat_embed_dim)
            
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)
            self.text_gat = SparseMMGATLayer(self.feat_embed_dim, self.feat_embed_dim)
        
        self.fusion = nn.Sequential(
            nn.Linear(self.feat_embed_dim * 2, self.feat_embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.feat_embed_dim, self.feat_embed_dim),
            nn.LayerNorm(self.feat_embed_dim)
        )
        
        self.build_graph_structure()
        
    def build_graph_structure(self):
        # Build UI graph
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
        
        self.norm_adj = self.sparse_mx_to_torch_sparse_tensor(norm_adj).to(self.device)
        
        # Build modal graphs
        if self.v_feat is not None:
            self.image_edge_index = self.build_knn_graph(self.v_feat)
        if self.t_feat is not None:
            self.text_edge_index = self.build_knn_graph(self.t_feat)

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        sparse_mx = sparse_mx.tocoo()
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col))).long()
        values = torch.FloatTensor(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def build_knn_graph(self, features):
        # Compute similarities
        norm_feat = F.normalize(features, p=2, dim=1)
        dist = torch.mm(norm_feat, norm_feat.t())
        
        # Get KNN
        _, topk_indices = dist.topk(k=self.knn_k, dim=1)
        rows = torch.arange(features.size(0)).view(-1, 1).repeat(1, self.knn_k)
        edge_index = torch.stack([rows.reshape(-1), topk_indices.reshape(-1)]).to(self.device)
        return edge_index

    def forward(self):
        image_out, text_out = None, None
        
        if self.v_feat is not None:
            image_feats = self.image_trs(self.image_embedding.weight)
            for _ in range(self.n_layers):
                image_feats = self.image_gat(image_feats, self.image_edge_index)
            image_out = image_feats
            
        if self.t_feat is not None:
            text_feats = self.text_trs(self.text_embedding.weight)
            for _ in range(self.n_layers):
                text_feats = self.text_gat(text_feats, self.text_edge_index)
            text_out = text_feats

        if image_out is not None and text_out is not None:
            mm_embedding = self.fusion(torch.cat([image_out, text_out], dim=1))
        else:
            mm_embedding = image_out if image_out is not None else text_out

        # UI Graph Convolution
        ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
        all_embeddings = [ego_embeddings]
        
        for _ in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.norm_adj, ego_embeddings)
            all_embeddings.append(ego_embeddings)
            
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1)
        
        user_embeddings, item_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items])
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

        # BPR Loss
        pos_scores = torch.sum(user_e * pos_e, dim=1)
        neg_scores = torch.sum(user_e * neg_e, dim=1)
        mf_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))

        # Modal Contrastive Loss
        modal_loss = 0.0
        if self.v_feat is not None and self.t_feat is not None:
            image_feats = F.normalize(self.image_trs(self.image_embedding.weight[pos_items]))
            text_feats = F.normalize(self.text_trs(self.text_embedding.weight[pos_items]))
            modal_loss = -torch.mean(F.cosine_similarity(image_feats, text_feats)) / self.temperature

        # Regularization Loss
        reg_loss = self.reg_weight * (
            torch.norm(user_e) +
            torch.norm(pos_e) +
            torch.norm(neg_e)
        )

        return mf_loss + 0.1 * modal_loss + reg_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        
        user_embeddings, item_embeddings = self.forward()
        u_embeddings = user_embeddings[user]
        
        scores = torch.matmul(u_embeddings, item_embeddings.transpose(0, 1))
        return scores