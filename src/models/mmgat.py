# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
import os
import numpy as np
import scipy.sparse as sp
from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss

class MMGAT(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MMGAT, self).__init__(config, dataset)

        self.embedding_dim = config["embedding_size"]
        self.feat_embed_dim = config["feat_embed_dim"]
        self.n_layers = config["n_mm_layers"]
        self.n_heads = config["n_heads"]
        self.dropout = config["dropout"]
        self.knn_k = config["knn_k"]
        self.reg_weight = config["reg_weight"]
        self.lambda_coeff = config["lambda_coeff"]
        
        self.n_nodes = self.n_users + self.n_items
        
        # Load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.norm_adj = self.get_norm_adj_mat().to(self.device)

        # Embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        # Modal processing
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)

        # Graph attention layers
        self.gat_layers = nn.ModuleList([
            dglnn.GATConv(
                self.feat_embed_dim, 
                self.feat_embed_dim, 
                self.n_heads,
                feat_drop=self.dropout,
                attn_drop=self.dropout,
                residual=True,
                activation=F.leaky_relu
            ) for _ in range(self.n_layers)
        ])

        self.modal_weight = nn.Parameter(torch.Tensor([0.5, 0.5]))
        self.softmax = nn.Softmax(dim=0)
        
        # Build modal graph
        self.mm_graph = None
        self.build_modal_graph()

    def build_modal_graph(self):
        # Create DGL graph for modal processing
        if self.v_feat is not None:
            image_feats = self.image_trs(self.image_embedding.weight)
            image_sim = F.cosine_similarity(image_feats.unsqueeze(1), image_feats.unsqueeze(0), dim=2)
            image_adj = torch.zeros_like(image_sim)
            topk_values, topk_indices = torch.topk(image_sim, self.knn_k, dim=1)
            image_adj.scatter_(1, topk_indices, 1)
            src_i, dst_i = torch.nonzero(image_adj, as_tuple=True)
            
            if self.t_feat is not None:
                text_feats = self.text_trs(self.text_embedding.weight)
                text_sim = F.cosine_similarity(text_feats.unsqueeze(1), text_feats.unsqueeze(0), dim=2)
                text_adj = torch.zeros_like(text_sim)
                topk_values, topk_indices = torch.topk(text_sim, self.knn_k, dim=1)
                text_adj.scatter_(1, topk_indices, 1)
                src_t, dst_t = torch.nonzero(text_adj, as_tuple=True)
                
                # Combine modalities
                weight = self.softmax(self.modal_weight)
                src = torch.cat([src_i, src_t])
                dst = torch.cat([dst_i, dst_t])
                self.mm_graph = dgl.graph((src, dst), num_nodes=self.n_items).to(self.device)
            else:
                self.mm_graph = dgl.graph((src_i, dst_i), num_nodes=self.n_items).to(self.device)
        else:
            text_feats = self.text_trs(self.text_embedding.weight)
            text_sim = F.cosine_similarity(text_feats.unsqueeze(1), text_feats.unsqueeze(0), dim=2)
            text_adj = torch.zeros_like(text_sim)
            topk_values, topk_indices = torch.topk(text_sim, self.knn_k, dim=1)
            text_adj.scatter_(1, topk_indices, 1)
            src_t, dst_t = torch.nonzero(text_adj, as_tuple=True)
            self.mm_graph = dgl.graph((src_t, dst_t), num_nodes=self.n_items).to(self.device)

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

    def forward(self, adj):
        # Multimodal feature learning
        if self.v_feat is not None:
            item_feats = self.image_trs(self.image_embedding.weight)
        if self.t_feat is not None:
            if self.v_feat is not None:
                text_feats = self.text_trs(self.text_embedding.weight)
                weight = self.softmax(self.modal_weight)
                item_feats = weight[0] * item_feats + weight[1] * text_feats
            else:
                item_feats = self.text_trs(self.text_embedding.weight)

        # Graph attention on item features
        h = item_feats
        for gat_layer in self.gat_layers:
            h = gat_layer(self.mm_graph, h).mean(1)
            h = F.dropout(h, self.dropout, training=self.training)

        # User-Item graph learning
        ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
        all_embeddings = [ego_embeddings]
        
        for _ in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(adj, ego_embeddings)
            all_embeddings.append(ego_embeddings)
            
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        
        # Combine item embeddings with multimodal features
        i_g_embeddings = i_g_embeddings + F.normalize(h, p=2, dim=1)
        
        return u_g_embeddings, i_g_embeddings

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        
        u_embeddings, i_embeddings = self.forward(self.norm_adj)
        
        u_embeddings = u_embeddings[users]
        pos_embeddings = i_embeddings[pos_items]
        neg_embeddings = i_embeddings[neg_items]
        
        # BPR Loss
        pos_scores = torch.sum(torch.mul(u_embeddings, pos_embeddings), dim=1)
        neg_scores = torch.sum(torch.mul(u_embeddings, neg_embeddings), dim=1)
        
        mf_loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores)))
        
        # Regularization Loss
        reg_loss = self.reg_weight * (
            torch.norm(u_embeddings) ** 2 +
            torch.norm(pos_embeddings) ** 2 +
            torch.norm(neg_embeddings) ** 2
        ) / (2 * len(users))
        
        return mf_loss + reg_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        
        u_embeddings, i_embeddings = self.forward(self.norm_adj)
        u_embeddings = u_embeddings[user]
        
        scores = torch.matmul(u_embeddings, i_embeddings.transpose(0, 1))
        return scores