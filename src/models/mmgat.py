# coding: utf-8
import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss

class MMGAT(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MMGAT, self).__init__(config, dataset)
        
        self.embedding_dim = config["embedding_size"]
        self.feat_embed_dim = config["feat_embed_dim"]
        self.n_mm_layers = config["n_mm_layers"]
        self.reg_weight = config["reg_weight"]
        self.knn_k = config["knn_k"]
        self.mm_image_weight = config["mm_image_weight"]
        self.dropout = config["dropout"]
        
        self.n_nodes = self.n_users + self.n_items
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.norm_adj = self.get_norm_adj_mat().to(self.device)
        
        # Initialize embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_id_embedding.weight, std=0.1)
        
        # Multimodal components
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_transform = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)
            self.image_predictor = nn.Linear(self.feat_embed_dim, self.feat_embed_dim)
            
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_transform = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)
            self.text_predictor = nn.Linear(self.feat_embed_dim, self.feat_embed_dim)
        
        # Modal fusion
        self.modal_fusion = nn.Linear(self.feat_embed_dim * 2, self.feat_embed_dim)
        self.modal_gate = nn.Sequential(
            nn.Linear(self.feat_embed_dim * 2, 2),
            nn.Softmax(dim=1)
        )
        
        # Graph structure
        self.mm_adj = None
        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        mm_adj_file = os.path.join(dataset_path, f'mm_adj_{self.knn_k}.pt')
        
        if os.path.exists(mm_adj_file):
            self.mm_adj = torch.load(mm_adj_file)
        else:
            if self.v_feat is not None:
                image_adj = self.build_knn_graph(self.image_embedding.weight.detach())
                self.mm_adj = image_adj
            if self.t_feat is not None:
                text_adj = self.build_knn_graph(self.text_embedding.weight.detach())
                self.mm_adj = text_adj
            if self.v_feat is not None and self.t_feat is not None:
                self.mm_adj = self.mm_image_weight * image_adj + (1.0 - self.mm_image_weight) * text_adj
            torch.save(self.mm_adj, mm_adj_file)
        
        self.to(self.device)

    def get_norm_adj_mat(self):
        A = sp.dok_matrix((self.n_nodes, self.n_nodes), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        A._update(data_dict)
        
        sumArr = (A > 0).sum(axis=1)
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        L = sp.coo_matrix(L)
        rows = L.row
        cols = L.col
        values = L.data
        indices = torch.LongTensor([rows, cols])
        values = torch.FloatTensor(values)
        return torch.sparse_coo_tensor(indices, values, [self.n_nodes, self.n_nodes])

    def build_knn_graph(self, features):
        dot_sim = torch.mm(features, features.t())
        norm = torch.norm(features, dim=1)
        norm_mat = torch.ger(norm, norm)
        sim_mat = dot_sim / (norm_mat + 1e-8)
        
        knn_val, knn_ind = torch.topk(sim_mat, self.knn_k, dim=-1)
        adj_size = sim_mat.size()
        
        indices0 = torch.arange(knn_ind.shape[0], device=self.device)
        indices0 = torch.unsqueeze(indices0, 1)
        indices0 = indices0.expand(-1, self.knn_k)
        indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)
        
        return self.compute_normalized_laplacian(indices, adj_size)

    def compute_normalized_laplacian(self, indices, adj_size):
        adj = torch.sparse_coo_tensor(indices, torch.ones_like(indices[0].float()), adj_size).to(self.device)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return torch.sparse_coo_tensor(indices, values, adj_size)

    def forward(self):
        # Process modalities
        if self.v_feat is not None:
            image_feat = F.dropout(self.image_transform(self.image_embedding.weight), p=self.dropout, training=self.training)
            image_embed = self.image_predictor(image_feat)
            
        if self.t_feat is not None:
            text_feat = F.dropout(self.text_transform(self.text_embedding.weight), p=self.dropout, training=self.training)
            text_embed = self.text_predictor(text_feat)
        
        # Modal fusion with attention
        if self.v_feat is not None and self.t_feat is not None:
            concat_embed = torch.cat([image_embed, text_embed], dim=1)
            gate_weights = self.modal_gate(concat_embed)
            modal_embed = gate_weights[:, 0].unsqueeze(1) * image_embed + gate_weights[:, 1].unsqueeze(1) * text_embed
        else:
            modal_embed = image_embed if self.v_feat is not None else text_embed
            
        # Graph convolution
        for _ in range(self.n_mm_layers):
            modal_embed = torch.sparse.mm(self.mm_adj, modal_embed)
            
        # User-item graph convolution
        ego_embeddings = torch.cat([self.user_embedding.weight, self.item_id_embedding.weight], dim=0)
        all_embeddings = [ego_embeddings]
        
        for _ in range(self.n_mm_layers):
            ego_embeddings = torch.sparse.mm(self.norm_adj, ego_embeddings)
            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
            all_embeddings.append(norm_embeddings)
            
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        user_embeddings, item_id_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items])
        
        # Combine item embeddings
        item_embeddings = item_id_embeddings + modal_embed
        return user_embeddings, item_embeddings

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        
        u_embeddings, i_embeddings = self.forward()
        
        u_embeddings = u_embeddings[users]
        pos_embeddings = i_embeddings[pos_items]
        neg_embeddings = i_embeddings[neg_items]
        
        # BPR loss
        pos_scores = torch.sum(u_embeddings * pos_embeddings, dim=1)
        neg_scores = torch.sum(u_embeddings * neg_embeddings, dim=1)
        loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8)
        mf_loss = torch.mean(loss)
        
        # L2 regularization
        reg_loss = self.reg_weight * (
            torch.norm(u_embeddings) +
            torch.norm(pos_embeddings) +
            torch.norm(neg_embeddings)
        )
        
        return mf_loss + reg_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_embeddings, item_embeddings = self.forward()
        u_embeddings = user_embeddings[user]
        scores = torch.matmul(u_embeddings, item_embeddings.transpose(0, 1))
        return scores