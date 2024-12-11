# coding: utf-8
import os
import random
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss, L2Loss

class MMGAT(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MMGAT, self).__init__(config, dataset)
        
        self.embedding_dim = config["embedding_size"]
        self.feat_embed_dim = config["feat_embed_dim"]
        self.n_layers = config["n_mm_layers"]
        self.reg_weight = config["reg_weight"]
        self.knn_k = config["knn_k"]
        self.lambda_coeff = config["lambda_coeff"]
        self.temperature = 0.2
        self.dropout = config["dropout"]
        
        self.n_nodes = self.n_users + self.n_items
        
        # Load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.edge_index, self.edge_value = self.get_edge_info()
        self.edge_index = self.edge_index.to(self.device)
        self.edge_value = self.edge_value.to(self.device)
        
        # Initialize embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        # Modality-specific components
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)
            self.image_gat = GATConv(self.feat_embed_dim, self.feat_embed_dim, heads=1, dropout=self.dropout)
            
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)
            self.text_gat = GATConv(self.feat_embed_dim, self.feat_embed_dim, heads=1, dropout=self.dropout)
            
        # Feature fusion
        self.modal_fusion = nn.Sequential(
            nn.Linear(self.feat_embed_dim * 2, self.feat_embed_dim),
            nn.LayerNorm(self.feat_embed_dim),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout)
        )
        
        # Move to device
        self.to(self.device)
        
        # Initialize modal graphs
        self.build_modal_graph()
        
    def get_edge_info(self):
        rows = torch.from_numpy(self.interaction_matrix.row)
        cols = torch.from_numpy(self.interaction_matrix.col + self.n_users)
        edge_index = torch.stack([rows, cols])
        edge_weight = torch.ones(edge_index.size(1))
        return edge_index, edge_weight
        
    def build_modal_graph(self):
        if self.v_feat is not None:
            self.image_edge_index = self.build_knn_graph(self.image_embedding.weight)
            
        if self.t_feat is not None:
            self.text_edge_index = self.build_knn_graph(self.text_embedding.weight)
            
    def build_knn_graph(self, features):
        sim = torch.mm(F.normalize(features, dim=1), F.normalize(features, dim=1).t())
        topk_indices = torch.topk(sim, k=self.knn_k, dim=1)[1]
        rows = torch.arange(features.size(0), device=self.device).repeat_interleave(self.knn_k)
        cols = topk_indices.reshape(-1)
        edge_index = torch.stack([rows, cols]).long()
        return edge_index
        
    def forward(self):
        # Process visual modality
        image_emb = None
        if self.v_feat is not None:
            image_emb = self.image_trs(self.image_embedding.weight)
            image_emb = F.dropout(image_emb, p=self.dropout, training=self.training)
            image_emb = self.image_gat(image_emb, self.image_edge_index)
            
        # Process textual modality
        text_emb = None
        if self.t_feat is not None:
            text_emb = self.text_trs(self.text_embedding.weight)
            text_emb = F.dropout(text_emb, p=self.dropout, training=self.training)
            text_emb = self.text_gat(text_emb, self.text_edge_index)
            
        # Fuse modalities
        if image_emb is not None and text_emb is not None:
            item_emb = self.modal_fusion(torch.cat([image_emb, text_emb], dim=1))
        else:
            item_emb = image_emb if image_emb is not None else text_emb
            
        # Process user-item graph
        ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
        all_embeddings = [ego_embeddings]
        
        # Graph convolution
        for _ in range(self.n_layers):
            side_embeddings = torch.sparse.mm(self.get_sparse_adj_mat(), ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings.append(ego_embeddings)
            
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1)
        
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items])
        i_g_embeddings = i_g_embeddings + item_emb
        
        return u_g_embeddings, i_g_embeddings
        
    def get_sparse_adj_mat(self):
        edge_index = torch.cat([self.edge_index, self.edge_index.flip(0)], dim=1)
        edge_value = torch.cat([self.edge_value, self.edge_value], dim=0)
        adj_shape = torch.Size([self.n_nodes, self.n_nodes])
        return torch.sparse_coo_tensor(edge_index, edge_value, adj_shape)
        
    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        
        u_embeddings, i_embeddings = self.forward()
        
        # BPR Loss
        u_embeddings = u_embeddings[users]
        pos_embeddings = i_embeddings[pos_items]
        neg_embeddings = i_embeddings[neg_items]
        
        pos_scores = torch.sum(u_embeddings * pos_embeddings, dim=1)
        neg_scores = torch.sum(u_embeddings * neg_embeddings, dim=1)
        mf_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        
        # Contrastive Loss for modal features
        modal_loss = 0.0
        if self.v_feat is not None and self.t_feat is not None:
            v_feat = F.normalize(self.image_trs(self.image_embedding.weight[pos_items]))
            t_feat = F.normalize(self.text_trs(self.text_embedding.weight[pos_items]))
            modal_loss = -torch.mean(F.cosine_similarity(v_feat, t_feat)) / self.temperature
            
        # L2 Regularization
        reg_loss = self.reg_weight * (
            torch.norm(u_embeddings) +
            torch.norm(pos_embeddings) +
            torch.norm(neg_embeddings)
        )
        
        return mf_loss + 0.1 * modal_loss + reg_loss
        
    def full_sort_predict(self, interaction):
        user = interaction[0]
        u_embeddings, i_embeddings = self.forward()
        score = torch.matmul(u_embeddings[user], i_embeddings.transpose(0, 1))
        return score