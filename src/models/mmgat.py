# coding: utf-8
import os
import random
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss, L2Loss

class DualGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=4):
        super(DualGATLayer, self).__init__()
        self.struct_gat = GATv2Conv(in_dim, out_dim // num_heads, heads=num_heads, 
                                   concat=True, dropout=0.1)
        self.feat_gat = GATv2Conv(in_dim, out_dim // num_heads, heads=num_heads, 
                                 concat=True, dropout=0.1)
        self.fusion = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim),
            nn.LayerNorm(out_dim),
            nn.PReLU()
        )

    def forward(self, x, struct_edge_index, feat_edge_index):
        struct_h = self.struct_gat(x, struct_edge_index)
        feat_h = self.feat_gat(x, feat_edge_index)
        return self.fusion(torch.cat([struct_h, feat_h], dim=-1))

class MMGAT(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MMGAT, self).__init__(config, dataset)
        
        # Basic configs
        self.embedding_dim = config["embedding_size"]
        self.feat_embed_dim = config["feat_embed_dim"]
        self.n_layers = config["n_mm_layers"]
        self.dropout = config["dropout"]
        self.knn_k = config["knn_k"]
        self.reg_weight = config["reg_weight"]
        self.mm_image_weight = config["mm_image_weight"]
        self.temperature = 0.2
        self.lambda_coeff = config["lambda_coeff"]
        self.n_nodes = self.n_users + self.n_items
        
        # Load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        
        # User-Item embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        
        # Modal processing
        if self.v_feat is not None:
            self.v_feat = torch.tensor(self.v_feat, dtype=torch.float32, device=self.device)
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_projection = nn.Sequential(
                nn.Linear(self.v_feat.shape[1], self.feat_embed_dim),
                nn.LayerNorm(self.feat_embed_dim),
                nn.PReLU(),
                nn.Dropout(0.1)
            )
            
        if self.t_feat is not None:
            self.t_feat = torch.tensor(self.t_feat, dtype=torch.float32, device=self.device)
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_projection = nn.Sequential(
                nn.Linear(self.t_feat.shape[1], self.feat_embed_dim),
                nn.LayerNorm(self.feat_embed_dim),
                nn.PReLU(),
                nn.Dropout(0.1)
            )
            
        # Dual-channel GAT
        self.gat_layers = nn.ModuleList([
            DualGATLayer(self.feat_embed_dim, self.feat_embed_dim)
            for _ in range(self.n_layers)
        ])
        
        # Cross-modal fusion
        self.modal_attention = nn.Sequential(
            nn.Linear(self.feat_embed_dim * 2, self.feat_embed_dim),
            nn.Tanh(),
            nn.Linear(self.feat_embed_dim, 2),
            nn.Softmax(dim=-1)
        )

        # Final prediction layers
        self.predictor = nn.Sequential(
            nn.Linear(self.feat_embed_dim, self.feat_embed_dim),
            nn.PReLU(),
            nn.Linear(self.feat_embed_dim, self.feat_embed_dim)
        )
        
        # Move to device and initialize
        self.to(self.device)
        self._init_weights()
        
        # Build graphs
        self.norm_adj = self.build_norm_adj().to(self.device)
        self.mm_adj, self.mm_edge_index = self.build_modal_graph()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)
        
    def build_norm_adj(self):
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
        
        norm_adj = norm_adj.tocoo()
        indices = torch.LongTensor([norm_adj.row, norm_adj.col])
        values = torch.FloatTensor(norm_adj.data)
        
        return torch.sparse_coo_tensor(indices, values, 
                                     (self.n_nodes, self.n_nodes))
    
    def build_modal_graph(self):
        # Get modal features
        if self.v_feat is not None:
            image_feats = self.image_projection(self.image_embedding.weight)
            image_sim = self.compute_similarity(image_feats)
            modal_adj = image_sim
            
        if self.t_feat is not None:
            text_feats = self.text_projection(self.text_embedding.weight)
            text_sim = self.compute_similarity(text_feats)
            if self.v_feat is not None:
                modal_adj = self.mm_image_weight * image_sim + (1 - self.mm_image_weight) * text_sim
            else:
                modal_adj = text_sim
                
        # KNN graph construction
        topk_values, topk_indices = torch.topk(modal_adj, self.knn_k, dim=1)
        mask = torch.zeros_like(modal_adj)
        mask.scatter_(1, topk_indices, 1.0)
        modal_adj = modal_adj * mask
        
        # Normalize
        deg = torch.sum(modal_adj, dim=1)
        deg_inv_sqrt = torch.pow(deg + 1e-9, -0.5)
        modal_adj = deg_inv_sqrt.unsqueeze(-1) * modal_adj * deg_inv_sqrt.unsqueeze(0)
        
        edge_index = torch.nonzero(modal_adj).t()
        
        return modal_adj.to(self.device), edge_index.to(self.device)
        
    def compute_similarity(self, features):
        features = F.normalize(features, p=2, dim=1)
        return torch.mm(features, features.t())
        
    def forward(self, build_graph=True):
        # Process modal features
        if self.v_feat is not None:
            image_feats = self.image_projection(self.image_embedding.weight)
        if self.t_feat is not None:
            text_feats = self.text_projection(self.text_embedding.weight)
        
        # Modal fusion with attention
        if self.v_feat is not None and self.t_feat is not None:
            concat_feats = torch.cat([image_feats, text_feats], dim=1)
            attention = self.modal_attention(concat_feats)
            item_feats = attention[:, 0].unsqueeze(1) * image_feats + \
                        attention[:, 1].unsqueeze(1) * text_feats
        else:
            item_feats = image_feats if self.v_feat is not None else text_feats
            
        # Dual-channel GAT
        x = item_feats
        for gat_layer in self.gat_layers:
            x = gat_layer(x, self.norm_adj._indices(), self.mm_edge_index)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
        # User-Item graph convolution
        ego_embeddings = torch.cat([self.user_embedding.weight, 
                                  self.item_id_embedding.weight], dim=0)
        all_embeddings = [ego_embeddings]
        
        for _ in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.norm_adj, ego_embeddings)
            all_embeddings.append(ego_embeddings)
            
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, 
                                                   [self.n_users, self.n_items])
        
        # Final prediction
        item_embeddings = i_g_embeddings + self.predictor(x)
        
        return u_g_embeddings, item_embeddings

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        
        u_embeddings, i_embeddings = self.forward()
        
        u_embeddings = u_embeddings[users]
        pos_embeddings = i_embeddings[pos_items]
        neg_embeddings = i_embeddings[neg_items]
        
        # BPR Loss
        pos_scores = torch.sum(u_embeddings * pos_embeddings, dim=1)
        neg_scores = torch.sum(u_embeddings * neg_embeddings, dim=1)
        mf_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        
        # Contrastive Loss for modalities
        if self.v_feat is not None and self.t_feat is not None:
            image_feats = self.image_projection(self.image_embedding.weight)[pos_items]
            text_feats = self.text_projection(self.text_embedding.weight)[pos_items]
            
            image_feats = F.normalize(image_feats, dim=-1)
            text_feats = F.normalize(text_feats, dim=-1)
            
            modal_sim = torch.exp(torch.sum(image_feats * text_feats, dim=-1) / self.temperature)
            modal_neg = torch.exp(torch.mm(image_feats, text_feats.t()) / self.temperature)
            
            contra_loss = -torch.mean(torch.log(modal_sim / (modal_neg.sum(dim=1))))
        else:
            contra_loss = 0.0
        
        # Regularization
        reg_loss = self.reg_weight * (
            torch.norm(u_embeddings) +
            torch.norm(pos_embeddings) +
            torch.norm(neg_embeddings)
        )
        
        return mf_loss + 0.1 * contra_loss + reg_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        u_embeddings, i_embeddings = self.forward()
        u_embeddings = u_embeddings[user]
        scores = torch.matmul(u_embeddings, i_embeddings.transpose(0, 1))
        return scores