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

class ModalGAT(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super(ModalGAT, self).__init__()
        self.gat = GATConv(in_dim, out_dim, heads=4, concat=False, dropout=dropout)
        self.norm = nn.LayerNorm(out_dim)
        self.prelu = nn.PReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index):
        x = self.dropout(x)
        x = self.gat(x, edge_index)
        x = self.norm(x)
        return self.prelu(x)

class ModalEncoder(nn.Module):
    def __init__(self, feat_dim, hidden_dim, embed_dim, n_layers, dropout=0.1):
        super(ModalEncoder, self).__init__()
        self.transform = nn.Linear(feat_dim, hidden_dim)
        self.gat_layers = nn.ModuleList([
            ModalGAT(hidden_dim, hidden_dim, dropout)
            for _ in range(n_layers)
        ])
        self.projection = nn.Linear(hidden_dim, embed_dim)
        
    def forward(self, features, edge_index):
        x = self.transform(features)
        for gat in self.gat_layers:
            x = x + gat(x, edge_index)
        return F.normalize(self.projection(x))

class MMGAT(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MMGAT, self).__init__(config, dataset)
        
        self.embedding_dim = config["embedding_size"]
        self.feat_embed_dim = config["feat_embed_dim"]
        self.hidden_dim = self.feat_embed_dim * 2
        self.n_layers = config["n_mm_layers"]
        self.reg_weight = config["reg_weight"]
        self.knn_k = config["knn_k"]
        self.lambda_coeff = config["lambda_coeff"]
        self.dropout = config["dropout"]
        self.temperature = 0.07
        
        self.n_nodes = self.n_users + self.n_items
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        
        # User-Item Graph Construction
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_normal_(self.user_embedding.weight)
        nn.init.xavier_normal_(self.item_id_embedding.weight)
        
        # Modal Encoders  
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_encoder = ModalEncoder(
                self.v_feat.shape[1], self.hidden_dim, self.feat_embed_dim, 
                self.n_layers, self.dropout
            )
            
        if self.t_feat is not None:  
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_encoder = ModalEncoder(
                self.t_feat.shape[1], self.hidden_dim, self.feat_embed_dim,
                self.n_layers, self.dropout  
            )
        
        # Modal Fusion
        self.fusion = nn.Sequential(
            nn.Linear(self.feat_embed_dim * 2, self.feat_embed_dim),
            nn.LayerNorm(self.feat_embed_dim),
            nn.PReLU(),
            nn.Dropout(self.dropout)
        )
        self.highway = nn.Sequential(
            nn.Linear(self.feat_embed_dim * 2, self.feat_embed_dim),
            nn.Sigmoid()
        )
        self.modal_weight = nn.Parameter(torch.Tensor([0.5, 0.5]))
        self.softmax = nn.Softmax(dim=0)
        
        # Loss weights
        self.ssl_weight = 0.1  
        
        self.to(self.device)
        self.build_graph()
    
    def build_graph(self):
        # Build UI graph
        indices = self.interaction_matrix.nonzero()
        self.edge_index_ui = torch.LongTensor(np.vstack([indices[0], indices[1] + self.n_users])).to(self.device)
        self.edge_weight_ui = torch.ones(self.edge_index_ui.size(1)).to(self.device)
        
        # Build modal graphs
        if self.v_feat is not None:
            self.image_edge_index = self.build_knn_graph(self.v_feat) 
        if self.t_feat is not None:
            self.text_edge_index = self.build_knn_graph(self.t_feat)
            
    def build_knn_graph(self, features):
        sim = torch.mm(features, features.t())
        _, knn_ind = torch.topk(sim, self.knn_k, dim=-1)  
        adj_size = sim.size()
        indices = torch.stack((torch.arange(adj_size[0], device=self.device).repeat_interleave(self.knn_k),
                               knn_ind.reshape(-1))).to(self.device)
        values = torch.ones(indices.size(1), device=self.device)
        mm_adj = torch.sparse.FloatTensor(indices, values, adj_size).to(self.device)
        return self.compute_normalized_laplacian(mm_adj)

    def compute_normalized_laplacian(self, adj):
        row_sum = torch.sparse.sum(adj, -1).to_dense()
        deg_inv_sqrt = torch.pow(row_sum, -0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
        indices = adj.indices()
        values = deg_inv_sqrt[indices[0]] * deg_inv_sqrt[indices[1]] * adj.values()
        return torch.sparse.FloatTensor(indices, values, adj.size())
        
    def forward(self, get_modal_embeddings=False):
        # Process modalities
        image_emb, text_emb = None, None
        
        if self.v_feat is not None:
            image_emb = self.image_encoder(self.image_embedding.weight, self.image_edge_index)
            
        if self.t_feat is not None:  
            text_emb = self.text_encoder(self.text_embedding.weight, self.text_edge_index)

        # Fuse modalities using adaptive weights
        if image_emb is not None and text_emb is not None:
            modal_weights = self.softmax(self.modal_weight)
            modal_cat = torch.cat([image_emb, text_emb], dim=1) 
            gate = self.highway(modal_cat)
            fused = self.fusion(modal_cat)
            item_modal_emb = gate * fused + (1 - gate) * (modal_weights[0] * image_emb + modal_weights[1] * text_emb)
        else:
            item_modal_emb = image_emb if image_emb is not None else text_emb
        
        # Normalize modal embeddings
        item_modal_emb = F.normalize(item_modal_emb)
        
        # Process UI interactions
        ego_embeddings = torch.cat([self.user_embedding.weight, self.item_id_embedding.weight], dim=0)  
        all_embeddings = [ego_embeddings]
        
        # Perform multi-layer LightGCN convolution
        for _ in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.ui_norm_adj, ego_embeddings)
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        
        # Normalize embeddings
        user_all_emb, item_all_emb = torch.split(all_embeddings, [self.n_users, self.n_items])
        user_all_emb = F.normalize(user_all_emb)
        item_all_emb = F.normalize(item_all_emb)
        
        item_all_emb = item_all_emb + item_modal_emb
        
        if get_modal_embeddings:
            return user_all_emb, item_all_emb, F.normalize(image_emb), F.normalize(text_emb)
            
        return user_all_emb, item_all_emb
        
    def ui_lightgcn_conv(self, x, edge_index):  
        row, col = edge_index
        deg = torch.sparse.sum(torch.sparse_coo_tensor(edge_index, torch.ones_like(row), (x.size(0), x.size(0))), dim=1).to_dense()
        deg_inv_sqrt = torch.pow(deg, -0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0  
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]   
        return torch.sparse.mm(torch.sparse_coo_tensor(edge_index, norm, (x.size(0), x.size(0))), x)
        
    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2] 
        
        # Get embeddings with modal info
        user_emb, item_emb, image_emb, text_emb = self.forward(get_modal_embeddings=True)
        
        u_embeddings = user_emb[users]
        pos_embeddings = item_emb[pos_items]
        neg_embeddings = item_emb[neg_items]
        
        # BPR Loss
        pos_scores = torch.sum(u_embeddings * pos_embeddings, dim=1)
        neg_scores = torch.sum(u_embeddings * neg_embeddings, dim=1)
        mf_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        
        # Modal Contrastive Loss
        modal_loss = 0.0
        if image_emb is not None and text_emb is not None:  
            i_emb = image_emb[pos_items]
            t_emb = text_emb[pos_items]
            modal_loss = -torch.mean(
                F.logsigmoid(torch.sum(i_emb * t_emb, dim=1) / self.temperature)
            )
        
        # Self-supervised Loss
        ssl_loss = self.ssl_weight * (  
            self.infonce_loss(pos_embeddings, u_embeddings, neg_embeddings) +
            self.infonce_loss(u_embeddings, pos_embeddings, neg_embeddings)
        )
        
        # L2 Regularization
        reg_loss = self.reg_weight * (
            torch.norm(u_embeddings) + 
            torch.norm(pos_embeddings) +
            torch.norm(neg_embeddings)
        )
        
        return mf_loss + modal_loss + ssl_loss + reg_loss
        
    def infonce_loss(self, anchor, positive, negative):
        pos_sim = torch.sum(anchor * positive, dim=1) / self.temperature 
        neg_sim = torch.matmul(anchor, negative.t()) / self.temperature
        loss = -torch.mean(
            pos_sim - torch.log(torch.exp(neg_sim).sum(dim=1) + 1e-8)  
        )
        return loss
        
    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_emb, item_emb = self.forward()
        scores = torch.matmul(user_emb[user], item_emb.transpose(0, 1))  
        return scores