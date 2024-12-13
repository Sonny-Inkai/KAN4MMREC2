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

class BootstrapEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.online_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        self.target_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        self.predictor = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim)
        )
        
        # Initialize target network with online network weights
        for online_params, target_params in zip(
            self.online_encoder.parameters(), self.target_encoder.parameters()
        ):
            target_params.data.copy_(online_params.data)
            target_params.requires_grad = False
            
    def forward(self, x, momentum=0.99):
        online_feat = self.online_encoder(x)
        with torch.no_grad():
            for online_params, target_params in zip(
                self.online_encoder.parameters(), self.target_encoder.parameters()
            ):
                target_params.data = momentum * target_params.data + (1 - momentum) * online_params.data
            target_feat = self.target_encoder(x)
            
        pred_feat = self.predictor(online_feat)
        return online_feat, pred_feat, target_feat

class MMGAT(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MMGAT, self).__init__(config, dataset)
        
        # Parameters
        self.embedding_dim = config["embedding_size"]
        self.feat_embed_dim = config["feat_embed_dim"]
        self.hidden_dim = self.feat_embed_dim * 2
        self.n_layers = config["n_mm_layers"]
        self.n_heads = 4
        self.dropout = config["dropout"]
        self.reg_weight = config["reg_weight"]
        self.knn_k = config["knn_k"]
        self.lambda_coeff = config["lambda_coeff"]
        
        # User-Item embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight, gain=0.1)
        nn.init.xavier_uniform_(self.item_embedding.weight, gain=0.1)
        
        # Modal encoders
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_encoder = BootstrapEncoder(
                self.v_feat.shape[1], self.hidden_dim, self.feat_embed_dim, self.dropout
            )
            
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_encoder = BootstrapEncoder(
                self.t_feat.shape[1], self.hidden_dim, self.feat_embed_dim, self.dropout
            )
        
        # Graph layers
        self.gat_layers = nn.ModuleList([
            GATConv(
                self.feat_embed_dim, 
                self.feat_embed_dim // self.n_heads,
                heads=self.n_heads,
                dropout=self.dropout,
                concat=True
            ) for _ in range(self.n_layers)
        ])
        
        # Layer norms for each modality path
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(self.feat_embed_dim) for _ in range(self.n_layers)
        ])
        
        # Load interaction data
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        
        # Build graphs
        self.edge_index = None
        self.edge_weight = None
        self.mm_adj = None
        self.build_graph_structure()
        
        self.to(self.device)
        
    def build_graph_structure(self):
        # Build user-item interaction graph
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.interaction_matrix.tolil()
        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()
        
        # Normalize adjacency matrix
        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum + 1e-9, -0.5).flatten()
        d_mat = sp.diags(d_inv)
        norm_adj = d_mat.dot(adj_mat).dot(d_mat)
        
        # Convert to tensor
        norm_adj = norm_adj.tocoo()
        edge_index = torch.LongTensor([norm_adj.row, norm_adj.col]).to(self.device)
        edge_weight = torch.FloatTensor(norm_adj.data).to(self.device)
        
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        
        # Build multimodal graph
        if self.v_feat is not None:
            v_feat = F.normalize(self.v_feat, p=2, dim=1)
            sim_matrix = torch.mm(v_feat, v_feat.t())
            self.mm_adj = self.build_knn_graph(sim_matrix)
            
        if self.t_feat is not None:
            t_feat = F.normalize(self.t_feat, p=2, dim=1)
            sim_matrix = torch.mm(t_feat, t_feat.t())
            if self.mm_adj is None:
                self.mm_adj = self.build_knn_graph(sim_matrix)
            else:
                self.mm_adj = self.lambda_coeff * self.mm_adj + (1 - self.lambda_coeff) * self.build_knn_graph(sim_matrix)
    
    def build_knn_graph(self, sim_matrix):
        values, indices = sim_matrix.topk(k=self.knn_k, dim=1)
        rows = torch.arange(sim_matrix.size(0), device=self.device).repeat_interleave(self.knn_k)
        edge_index = torch.stack([rows, indices.reshape(-1)])
        edge_weight = values.reshape(-1)
        
        # Symmetrize and normalize
        adj = torch.sparse_coo_tensor(
            edge_index, edge_weight,
            [sim_matrix.size(0), sim_matrix.size(0)]
        ).to_dense()
        adj = (adj + adj.t()) / 2
        
        # Convert back to sparse
        adj = adj.to_sparse()
        return adj

    def forward(self):
        # Process modalities with bootstrap learning
        img_online = img_pred = img_target = None
        txt_online = txt_pred = txt_target = None
        
        if self.v_feat is not None:
            img_online, img_pred, img_target = self.image_encoder(self.image_embedding.weight)
            
        if self.t_feat is not None:
            txt_online, txt_pred, txt_target = self.text_encoder(self.text_embedding.weight)
        
        # Graph convolution for modalities
        modal_emb = None
        if img_online is not None and txt_online is not None:
            modal_feat = torch.stack([img_online, txt_online])
            for i, (gat, norm) in enumerate(zip(self.gat_layers, self.layer_norms)):
                modal_feat = modal_feat + F.dropout(
                    norm(gat(modal_feat, self.edge_index)),
                    p=self.dropout,
                    training=self.training
                )
            modal_emb = modal_feat.mean(dim=0)
        elif img_online is not None:
            modal_emb = img_online
        elif txt_online is not None:
            modal_emb = txt_online
            
        # User-item propagation
        x = torch.cat([self.user_embedding.weight, self.item_embedding.weight])
        all_embs = [F.normalize(x, p=2, dim=1)]
        
        for i in range(self.n_layers):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.normalize(torch.sparse.mm(self.mm_adj, x), p=2, dim=1)
            all_embs.append(x)
            
        x = torch.stack(all_embs).mean(dim=0)
        user_emb, item_emb = torch.split(x, [self.n_users, self.n_items])
        
        if modal_emb is not None:
            item_emb = item_emb + modal_emb
            
        return (
            user_emb, item_emb,
            (img_online, img_pred, img_target),
            (txt_online, txt_pred, txt_target)
        )

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        
        user_emb, item_emb, img_features, txt_features = self.forward()
        
        # Basic embeddings
        u_emb = user_emb[users]
        pos_emb = item_emb[pos_items]
        neg_emb = item_emb[neg_items]
        
        # BPR loss
        pos_scores = torch.sum(u_emb * pos_emb, dim=1)
        neg_scores = torch.sum(u_emb * neg_emb, dim=1)
        bpr_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        
        # Bootstrap contrastive loss
        modal_loss = 0.0
        if img_features[0] is not None and txt_features[0] is not None:
            img_online, img_pred, img_target = img_features
            txt_online, txt_pred, txt_target = txt_features
            
            modal_loss = (
                -torch.mean(F.cosine_similarity(img_pred[pos_items], txt_target[pos_items].detach())) +
                -torch.mean(F.cosine_similarity(txt_pred[pos_items], img_target[pos_items].detach()))
            ) / 2
            
        # L2 regularization
        reg_loss = self.reg_weight * (
            torch.norm(u_emb) +
            torch.norm(pos_emb) +
            torch.norm(neg_emb)
        )
        
        return bpr_loss + 0.2 * modal_loss + reg_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_emb, item_emb, _, _ = self.forward()
        scores = torch.matmul(user_emb[user], item_emb.transpose(0, 1))
        return scores