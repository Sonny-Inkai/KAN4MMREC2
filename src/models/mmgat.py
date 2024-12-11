# coding: utf-8
import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss, L2Loss

class MMGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MMGATLayer, self).__init__()
        self.gat = GATv2Conv(in_dim, out_dim, heads=1, concat=False, dropout=0.1)
        self.norm = nn.LayerNorm(out_dim)
        
    def forward(self, x, edge_index):
        out = self.gat(x, edge_index)
        return self.norm(out)

class MMGAT(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MMGAT, self).__init__(config, dataset)
        
        self.embedding_dim = config["embedding_size"]
        self.feat_embed_dim = config["feat_embed_dim"]
        self.n_layers = config["n_mm_layers"]
        self.reg_weight = config["reg_weight"]
        self.mm_image_weight = config["mm_image_weight"]
        self.lambda_coeff = config["lambda_coeff"]
        self.temperature = 0.2
        self.n_nodes = self.n_users + self.n_items
        
        # Initialize embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        # Load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        
        # Feature processing
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)
            
        # GNN layers
        self.gnn_layers = nn.ModuleList([
            MMGATLayer(self.feat_embed_dim, self.feat_embed_dim)
            for _ in range(self.n_layers)
        ])
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(self.feat_embed_dim * 2, self.feat_embed_dim),
            nn.LayerNorm(self.feat_embed_dim),
            nn.PReLU()
        )
        
        # Move to device
        self.to(self.device)
        
        # Build adjacency matrices
        self._build_adj_matrices()
        
    def _build_adj_matrices(self):
        # Build normalized adjacency for user-item graph
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
        
        # Convert to sparse tensor
        coo = norm_adj.tocoo()
        indices = torch.LongTensor([coo.row, coo.col])
        values = torch.FloatTensor(coo.data)
        self.norm_adj = torch.sparse_coo_tensor(
            indices, values, [self.n_nodes, self.n_nodes]
        ).to(self.device)
        
        # Edge index for gnn
        self.edge_index = self.norm_adj._indices()
        
    def forward(self):
        # Process modal features
        if self.v_feat is not None:
            image_feats = self.image_trs(self.image_embedding.weight)
        if self.t_feat is not None:
            text_feats = self.text_trs(self.text_embedding.weight)
            
        # Feature fusion
        if self.v_feat is not None and self.t_feat is not None:
            x = self.fusion(torch.cat([image_feats, text_feats], dim=1))
        else:
            x = image_feats if self.v_feat is not None else text_feats
            
        # Apply GNN layers
        h = x
        for gnn in self.gnn_layers:
            h_new = gnn(h, self.edge_index)
            h = h + h_new
            
        # User-Item propagation
        ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
        all_embeddings = [ego_embeddings]
        
        for _ in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.norm_adj, ego_embeddings)
            all_embeddings.append(ego_embeddings)
            
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items])
        i_g_embeddings = i_g_embeddings + h
        
        return u_g_embeddings, i_g_embeddings
        
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
        mf_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        
        # Contrastive loss
        if self.v_feat is not None and self.t_feat is not None:
            image_feats = self.image_trs(self.image_embedding.weight)[pos_items]
            text_feats = self.text_trs(self.text_embedding.weight)[pos_items]
            
            contra_loss = -torch.mean(
                torch.sum(F.normalize(image_feats, dim=1) * F.normalize(text_feats, dim=1), dim=1)  
                / self.temperature
            )
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
        scores = torch.matmul(u_embeddings[user], i_embeddings.transpose(0, 1))
        return scores