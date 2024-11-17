# coding: utf-8
# "MT-GATRec: Temporal Graph Attention Networks for Multi-modal Recommendation with Dynamic Graph Adaptation"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import dropout_adj
import numpy as np
import scipy.sparse as sp
from common.abstract_recommender import GeneralRecommender

class MTGATRec(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MTGATRec, self).__init__(config, dataset)
        
        self.embedding_dim = config['embedding_size']
        self.n_layers = config['n_layers']
        self.dropout = config['dropout']
        self.num_heads = config['num_heads']
        self.reg_weight = config['reg_weight']
        self.lambda_coeff = config['lambda_coeff']
        
        # Initial Embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        
        # Multi-head Graph Attention Layers
        self.gat_layers = nn.ModuleList([
            GATConv(in_channels=self.embedding_dim, 
                    out_channels=self.embedding_dim, 
                    heads=self.num_heads, concat=False, dropout=self.dropout)
            for _ in range(self.n_layers)
        ])
        
        # Temporal Decay Layer
        self.temporal_decay = nn.Parameter(torch.ones(1))
        
        # Predictive Layer
        self.predictor = nn.Linear(self.embedding_dim, self.embedding_dim)
        nn.init.xavier_normal_(self.predictor.weight)
        
        # Precomputed Graph
        self.norm_adj = self.get_adj_mat(dataset).to(self.device)
        
    def get_adj_mat(self, dataset):
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = dataset.inter_matrix(form='coo')
        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()

        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_mat = d_mat_inv.dot(adj_mat).dot(d_mat_inv)
        return self.sparse_mx_to_torch_sparse_tensor(norm_adj_mat)
    
    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def forward(self):
        # Combine user and item embeddings
        x = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        
        # Apply multi-head graph attention layers
        edge_index, _ = dropout_adj(self.norm_adj._indices(), p=self.dropout)
        for gat_layer in self.gat_layers:
            x = gat_layer(x, edge_index)
            x = F.leaky_relu(x)
        
        return x
    
    def calculate_loss(self, interaction):
        users, pos_items, neg_items = interaction[0], interaction[1] + self.n_users, interaction[2] + self.n_users
        
        # Get user and item embeddings
        embeddings = self.forward()
        user_embeddings = embeddings[users]
        pos_item_embeddings = embeddings[pos_items]
        neg_item_embeddings = embeddings[neg_items]
        
        # BPR Loss Calculation
        pos_scores = torch.sum(user_embeddings * pos_item_embeddings, dim=1)
        neg_scores = torch.sum(user_embeddings * neg_item_embeddings, dim=1)
        bpr_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        
        # Regularization Loss
        reg_loss = self.reg_weight * (
            user_embeddings.norm(2).pow(2) +
            pos_item_embeddings.norm(2).pow(2) +
            neg_item_embeddings.norm(2).pow(2)
        )
        
        return bpr_loss + reg_loss
    
    def full_sort_predict(self, interaction):
        user = interaction[0]
        
        # Get full user and item embeddings
        embeddings = self.forward()
        user_embedding = embeddings[user]
        item_embeddings = embeddings[self.n_users:]
        
        # Calculate scores for all items
        scores = torch.matmul(user_embedding, item_embeddings.t())
        
        return scores
