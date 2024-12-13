import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.utils import add_self_loops, degree
from torch_scatter import scatter_add, scatter_mean, scatter_max

from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss

class MMGAT(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MMGAT, self).__init__(config, dataset)
        
        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.n_layers = config['n_layers']
        self.knn_k = config['knn_k']
        self.reg_weight = config['reg_weight']
        self.temp = 0.2
        self.n_nodes = self.n_users + self.n_items
        
        # Graph layers
        self.gat_layers = nn.ModuleList([
            GATConv(self.embedding_dim, self.embedding_dim, heads=4, dropout=0.2)
            for _ in range(self.n_layers)
        ])
        
        # Load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.norm_adj = self.get_norm_adj_mat().to(self.device)
        self.edge_index = self.get_edge_index().to(self.device)
        
        # Embeddings
        self.user_embedding = nn.Parameter(
            torch.normal(mean=0, std=0.1, 
                       size=(self.n_users, self.embedding_dim))
        )
        self.item_id_embedding = nn.Parameter(
            torch.normal(mean=0, std=0.1, 
                       size=(self.n_items, self.embedding_dim))
        )
        
        # Modal-specific components
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_proj = nn.Sequential(
                nn.Linear(self.v_feat.shape[1], self.feat_embed_dim),
                nn.LayerNorm(self.feat_embed_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            )
            self.image_encoder = nn.ModuleList([
                GCNConv(self.feat_embed_dim, self.feat_embed_dim)
                for _ in range(2)
            ])
            
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_proj = nn.Sequential(
                nn.Linear(self.t_feat.shape[1], self.feat_embed_dim),
                nn.LayerNorm(self.feat_embed_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            )
            self.text_encoder = nn.ModuleList([
                GCNConv(self.feat_embed_dim, self.feat_embed_dim)
                for _ in range(2)
            ])
            
        # Cross-modal fusion
        self.modal_fusion = nn.Sequential(
            nn.Linear(self.feat_embed_dim * 2, self.feat_embed_dim),
            nn.LayerNorm(self.feat_embed_dim),
            nn.ReLU()
        )
        
        # Initialize modality weights
        self.modal_weights = nn.Parameter(torch.ones(2) / 2)
        self.softmax = nn.Softmax(dim=0)
        
        # Loss functions
        self.bpr_loss = BPRLoss()
        self.reg_loss = EmbLoss()
        
        # Graph structure
        self.mm_adj = None
        if self.v_feat is not None:
            self.mm_adj = self.build_knn_graph(self.image_embedding.weight)
        if self.t_feat is not None:
            text_adj = self.build_knn_graph(self.text_embedding.weight)
            self.mm_adj = text_adj if self.mm_adj is None else self.mm_adj + text_adj
            
    def build_knn_graph(self, embeddings):
        sim = F.normalize(embeddings, dim=-1) @ F.normalize(embeddings, dim=-1).t()
        topk_values, topk_indices = torch.topk(sim, k=self.knn_k, dim=-1)
        rows = torch.arange(embeddings.size(0)).unsqueeze(1).expand_as(topk_indices)
        adj = torch.zeros_like(sim)
        adj[rows.flatten(), topk_indices.flatten()] = topk_values.flatten()
        return self.compute_normalized_laplacian(adj)
    
    def compute_normalized_laplacian(self, adj):
        deg = adj.sum(dim=1)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        return deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(-2)
        
    def get_norm_adj_mat(self):
        A = sp.dok_matrix((self.n_nodes, self.n_nodes), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users),
                            [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col),
                                [1] * inter_M_t.nnz)))
        A._update(data_dict)
        
        summ = A.sum(axis=1)
        diag = np.array(summ.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D @ A @ D
        L = sp.coo_matrix(L)
        indices = torch.LongTensor(np.array([L.row, L.col]))
        values = torch.FloatTensor(L.data)
        return torch.sparse.FloatTensor(indices, values, torch.Size((self.n_nodes, self.n_nodes)))
    
    def get_edge_index(self):
        rows = torch.from_numpy(self.interaction_matrix.row)
        cols = torch.from_numpy(self.interaction_matrix.col) + self.n_users
        edge_index = torch.stack([
            torch.cat([rows, cols]),
            torch.cat([cols, rows])
        ])
        return edge_index
        
    def forward(self):
        # Process modalities
        modal_embeds = []
        if self.v_feat is not None:
            image_feats = self.image_proj(self.image_embedding.weight)
            for layer in self.image_encoder:
                image_feats = layer(image_feats, self.edge_index)
            modal_embeds.append(image_feats)
            
        if self.t_feat is not None:
            text_feats = self.text_proj(self.text_embedding.weight)
            for layer in self.text_encoder:
                text_feats = layer(text_feats, self.edge_index)
            modal_embeds.append(text_feats)
        
        # Fuse modalities if both present
        if len(modal_embeds) == 2:
            weights = self.softmax(self.modal_weights)
            modal_embeds = [weights[i] * embed for i, embed in enumerate(modal_embeds)]
            item_embeds = self.modal_fusion(torch.cat(modal_embeds, dim=1))
        else:
            item_embeds = modal_embeds[0]
        
        # Process user-item graph
        all_embeds = torch.cat([self.user_embedding, self.item_id_embedding], dim=0)
        embeds_list = [all_embeds]
        
        # Multi-head attention layers
        for gat in self.gat_layers:
            all_embeds = gat(all_embeds, self.edge_index)
            embeds_list.append(F.normalize(all_embeds, p=2, dim=1))
            
        all_embeds = torch.mean(torch.stack(embeds_list, dim=0), dim=0)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeds, [self.n_users, self.n_items])
        
        # Combine with modal embeddings
        i_g_embeddings = i_g_embeddings + item_embeds
        
        return u_g_embeddings, i_g_embeddings
        
    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        
        u_embeddings, i_embeddings = self.forward()
        
        u_embeddings = u_embeddings[users]
        pos_embeddings = i_embeddings[pos_items]
        neg_embeddings = i_embeddings[neg_items]
        
        # BPR Loss
        pos_scores = (u_embeddings * pos_embeddings).sum(dim=1)
        neg_scores = (u_embeddings * neg_embeddings).sum(dim=1)
        bpr_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        
        # Contrastive Loss
        norm_embeddings = F.normalize(i_embeddings, p=2, dim=1)
        pos_sim = F.cosine_similarity(norm_embeddings[pos_items], norm_embeddings[neg_items])
        contrastive_loss = -torch.mean(F.logsigmoid(-pos_sim / self.temp))
        
        # L2 regularization
        reg_loss = self.reg_weight * (
            torch.norm(u_embeddings) +
            torch.norm(pos_embeddings) +
            torch.norm(neg_embeddings)
        )
        
        return bpr_loss + 0.1 * contrastive_loss + reg_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        u_embeddings, i_embeddings = self.forward()
        
        score = torch.matmul(u_embeddings[user], i_embeddings.transpose(0, 1))
        return score