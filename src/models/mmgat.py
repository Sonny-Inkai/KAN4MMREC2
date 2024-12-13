import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from common.abstract_recommender import GeneralRecommender

class LaplacianNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        
    def forward(self, x, adj):
        deg = torch.sparse.sum(adj, dim=1).to_dense().view(-1, 1)
        deg_isqrt = torch.pow(deg + self.eps, -0.5)
        adj = torch.sparse.mm(torch.diag(deg_isqrt.view(-1)), adj)
        adj = torch.sparse.mm(adj, torch.diag(deg_isqrt.view(-1)))
        
        mean = torch.sparse.mm(adj, x).mean(dim=1, keepdim=True)
        var = torch.sparse.mm(adj, (x - mean).pow(2)).mean(dim=1, keepdim=True)
        x = (x - mean) / (var + self.eps).sqrt()
        return x * self.scale + self.bias

class StableGATLayer(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.gat = GATConv(dim, dim // heads, heads=heads, dropout=dropout, concat=True)
        self.norm1 = LaplacianNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim)
        )
        
    def forward(self, x, edge_index, edge_weight=None):
        identity = x
        out = self.gat(x, edge_index)
        out = self.norm1(out, self._get_adj(edge_index, out.size(0)))
        out = self.dropout(out) + identity
        out = self.norm2(out)
        out = self.residual(out) + out
        return F.normalize(out, p=2, dim=-1)
    
    def _get_adj(self, edge_index, num_nodes):
        edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
        return torch.sparse_coo_tensor(
            edge_index, edge_weight, 
            (num_nodes, num_nodes)
        )

class ModalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.bn2 = nn.BatchNorm1d(output_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()
        
    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        return F.normalize(x, p=2, dim=1)

class MMGAT(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MMGAT, self).__init__(config, dataset)
        
        self.embedding_dim = config["embedding_size"]
        self.feat_embed_dim = config["feat_embed_dim"]
        self.hidden_dim = self.feat_embed_dim * 2
        self.n_layers = config["n_mm_layers"]
        self.dropout = config["dropout"]
        self.reg_weight = config["reg_weight"]
        self.knn_k = config["knn_k"]
        
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        self.norm_embeddings = nn.LayerNorm(self.embedding_dim)
        
        nn.init.kaiming_normal_(self.user_embedding.weight, mode='fan_out')
        nn.init.kaiming_normal_(self.item_embedding.weight, mode='fan_out')
        
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_encoder = ModalEncoder(self.v_feat.shape[1], self.hidden_dim, self.feat_embed_dim, self.dropout)
            
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_encoder = ModalEncoder(self.t_feat.shape[1], self.hidden_dim, self.feat_embed_dim, self.dropout)
            
        # Weight-sharing between modal layers for better regularization
        self.mm_layer = StableGATLayer(self.feat_embed_dim, dropout=self.dropout)
        self.ui_layer = StableGATLayer(self.embedding_dim, dropout=self.dropout)
        
        if self.v_feat is not None and self.t_feat is not None:
            self.modal_fusion = nn.Sequential(
                nn.Linear(self.feat_embed_dim * 2, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim, self.feat_embed_dim),
                nn.LayerNorm(self.feat_embed_dim)
            )
            self.modal_weight = nn.Parameter(torch.ones(2))
        
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.norm_adj = self.get_norm_adj_mat().to(self.device)
        self.edge_index = self.build_edges()
        self.mm_edge_index = None
        self.build_modal_graph()
        
        self.to(self.device)

    def get_norm_adj_mat(self):
        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))
            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj = d_mat_inv.dot(adj)
            return norm_adj.tocoo()
            
        A = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        for key, value in data_dict.items():
            A[key] = value
        
        norm_adj_mat = normalized_adj_single(A + sp.eye(A.shape[0]))
        return self._convert_sp_mat_to_sp_tensor(norm_adj_mat)

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def build_edges(self):
        rows = self.interaction_matrix.row
        cols = self.interaction_matrix.col + self.n_users
        edge_index = torch.tensor(np.vstack([
            np.concatenate([rows, cols]),
            np.concatenate([cols, rows])
        ]), dtype=torch.long).to(self.device)
        return edge_index

    def build_modal_graph(self):
        if self.v_feat is None and self.t_feat is None:
            return
            
        if self.v_feat is not None and self.t_feat is not None:
            weights = F.softmax(self.modal_weight, dim=0)
            v_sim = F.cosine_similarity(self.v_feat.unsqueeze(1), self.v_feat.unsqueeze(0), dim=2)
            t_sim = F.cosine_similarity(self.t_feat.unsqueeze(1), self.t_feat.unsqueeze(0), dim=2)
            sim = weights[0] * v_sim + weights[1] * t_sim
        else:
            feat = self.v_feat if self.v_feat is not None else self.t_feat
            sim = F.cosine_similarity(feat.unsqueeze(1), feat.unsqueeze(0), dim=2)
        
        topk_values, topk_indices = sim.topk(k=self.knn_k, dim=1)
        rows = torch.arange(sim.size(0), device=self.device).view(-1, 1).expand_as(topk_indices)
        self.mm_edge_weights = F.normalize(topk_values.reshape(-1), p=1, dim=0)
        self.mm_edge_index = torch.stack([
            rows.reshape(-1), topk_indices.reshape(-1)
        ]).to(self.device)

    def forward(self):
        img_emb = txt_emb = None
        
        # Process modalities with gradient scaling
        if self.v_feat is not None:
            img_emb = self.image_encoder(self.image_embedding.weight)
            
        if self.t_feat is not None:
            txt_emb = self.text_encoder(self.text_embedding.weight)
        
        # Multi-layer modal processing with residual connections
        if img_emb is not None and txt_emb is not None:
            weights = F.softmax(self.modal_weight, dim=0)
            modal_emb = torch.stack([img_emb, txt_emb], dim=1)
            modal_emb = (modal_emb * weights.view(1, -1, 1)).sum(1)
            modal_emb = self.modal_fusion(torch.cat([img_emb, txt_emb], dim=1))
        else:
            modal_emb = img_emb if img_emb is not None else txt_emb
            
        if modal_emb is not None:
            for _ in range(self.n_layers):
                modal_emb = self.mm_layer(modal_emb, self.mm_edge_index, self.mm_edge_weights)
        
        # Process user-item graph with message passing
        x = torch.cat([
            self.user_embedding.weight,
            self.item_embedding.weight
        ])
        x = self.norm_embeddings(x)
        
        all_embs = [x]
        for _ in range(self.n_layers):
            x = self.ui_layer(x, self.edge_index)
            all_embs.append(x)
            
        x = torch.stack(all_embs)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x.mean(0)
        
        user_emb, item_emb = torch.split(x, [self.n_users, self.n_items])
        
        if modal_emb is not None:
            item_emb = item_emb + F.normalize(modal_emb, p=2, dim=1)
            
        return user_emb, item_emb, img_emb, txt_emb

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        
        user_emb, item_emb, img_emb, txt_emb = self.forward()
        
        u_emb = user_emb[users]
        pos_emb = item_emb[pos_items]
        neg_emb = item_emb[neg_items]
        
        # InfoNCE loss with temperature scaling
        temp = 0.07
        pos_scores = (u_emb * pos_emb).sum(dim=1) / temp
        neg_scores = (u_emb * neg_emb).sum(dim=1) / temp
        
        loss = -torch.log(
            torch.exp(pos_scores) / 
            (torch.exp(pos_scores) + torch.exp(neg_scores))
        ).mean()
        
        # Modal contrastive loss with adaptive margin
        modal_loss = 0.0
        if img_emb is not None and txt_emb is not None:
            i_emb = F.normalize(img_emb[pos_items], p=2, dim=1)
            t_emb = F.normalize(txt_emb[pos_items], dim=1)
            modal_sim = (i_emb * t_emb).sum(dim=1)
            modal_loss = F.huber_loss(modal_sim, torch.ones_like(modal_sim), reduction='mean')
        
        # L2 regularization with decay
        l2_reg = self.reg_weight * (
            torch.norm(u_emb) +
            torch.norm(pos_emb) +
            torch.norm(neg_emb)
        )
        
        return loss + 0.1 * modal_loss + l2_reg

    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_emb, item_emb, _, _ = self.forward()
        scores = torch.matmul(user_emb[user], item_emb.transpose(0, 1))
        return scores