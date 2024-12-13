# coding: utf-8
import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from common.abstract_recommender import GeneralRecommender

class HierarchicalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_levels=3, dropout=0.1):
        super().__init__()
        self.num_levels = num_levels
        dims = [input_dim] + [hidden_dim] * (num_levels-1) + [output_dim]
        
        self.level_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dims[i], dims[i+1]),
                nn.LayerNorm(dims[i+1]),
                nn.GELU(),
                nn.Dropout(dropout)
            ) for i in range(num_levels)
        ])
        
        self.level_attentions = nn.ModuleList([
            nn.MultiheadAttention(dims[i+1], 4, dropout=dropout)
            for i in range(num_levels-1)
        ])
        
        self.predictor = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim)
        )
        
    def forward(self, x):
        level_outputs = []
        current = x
        
        for i in range(self.num_levels):
            current = self.level_encoders[i](current)
            if i < self.num_levels - 1:
                attn_out, _ = self.level_attentions[i](
                    current.unsqueeze(0),
                    current.unsqueeze(0),
                    current.unsqueeze(0)
                )
                current = current + attn_out.squeeze(0)
            level_outputs.append(current)
            
        output = sum(level_outputs) / len(level_outputs)
        return output, self.predictor(output.detach())

class SemanticAggregator(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads)
        self.gru = nn.GRU(dim, dim, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x, adj):
        # Graph propagation with semantic attention
        h = torch.mm(adj, x)
        
        # Self-attention for semantic refinement
        attn_out, _ = self.attention(
            h.unsqueeze(0), h.unsqueeze(0), h.unsqueeze(0)
        )
        h = h + attn_out.squeeze(0)
        
        # Sequential refinement
        h, _ = self.gru(h.unsqueeze(0))
        h = h.squeeze(0)
        
        return self.norm(h + x)

class MMGAT(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MMGAT, self).__init__(config, dataset)
        
        self.embedding_dim = config["embedding_size"]
        self.feat_embed_dim = config["feat_embed_dim"]
        self.hidden_dim = self.feat_embed_dim * 2
        self.n_layers = config["n_mm_layers"]
        self.reg_weight = config["reg_weight"]
        self.dropout = config["dropout"]
        self.knn_k = config["knn_k"]
        
        # Core embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)
        
        # Modal encoders
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_encoder = HierarchicalEncoder(
                self.v_feat.shape[1], self.hidden_dim, self.feat_embed_dim
            )
            
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_encoder = HierarchicalEncoder(
                self.t_feat.shape[1], self.hidden_dim, self.feat_embed_dim
            )
        
        # Semantic aggregators
        self.semantic_layers = nn.ModuleList([
            SemanticAggregator(self.feat_embed_dim)
            for _ in range(self.n_layers)
        ])
        
        # Cross-modal integration
        self.modal_fusion = nn.Sequential(
            nn.Linear(self.feat_embed_dim * 2, self.feat_embed_dim),
            nn.LayerNorm(self.feat_embed_dim),
            nn.GELU()
        )
        
        # Load data
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.norm_adj = self.get_norm_adj_mat().to(self.device)
        
        # Build semantic graphs
        self.modal_adj = None
        self.build_semantic_graphs()
        
        self.to(self.device)
        
    def get_norm_adj_mat(self):
        # Build user-item adjacency matrix
        A = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        for key, value in data_dict.items():
            A[key] = value
        
        # Normalize adjacency matrix
        sumArr = (A > 0).sum(axis=1)
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        L = sp.coo_matrix(L)
        
        indices = np.vstack((L.row, L.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(L.data)
        return torch.sparse_coo_tensor(i, v, L.shape)

    def build_semantic_graphs(self):
        adj_matrices = []
        
        if self.v_feat is not None:
            v_feat = F.normalize(self.v_feat, p=2, dim=1)
            v_sim = torch.mm(v_feat, v_feat.t())
            adj_matrices.append(self.compute_semantic_adj(v_sim))
            
        if self.t_feat is not None:
            t_feat = F.normalize(self.t_feat, p=2, dim=1)
            t_sim = torch.mm(t_feat, t_feat.t())
            adj_matrices.append(self.compute_semantic_adj(t_sim))
        
        if len(adj_matrices) > 0:
            self.modal_adj = sum(adj_matrices) / len(adj_matrices)
        
    def compute_semantic_adj(self, sim_matrix):
        # KNN graph construction
        values, indices = sim_matrix.topk(k=self.knn_k, dim=1)
        rows = torch.arange(sim_matrix.size(0)).view(-1, 1).expand_as(indices)
        adj = torch.zeros_like(sim_matrix)
        adj[rows.reshape(-1), indices.reshape(-1)] = values.reshape(-1)
        
        # Symmetric normalization
        adj = (adj + adj.t()) / 2
        deg = torch.sum(adj, dim=1)
        deg_inv_sqrt = torch.pow(deg + 1e-12, -0.5)
        norm_adj = deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)
        return norm_adj.to(self.device)

    def forward(self):
        # Process modalities with hierarchical encoding
        img_feat = img_pred = txt_feat = txt_pred = None
        
        if self.v_feat is not None:
            img_feat, img_pred = self.image_encoder(self.image_embedding.weight)
            
        if self.t_feat is not None:
            txt_feat, txt_pred = self.text_encoder(self.text_embedding.weight)
        
        # Semantic aggregation
        modal_feat = None
        if img_feat is not None and txt_feat is not None:
            modal_cat = torch.cat([img_feat, txt_feat], dim=1)
            modal_feat = self.modal_fusion(modal_cat)
            
            # Apply semantic aggregation
            for layer in self.semantic_layers:
                modal_feat = layer(modal_feat, self.modal_adj)
        else:
            modal_feat = img_feat if img_feat is not None else txt_feat
            if modal_feat is not None:
                for layer in self.semantic_layers:
                    modal_feat = layer(modal_feat, self.modal_adj)
        
        # User-item propagation
        x = torch.cat([self.user_embedding.weight, self.item_embedding.weight])
        all_embs = [x]
        
        for _ in range(self.n_layers):
            x = torch.sparse.mm(self.norm_adj, x)
            all_embs.append(x)
            
        x = torch.stack(all_embs, dim=1).mean(dim=1)
        user_emb, item_emb = torch.split(x, [self.n_users, self.n_items])
        
        # Combine with modal features
        if modal_feat is not None:
            item_emb = item_emb + modal_feat
            
        return user_emb, item_emb, (img_feat, img_pred), (txt_feat, txt_pred)

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        
        user_emb, item_emb, (img_feat, img_pred), (txt_feat, txt_pred) = self.forward()
        
        u_e = user_emb[users]
        pos_e = item_emb[pos_items]
        neg_e = item_emb[neg_items]
        
        # Hierarchical recommendation loss
        pos_scores = torch.sum(u_e * pos_e, dim=1)
        neg_scores = torch.sum(u_e * neg_e, dim=1)
        rec_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        
        # Semantic contrastive loss
        sem_loss = 0.0
        if img_feat is not None and txt_feat is not None:
            i_feat = F.normalize(img_feat[pos_items])
            t_feat = F.normalize(txt_feat[pos_items])
            i_pred = F.normalize(img_pred[pos_items])
            t_pred = F.normalize(txt_pred[pos_items])
            
            # Bidirectional semantic alignment
            sem_loss = -(
                torch.mean(F.cosine_similarity(i_feat, t_pred.detach())) +
                torch.mean(F.cosine_similarity(t_feat, i_pred.detach()))
            ) / 2
        
        # Regularization
        reg_loss = self.reg_weight * (
            torch.norm(u_e) +
            torch.norm(pos_e) +
            torch.norm(neg_e)
        )
        
        return rec_loss + 0.2 * sem_loss + reg_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_emb, item_emb, _, _ = self.forward()
        scores = torch.matmul(user_emb[user], item_emb.transpose(0, 1))
        return scores