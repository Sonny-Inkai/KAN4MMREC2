# coding: utf-8
import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from common.abstract_recommender import GeneralRecommender
from torch_geometric.utils import remove_self_loops, add_self_loops, degree
# 0.0260
class ModalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
    def forward(self, x):
        return F.normalize(self.encoder(x), p=2, dim=1)

class GraphAttentionLayer(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.gat = GATConv(dim, dim // 4, heads=4, dropout=dropout)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index):
        return self.norm(x + self.dropout(self.gat(x, edge_index)))

class EnhancedModalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        # Add residual connection
        self.residual = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        
    def forward(self, x):
        h = self.encoder(x)
        x = self.residual(x)
        return F.normalize(h + x, p=2, dim=1)

class EnhancedGATLayer(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.2):
        super().__init__()
        self.gat = GATConv(dim, dim // heads, heads=heads, dropout=dropout)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index):
        h = self.gat(x, edge_index)
        h = self.dropout(h)
        h = self.norm(h)
        return h

class ModalFusionLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, 2),
            nn.Softmax(dim=1)
        )
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
        
    def forward(self, img_emb, txt_emb):
        # Compute attention weights
        combined = torch.cat([img_emb, txt_emb], dim=1)
        weights = self.attention(combined)
        
        # Weighted combination
        fused = weights[:, 0:1] * img_emb + weights[:, 1:2] * txt_emb
        return fused

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
        
        # Add new parameters
        self.n_ui_layers = config.get("n_ui_layers", 2)  # Number of user-item graph layers
        self.lambda_coeff = config.get("lambda_coeff", 0.5) # Weight for modal loss
        
        # User-Item embeddings with better initialization
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight, gain=1.0)
        nn.init.xavier_uniform_(self.item_embedding.weight, gain=1.0)

        # Improved modal encoders with residual connections
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_encoder = EnhancedModalEncoder(self.v_feat.shape[1], 
                                                    self.hidden_dim,
                                                    self.feat_embed_dim, 
                                                    self.dropout)
            
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_encoder = EnhancedModalEncoder(self.t_feat.shape[1],
                                                   self.hidden_dim,
                                                   self.feat_embed_dim,
                                                   self.dropout)

        # Enhanced GAT layers with multi-head attention
        self.mm_layers = nn.ModuleList([
            EnhancedGATLayer(self.feat_embed_dim, heads=4, dropout=self.dropout)
            for _ in range(self.n_layers)
        ])
        
        self.ui_layers = nn.ModuleList([
            EnhancedGATLayer(self.embedding_dim, heads=4, dropout=self.dropout)
            for _ in range(self.n_ui_layers)
        ])

        # Advanced modal fusion with gating mechanism
        if self.v_feat is not None and self.t_feat is not None:
            self.fusion = ModalFusionLayer(self.feat_embed_dim)
        
        # Load and process interaction data
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.edge_index = self.build_edges()
        self.mm_edge_index = None
        self.build_modal_graph()
        
        self.to(self.device)

    def build_edges(self):
        # Enhanced edge building with normalized adjacency matrix
        rows = self.interaction_matrix.row
        cols = self.interaction_matrix.col + self.n_users
        
        edge_index = torch.tensor(np.vstack([
            np.concatenate([rows, cols]),
            np.concatenate([cols, rows])
        ]), dtype=torch.long).to(self.device)
        
        # Add self-loops and normalize
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=self.n_users + self.n_items)
        
        return edge_index

    def build_modal_graph(self):
        if self.v_feat is None and self.t_feat is None:
            return
            
        # Build modal graph with enhanced similarity computation
        if self.v_feat is not None:
            v_feats = F.normalize(self.v_feat, p=2, dim=1)
            v_sim = torch.mm(v_feats, v_feats.t())
            v_values, v_indices = v_sim.topk(k=self.knn_k, dim=1)
            
        if self.t_feat is not None:
            t_feats = F.normalize(self.t_feat, p=2, dim=1)
            t_sim = torch.mm(t_feats, t_feats.t())
            t_values, t_indices = t_sim.topk(k=self.knn_k, dim=1)
        
        # Combine modalities with adaptive weighting
        if self.v_feat is not None and self.t_feat is not None:
            # Adaptive weighting based on feature quality
            v_weight = torch.sigmoid(torch.mean(v_values))
            t_weight = torch.sigmoid(torch.mean(t_values))
            total = v_weight + t_weight
            v_weight = v_weight / total
            t_weight = t_weight / total
            
            indices = v_weight * v_indices + t_weight * t_indices
        else:
            indices = v_indices if self.v_feat is not None else t_indices
            
        rows = torch.arange(indices.size(0), device=self.device).view(-1, 1).expand_as(indices)
        
        self.mm_edge_index = torch.stack([
            torch.cat([rows.reshape(-1), indices.reshape(-1)]),
            torch.cat([indices.reshape(-1), rows.reshape(-1)])
        ]).to(self.device)

    def forward(self):
        # Enhanced forward pass with residual connections and layer normalization
        img_emb = txt_emb = None
        
        if self.v_feat is not None:
            img_emb = self.image_encoder(self.image_embedding.weight)
            
        if self.t_feat is not None:
            txt_emb = self.text_encoder(self.text_embedding.weight)
        
        # Modal fusion with attention
        if img_emb is not None and txt_emb is not None:
            modal_emb = self.fusion(img_emb, txt_emb)
        else:
            modal_emb = img_emb if img_emb is not None else txt_emb
            
        # Process modalities through GAT layers
        if modal_emb is not None:
            h = modal_emb
            for layer in self.mm_layers:
                h_new = layer(h, self.mm_edge_index)
                h = h + h_new  # Residual connection
                h = F.layer_norm(h, h.size()[1:])
            modal_emb = h
            
        # Process user-item graph
        x = torch.cat([self.user_embedding.weight, self.item_embedding.weight])
        all_embs = [x]
        
        for layer in self.ui_layers:
            if self.training:
                x = F.dropout(x, p=self.dropout)
            x_new = layer(x, self.edge_index)
            x = x + x_new  # Residual connection
            x = F.layer_norm(x, x.size()[1:])
            all_embs.append(x)
            
        x = torch.stack(all_embs, dim=1).mean(dim=1)
        user_emb, item_emb = torch.split(x, [self.n_users, self.n_items])
        
        # Combine with modal embeddings using gating
        if modal_emb is not None:
            gate = torch.sigmoid(self.fusion.gate(item_emb, modal_emb))
            item_emb = gate * item_emb + (1 - gate) * modal_emb
            
        return user_emb, item_emb, img_emb, txt_emb

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        
        user_emb, item_emb, img_emb, txt_emb = self.forward()
        
        u_emb = user_emb[users]
        pos_emb = item_emb[pos_items]
        neg_emb = item_emb[neg_items]
        
        # BPR loss
        pos_scores = torch.sum(u_emb * pos_emb, dim=1)
        neg_scores = torch.sum(u_emb * neg_emb, dim=1)
        mf_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        
        # Modal contrastive loss
        modal_loss = 0.0
        if img_emb is not None and txt_emb is not None:
            pos_sim = F.cosine_similarity(img_emb[pos_items], txt_emb[pos_items])
            neg_sim = F.cosine_similarity(img_emb[pos_items], txt_emb[neg_items])
            modal_loss = -torch.mean(F.logsigmoid(pos_sim - neg_sim))
        
        # Regularization loss
        reg_loss = self.reg_weight * (
            torch.norm(u_emb) +
            torch.norm(pos_emb) +
            torch.norm(neg_emb)
        )
        
        return mf_loss + 0.1 * modal_loss + reg_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_emb, item_emb, _, _ = self.forward()
        scores = torch.matmul(user_emb[user], item_emb.transpose(0, 1))
        return scores