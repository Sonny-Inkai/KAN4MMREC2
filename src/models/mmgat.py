# coding: utf-8
import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from common.abstract_recommender import GeneralRecommender

class ModalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        self.gnn = SAGEConv(output_dim, output_dim, aggr='mean')
        self.norm = nn.LayerNorm(output_dim)
        
    def forward(self, x, edge_index):
        x = self.encoder(x)
        x = x + self.gnn(x, edge_index)
        return self.norm(x)

class CrossModalAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
    def forward(self, x1, x2):
        batch_size = x1.size(0)
        q = self.q_proj(x1).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x2).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x2).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        x = torch.matmul(attn, v)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        x = self.out_proj(x)
        return x

class InteractionLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gat = GATConv(dim, dim, heads=4, dropout=0.1, concat=False)
        self.gcn = GCNConv(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, 2),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x, edge_index):
        gat_out = self.gat(x, edge_index)
        gcn_out = self.gcn(x, edge_index)
        gate = self.gate(torch.cat([gat_out, gcn_out], dim=-1))
        x = gate[:, 0:1] * gat_out + gate[:, 1:2] * gcn_out
        return self.norm(x + gat_out)

class MMGAT(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MMGAT, self).__init__(config, dataset)
        
        self.embedding_dim = config["embedding_size"]
        self.feat_embed_dim = config["feat_embed_dim"]
        self.hidden_dim = self.feat_embed_dim * 2
        self.n_layers = config["n_mm_layers"]
        self.n_heads = 4
        self.reg_weight = config["reg_weight"]
        self.dropout = config["dropout"]
        self.knn_k = config["knn_k"]
        
        # User-Item embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_normal_(self.user_embedding.weight)
        nn.init.xavier_normal_(self.item_embedding.weight)
        
        # Modal encoders
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_encoder = ModalEncoder(self.v_feat.shape[1], self.hidden_dim, self.feat_embed_dim)
            
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_encoder = ModalEncoder(self.t_feat.shape[1], self.hidden_dim, self.feat_embed_dim)
        
        # Cross-modal attention
        self.cross_attn = CrossModalAttention(self.feat_embed_dim)
        
        # Graph layers
        self.interaction_layers = nn.ModuleList([
            InteractionLayer(self.embedding_dim) for _ in range(self.n_layers)
        ])
        
        # Modal fusion
        self.fusion = nn.Sequential(
            nn.Linear(self.feat_embed_dim * 2, self.feat_embed_dim),
            nn.LayerNorm(self.feat_embed_dim),
            nn.GELU()
        )
        
        # Load data and build graphs
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.edge_index = self.build_edges()
        self.modal_edge_index = None
        self.build_modal_graph()
        self.to(self.device)

    def build_edges(self):
        rows = self.interaction_matrix.row
        cols = self.interaction_matrix.col + self.n_users
        
        edge_index = torch.tensor(np.vstack([
            np.concatenate([rows, cols, cols, rows]),
            np.concatenate([cols, rows, rows, cols])
        ]), dtype=torch.long).to(self.device)
        
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index)
        return edge_index

    def build_modal_graph(self):
        if not (self.v_feat is not None or self.t_feat is not None):
            return
            
        with torch.no_grad():
            features = []
            if self.v_feat is not None:
                img_feat = F.normalize(self.v_feat, p=2, dim=1)
                features.append(img_feat)
            
            if self.t_feat is not None:
                txt_feat = F.normalize(self.t_feat, p=2, dim=1)
                features.append(txt_feat)
            
            feat = torch.mean(torch.stack(features), dim=0) if len(features) > 1 else features[0]
            sim = torch.mm(feat, feat.t())
            
            # Build KNN graph
            values, indices = sim.topk(k=self.knn_k, dim=1)
            rows = torch.arange(feat.size(0), device=self.device).view(-1, 1).expand_as(indices)
            self.modal_edge_index = torch.stack([rows.reshape(-1), indices.reshape(-1)]).long()
            
            # Add reverse edges and self-loops
            self.modal_edge_index = torch.cat([
                self.modal_edge_index,
                self.modal_edge_index.flip(0)
            ], dim=1)
            self.modal_edge_index, _ = remove_self_loops(self.modal_edge_index)
            self.modal_edge_index, _ = add_self_loops(self.modal_edge_index)

    def forward(self):
        # Process modalities
        img_emb, txt_emb = None, None
        
        if self.v_feat is not None:
            img_emb = self.image_encoder(self.image_embedding.weight, self.modal_edge_index)
            
        if self.t_feat is not None:
            txt_emb = self.text_encoder(self.text_embedding.weight, self.modal_edge_index)
            
        # Cross-modal fusion
        if img_emb is not None and txt_emb is not None:
            img_enhanced = self.cross_attn(img_emb.unsqueeze(0), txt_emb.unsqueeze(0)).squeeze(0)
            txt_enhanced = self.cross_attn(txt_emb.unsqueeze(0), img_emb.unsqueeze(0)).squeeze(0)
            modal_feat = self.fusion(torch.cat([img_enhanced, txt_enhanced], dim=1))
        else:
            modal_feat = img_emb if img_emb is not None else txt_emb
        
        # User-item graph interaction
        x = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        
        for layer in self.interaction_layers:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = layer(x, self.edge_index)
            
        user_emb, item_emb = torch.split(x, [self.n_users, self.n_items])
        
        # Combine with modal features
        if modal_feat is not None:
            item_emb = item_emb + modal_feat
            
        return user_emb, item_emb, (img_emb, txt_emb)

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        
        user_emb, item_emb, (img_emb, txt_emb) = self.forward()
        
        user_e = user_emb[users]
        pos_e = item_emb[pos_items]
        neg_e = item_emb[neg_items]
        
        # Recommendation loss
        pos_scores = torch.sum(user_e * pos_e, dim=1)
        neg_scores = torch.sum(user_e * neg_e, dim=1)
        rec_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        
        # Modal contrastive loss
        modal_loss = 0.0
        if img_emb is not None and txt_emb is not None:
            img_pos = F.normalize(img_emb[pos_items], dim=1)
            txt_pos = F.normalize(txt_emb[pos_items], dim=1)
            img_neg = F.normalize(img_emb[neg_items], dim=1)
            txt_neg = F.normalize(txt_emb[neg_items], dim=1)
            
            pos_sim = torch.sum(img_pos * txt_pos, dim=1)
            neg_sim = torch.sum(img_pos * txt_neg, dim=1)
            modal_loss = -torch.mean(F.logsigmoid(pos_sim - neg_sim))
        
        # L2 regularization
        reg_loss = self.reg_weight * (
            torch.norm(user_e) +
            torch.norm(pos_e) +
            torch.norm(neg_e)
        )
        
        return rec_loss + 0.2 * modal_loss + reg_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_emb, item_emb, _ = self.forward()
        scores = torch.matmul(user_emb[user], item_emb.transpose(0, 1))
        return scores