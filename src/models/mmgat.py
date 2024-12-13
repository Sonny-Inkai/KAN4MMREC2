# coding: utf-8
import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from common.abstract_recommender import GeneralRecommender

class ModalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        return self.encoder(x)

class EfficientGraphAttention(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.1):
        super().__init__()
        self.gat = GATConv(dim, dim // heads, heads=heads, dropout=dropout)
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index):
        # GAT layer with residual
        h = self.gat(x, edge_index)
        x = self.norm(x + self.dropout(h))
        
        # FFN with residual
        h = self.ffn(x)
        x = self.norm(x + self.dropout(h))
        
        return x

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
        
        # Initialize embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight, gain=1.0)
        nn.init.xavier_uniform_(self.item_embedding.weight, gain=1.0)
        
        # Modal encoders
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_encoder = ModalEncoder(self.v_feat.shape[1], self.hidden_dim, self.embedding_dim, self.dropout)
            
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_encoder = ModalEncoder(self.t_feat.shape[1], self.hidden_dim, self.embedding_dim, self.dropout)
            
        # Graph attention layers with memory efficiency
        self.mm_layers = nn.ModuleList([
            EfficientGraphAttention(self.embedding_dim, dropout=self.dropout)
            for _ in range(self.n_layers)
        ])
        
        self.ui_layers = nn.ModuleList([
            EfficientGraphAttention(self.embedding_dim, dropout=self.dropout)
            for _ in range(self.n_layers)
        ])
        
        # Lightweight modal fusion
        if self.v_feat is not None and self.t_feat is not None:
            self.modal_gate = nn.Sequential(
                nn.Linear(self.embedding_dim * 2, 2),
                nn.Softmax(dim=-1)
            )
        
        # Load and process interaction data
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.edge_index = self.build_edges()
        self.mm_edge_index = None
        self.build_modal_graph()
        
        self.apply(self._init_weights)
        self.to(self.device)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=1.0)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
                
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
            
        # Build KNN graph for modalities
        if self.v_feat is not None:
            feats = F.normalize(self.v_feat, p=2, dim=1)
        elif self.t_feat is not None:
            feats = F.normalize(self.t_feat, p=2, dim=1)
            
        with torch.no_grad():
            sim = torch.mm(feats, feats.t())
            values, indices = sim.topk(k=self.knn_k, dim=1)
            rows = torch.arange(feats.size(0), device=self.device).view(-1, 1).expand_as(indices)
            
            self.mm_edge_index = torch.stack([
                torch.cat([rows.reshape(-1), indices.reshape(-1)]),
                torch.cat([indices.reshape(-1), rows.reshape(-1)])
            ]).to(self.device)
        
        del sim  # Free memory

    def forward(self):
        # Process modalities
        img_emb = txt_emb = None
        
        if self.v_feat is not None:
            img_emb = self.image_encoder(self.image_embedding.weight)
            
        if self.t_feat is not None:
            txt_emb = self.text_encoder(self.text_embedding.weight)
        
        # Efficient modal fusion
        if img_emb is not None and txt_emb is not None:
            concat_emb = torch.cat([img_emb, txt_emb], dim=-1)
            gates = self.modal_gate(concat_emb)
            modal_emb = gates[:, 0:1] * img_emb + gates[:, 1:2] * txt_emb
        else:
            modal_emb = img_emb if img_emb is not None else txt_emb
        
        # Process user-item graph
        x = torch.cat([self.user_embedding.weight, self.item_embedding.weight])
        all_embs = [x]
        
        # Layer-wise propagation with gradient checkpointing
        for layer in self.ui_layers:
            if self.training:
                x = F.dropout(x, p=self.dropout, training=True)
            x = layer(x, self.edge_index)
            all_embs.append(x)
            
        # Memory-efficient aggregation
        x = torch.stack(all_embs).mean(dim=0)
        user_emb, item_emb = torch.split(x, [self.n_users, self.n_items])
        
        # Combine with modal embeddings
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
        
        # BPR loss with temperature scaling
        temperature = 0.2
        pos_scores = torch.sum(u_emb * pos_emb, dim=1) / temperature
        neg_scores = torch.sum(u_emb * neg_emb, dim=1) / temperature
        bpr_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        
        # Memory-efficient modal contrastive loss
        modal_loss = 0.0
        if img_emb is not None and txt_emb is not None:
            pos_sim = F.cosine_similarity(img_emb[pos_items], txt_emb[pos_items])
            neg_sim = F.cosine_similarity(img_emb[pos_items], txt_emb[neg_items])
            modal_loss = -torch.mean(F.logsigmoid(pos_sim - neg_sim))
        
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