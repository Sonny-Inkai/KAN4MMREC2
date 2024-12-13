# coding: utf-8
import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv
from common.abstract_recommender import GeneralRecommender

class ModalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embed_dim, dropout=0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        self.predictor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
    def forward(self, x):
        h = self.encoder(x)
        return h, self.predictor(h.detach())

class GraphLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.gat = GATConv(in_dim, out_dim // 4, heads=4, concat=True)
        self.gcn = GCNConv(in_dim, out_dim)
        self.gate = nn.Sequential(
            nn.Linear(out_dim * 2, 2),
            nn.Softmax(dim=-1)
        )
        self.norm = nn.LayerNorm(out_dim)
        
    def forward(self, x, edge_index):
        gat_out = self.gat(x, edge_index)
        gcn_out = self.gcn(x, edge_index)
        gate_weights = self.gate(torch.cat([gat_out, gcn_out], dim=1))
        out = gate_weights[:, 0:1] * gat_out + gate_weights[:, 1:2] * gcn_out
        return self.norm(out + x)

class MMGAT(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MMGAT, self).__init__(config, dataset)
        
        # Basic settings
        self.embedding_dim = config["embedding_size"]
        self.feat_embed_dim = config["feat_embed_dim"]
        self.hidden_dim = self.feat_embed_dim * 2
        self.n_ui_layers = config["n_ui_layers"]
        self.n_mm_layers = config["n_mm_layers"]
        self.reg_weight = config["reg_weight"]
        self.dropout = config["dropout"]
        self.knn_k = config["knn_k"]
        
        # Embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_normal_(self.user_embedding.weight, gain=0.1)
        nn.init.xavier_normal_(self.item_embedding.weight, gain=0.1)
        
        # Modal processors
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_encoder = ModalEncoder(
                self.v_feat.shape[1], self.hidden_dim, self.feat_embed_dim, self.dropout
            )
            
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_encoder = ModalEncoder(
                self.t_feat.shape[1], self.hidden_dim, self.feat_embed_dim, self.dropout
            )
        
        # Graph layers
        self.ui_graph_layers = nn.ModuleList()
        self.mm_graph_layers = nn.ModuleList()
        
        for _ in range(max(self.n_ui_layers, self.n_mm_layers)):
            if len(self.ui_graph_layers) < self.n_ui_layers:
                self.ui_graph_layers.append(GraphLayer(self.embedding_dim, self.embedding_dim))
            if len(self.mm_graph_layers) < self.n_mm_layers:
                self.mm_graph_layers.append(GraphLayer(self.feat_embed_dim, self.feat_embed_dim))
        
        # Modal fusion
        self.fusion = nn.Sequential(
            nn.Linear(self.feat_embed_dim * 2, self.feat_embed_dim),
            nn.LayerNorm(self.feat_embed_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.feat_embed_dim, self.feat_embed_dim),
            nn.LayerNorm(self.feat_embed_dim)
        )
        
        # Load data
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.edge_index = self.build_edges()
        self.mm_edge_index = None
        self.build_modal_graph()
        
        self.to(self.device)
        
    def build_edges(self):
        rows = self.interaction_matrix.row
        cols = self.interaction_matrix.col + self.n_users
        edge_index = torch.tensor(np.vstack([
            np.concatenate([rows, cols, cols, rows]),
            np.concatenate([cols, rows, rows, cols])
        ]), dtype=torch.long).to(self.device)
        return edge_index
        
    def build_modal_graph(self):
        if not (self.v_feat is not None or self.t_feat is not None):
            return
            
        features = []
        if self.v_feat is not None:
            features.append(F.normalize(self.v_feat, p=2, dim=1))
        if self.t_feat is not None:
            features.append(F.normalize(self.t_feat, p=2, dim=1))
            
        feat = torch.stack(features).mean(dim=0)
        sim = torch.mm(feat, feat.t())
        
        _, indices = sim.topk(k=self.knn_k, dim=1)
        rows = torch.arange(feat.size(0), device=self.device)
        rows = rows.view(-1, 1).expand_as(indices)
        
        self.mm_edge_index = torch.stack([
            torch.cat([rows.reshape(-1), indices.reshape(-1)]),
            torch.cat([indices.reshape(-1), rows.reshape(-1)])
        ]).to(self.device)
        
    def process_modal_features(self):
        img_feat = txt_feat = None
        img_pred = txt_pred = None
        
        if self.v_feat is not None:
            img_feat, img_pred = self.image_encoder(self.image_embedding.weight)
            for layer in self.mm_graph_layers:
                img_feat = layer(img_feat, self.mm_edge_index)
                
        if self.t_feat is not None:
            txt_feat, txt_pred = self.text_encoder(self.text_embedding.weight)
            for layer in self.mm_graph_layers:
                txt_feat = layer(txt_feat, self.mm_edge_index)
                
        if img_feat is not None and txt_feat is not None:
            modal_feat = self.fusion(torch.cat([img_feat, txt_feat], dim=1))
        else:
            modal_feat = img_feat if img_feat is not None else txt_feat
            
        return modal_feat, (img_feat, txt_feat), (img_pred, txt_pred)
        
    def forward(self):
        # Process modalities
        modal_feat, (img_feat, txt_feat), (img_pred, txt_pred) = self.process_modal_features()
        
        # Process user-item graph
        x = torch.cat([self.user_embedding.weight, self.item_embedding.weight])
        all_embs = [x]
        
        for layer in self.ui_graph_layers:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = layer(x, self.edge_index)
            all_embs.append(x)
            
        x = torch.stack(all_embs, dim=1).mean(dim=1)
        user_emb, item_emb = torch.split(x, [self.n_users, self.n_items])
        
        # Combine with modal features
        if modal_feat is not None:
            item_emb = item_emb + modal_feat
            
        return user_emb, item_emb, (img_feat, txt_feat), (img_pred, txt_pred)

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        
        user_emb, item_emb, (img_feat, txt_feat), (img_pred, txt_pred) = self.forward()
        
        # Basic BPR loss
        u_e = user_emb[users]
        pos_e = item_emb[pos_items]
        neg_e = item_emb[neg_items]
        
        pos_scores = torch.sum(u_e * pos_e, dim=1)
        neg_scores = torch.sum(u_e * neg_e, dim=1)
        bpr_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        
        # Modal contrastive loss
        modal_loss = 0.0
        if img_feat is not None and txt_feat is not None:
            i_feat = F.normalize(img_feat, dim=1)
            t_feat = F.normalize(txt_feat, dim=1)
            i_pred = F.normalize(img_pred, dim=1)
            t_pred = F.normalize(txt_pred, dim=1)
            
            # Bidirectional prediction
            modal_loss = -(
                torch.mean(F.cosine_similarity(i_feat[pos_items], t_pred[pos_items])) +
                torch.mean(F.cosine_similarity(t_feat[pos_items], i_pred[pos_items]))
            ) / 2
        
        # L2 regularization
        reg_loss = self.reg_weight * (
            torch.norm(u_e) +
            torch.norm(pos_e) +
            torch.norm(neg_e)
        )
        
        # Total loss
        total_loss = bpr_loss + 0.2 * modal_loss + reg_loss
        return total_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_emb, item_emb, _, _ = self.forward()
        scores = torch.matmul(user_emb[user], item_emb.transpose(0, 1))
        return scores