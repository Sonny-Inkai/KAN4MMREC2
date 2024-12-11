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
from common.loss import BPRLoss, EmbLoss, L2Loss

class ModalPredictor(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(ModalPredictor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, in_dim)
        )

    def forward(self, x):
        return self.net(x)

class MMGATLayer(nn.Module):
    def __init__(self, dim, heads=4):
        super(MMGATLayer, self).__init__()
        self.gat = GATConv(dim, dim // heads, heads=heads, concat=True)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.PReLU()
        
    def forward(self, x, edge_index):
        res = x
        out = self.gat(x, edge_index)
        out = self.norm(res + out)
        return self.act(out)

class MMGAT(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MMGAT, self).__init__(config, dataset)
        
        self.embedding_dim = config["embedding_size"]
        self.feat_embed_dim = config["feat_embed_dim"]
        self.n_layers = config["n_mm_layers"]
        self.reg_weight = config["reg_weight"]
        self.dropout = config["dropout"]
        self.knn_k = config["knn_k"]
        self.n_nodes = self.n_users + self.n_items
        self.temperature = 0.2

        # Load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        
        # Basic embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)
        
        # Modality processing
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_proj = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)
            self.image_predictor = ModalPredictor(self.feat_embed_dim, self.feat_embed_dim * 2)
            self.image_gat = nn.ModuleList([
                MMGATLayer(self.feat_embed_dim) for _ in range(self.n_layers)
            ])
            
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_proj = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)
            self.text_predictor = ModalPredictor(self.feat_embed_dim, self.feat_embed_dim * 2)
            self.text_gat = nn.ModuleList([
                MMGATLayer(self.feat_embed_dim) for _ in range(self.n_layers)
            ])
        
        self.dropout = nn.Dropout(config['dropout'])
        self.to(self.device)
        self.build_graph()

    def build_graph(self):
        # User-item interaction graph
        ui_adj = sp.dok_matrix((self.n_nodes, self.n_nodes), dtype=np.float32)
        ui_adj = ui_adj.tolil()
        R = self.interaction_matrix.tolil()
        ui_adj[:self.n_users, self.n_users:] = R
        ui_adj[self.n_users:, :self.n_users] = R.T
        ui_adj = ui_adj.todok()
        
        # Normalize adjacency matrix
        rowsum = np.array(ui_adj.sum(axis=1))
        d_inv = np.power(rowsum + 1e-9, -0.5).flatten()
        d_mat = sp.diags(d_inv)
        norm_adj = d_mat.dot(ui_adj).dot(d_mat)
        
        # Convert to sparse tensor
        coo = norm_adj.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.FloatTensor(coo.data)
        self.ui_adj = torch.sparse_coo_tensor(i, v, coo.shape).to(self.device)
        
        # Build modal graphs
        if self.v_feat is not None:
            self.image_edge_index = self.build_knn_graph(self.v_feat)
        if self.t_feat is not None:
            self.text_edge_index = self.build_knn_graph(self.t_feat)

    def build_knn_graph(self, features):
        sim = torch.mm(F.normalize(features, dim=1), F.normalize(features, dim=1).t())
        _, indices = torch.topk(sim, k=self.knn_k, dim=1)
        rows = torch.arange(features.size(0), device=self.device).repeat_interleave(self.knn_k)
        cols = indices.reshape(-1)
        edge_index = torch.stack([rows, cols])
        return edge_index.to(self.device)

    def process_modal(self, x, edge_index, proj, predictor, gat_layers):
        online_x = proj(x)
        
        with torch.no_grad():
            target_x = online_x.clone().detach()
            target_x = F.dropout(target_x, p=self.dropout, training=self.training)
            
        for gat in gat_layers:
            online_x = gat(online_x, edge_index)
            target_x = gat(target_x, edge_index)
            
        online_pred = predictor(online_x)
        return online_x, online_pred, target_x

    def forward(self):
        image_feat, text_feat = None, None
        image_pred, text_pred = None, None
        image_target, text_target = None, None
        
        if self.v_feat is not None:
            image_feat, image_pred, image_target = self.process_modal(
                self.image_embedding.weight,
                self.image_edge_index,
                self.image_proj,
                self.image_predictor,
                self.image_gat
            )
            
        if self.t_feat is not None:
            text_feat, text_pred, text_target = self.process_modal(
                self.text_embedding.weight,
                self.text_edge_index,
                self.text_proj,
                self.text_predictor,
                self.text_gat
            )

        # Combine modal features
        if image_feat is not None and text_feat is not None:
            item_modal = (image_feat + text_feat) / 2
        else:
            item_modal = image_feat if image_feat is not None else text_feat

        # User-item propagation
        ego_embeddings = torch.cat([self.user_embedding.weight, self.item_id_embedding.weight], dim=0)
        embeddings_list = [ego_embeddings]
        
        for _ in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.ui_adj, ego_embeddings)
            embeddings_list.append(ego_embeddings)
            
        all_embeddings = torch.stack(embeddings_list, dim=1).mean(dim=1)
        user_all, item_all = torch.split(all_embeddings, [self.n_users, self.n_items])
        
        item_all = item_all + item_modal
        return (user_all, item_all, image_pred, text_pred, image_target, text_target)

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        user_all, item_all, image_pred, text_pred, image_target, text_target = self.forward()
        
        user_e = user_all[users]
        pos_e = item_all[pos_items]
        neg_e = item_all[neg_items]

        # BPR Loss
        pos_scores = torch.sum(user_e * pos_e, dim=1)
        neg_scores = torch.sum(user_e * neg_e, dim=1)
        bpr_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))

        # Bootstrap Loss
        modal_loss = 0.0
        if image_pred is not None and text_pred is not None:
            modal_loss = (
                F.mse_loss(image_pred[pos_items], text_target[pos_items].detach()) +
                F.mse_loss(text_pred[pos_items], image_target[pos_items].detach())
            )

        # Regularization Loss
        reg_loss = self.reg_weight * (
            torch.norm(user_e) +
            torch.norm(pos_e) +
            torch.norm(neg_e)
        )

        return bpr_loss + 0.2 * modal_loss + reg_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_all, item_all, _, _, _, _ = self.forward()
        scores = torch.matmul(user_all[user], item_all.transpose(0, 1))
        return scores