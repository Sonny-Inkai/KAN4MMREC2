# mvcgnn.py
# coding: utf-8
# @email: your_email@example.com

"""
MVCGNN: Multi-View Contrastive Graph Neural Networks for Multimodal Recommendation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree

from common.abstract_recommender import GeneralRecommender


class MVCGNN(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MVCGNN, self).__init__(config, dataset)

        self.embedding_dim = config["embedding_size"]
        self.feat_embed_dim = config["feat_embed_dim"]
        self.n_layers = config["n_layers"]  # Number of GNN layers
        self.reg_weight = config["reg_weight"]
        self.ssl_temp = config["ssl_temp"]
        self.ssl_reg = config["ssl_reg"]
        self.k = config["knn_k"]
        self.device = config["device"]

        # Initialize embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        # Modality-specific embeddings
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(
                self.v_feat, freeze=False
            )
            self.image_trs = nn.Linear(
                self.v_feat.shape[1], self.embedding_dim, bias=False
            )
            nn.init.xavier_uniform_(self.image_trs.weight)
        else:
            raise ValueError("Visual features (v_feat) are required for MVCGNN.")

        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(
                self.t_feat, freeze=False
            )
            self.text_trs = nn.Linear(
                self.t_feat.shape[1], self.embedding_dim, bias=False
            )
            nn.init.xavier_uniform_(self.text_trs.weight)
        else:
            raise ValueError("Textual features (t_feat) are required for MVCGNN.")

        # GNN layers
        self.interaction_gnn_layers = nn.ModuleList()
        self.visual_gnn_layers = nn.ModuleList()
        self.textual_gnn_layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.interaction_gnn_layers.append(LightGCNConv())
            self.visual_gnn_layers.append(LightGCNConv())
            self.textual_gnn_layers.append(LightGCNConv())

        # Fusion layer
        self.fusion_layer = nn.Linear(self.embedding_dim * 3, self.embedding_dim)
        nn.init.xavier_uniform_(self.fusion_layer.weight)

        # Contrastive loss
        self.ssl_criterion = nn.CrossEntropyLoss()

        # Build graphs
        self.inter_edge_index = self.build_interaction_graph(dataset)
        self.visual_edge_index = self.build_modality_graph(
            self.image_trs(self.image_embedding.weight), self.k
        )
        self.textual_edge_index = self.build_modality_graph(
            self.text_trs(self.text_embedding.weight), self.k
        )

    def build_interaction_graph(self, dataset):
        # Build user-item interaction graph
        interactions = dataset.inter_matrix(form="coo").astype(np.int64)
        rows = torch.tensor(interactions.row, dtype=torch.long)
        cols = torch.tensor(interactions.col + self.n_users, dtype=torch.long)
        edge_index = torch.stack(
            [torch.cat([rows, cols]), torch.cat([cols, rows])], dim=0
        )
        return edge_index.to(self.device)

    def build_modality_graph(self, features, k):
        # Build k-NN graph based on feature similarities
        features = F.normalize(features, p=2, dim=1)
        similarity = torch.matmul(features, features.t())
        n_items = features.size(0)
        _, topk_indices = torch.topk(similarity, k=k + 1, dim=1)
        edge_index = []
        for i in range(n_items):
            neighbors = topk_indices[i][1:]  # Exclude self-loop
            for neighbor in neighbors:
                edge_index.append([i, neighbor.item()])
        edge_index = torch.tensor(edge_index).t()
        return edge_index.to(self.device)

    def forward(self):
        # Get embeddings
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight

        # Interaction GNN
        all_embeddings = torch.cat([user_emb, item_emb], dim=0)
        inter_embeddings = [all_embeddings]
        for gnn in self.interaction_gnn_layers:
            all_embeddings = gnn(all_embeddings, self.inter_edge_index)
            inter_embeddings.append(all_embeddings)
        inter_embeddings = torch.stack(inter_embeddings, dim=1)
        inter_embeddings = torch.mean(inter_embeddings, dim=1)
        user_inter_emb, item_inter_emb = (
            inter_embeddings[: self.n_users],
            inter_embeddings[self.n_users :],
        )

        # Visual GNN
        visual_feat = self.image_trs(self.image_embedding.weight)
        visual_embeddings = [visual_feat]
        for gnn in self.visual_gnn_layers:
            visual_feat = gnn(visual_feat, self.visual_edge_index)
            visual_embeddings.append(visual_feat)
        visual_embeddings = torch.stack(visual_embeddings, dim=1)
        visual_embeddings = torch.mean(visual_embeddings, dim=1)
        item_visual_emb = visual_embeddings

        # Textual GNN
        textual_feat = self.text_trs(self.text_embedding.weight)
        textual_embeddings = [textual_feat]
        for gnn in self.textual_gnn_layers:
            textual_feat = gnn(textual_feat, self.textual_edge_index)
            textual_embeddings.append(textual_feat)
        textual_embeddings = torch.stack(textual_embeddings, dim=1)
        textual_embeddings = torch.mean(textual_embeddings, dim=1)
        item_textual_emb = textual_embeddings

        # Fusion
        item_final_emb = self.fusion_layer(
            torch.cat([item_inter_emb, item_visual_emb, item_textual_emb], dim=1)
        )
        item_final_emb = F.normalize(item_final_emb, p=2, dim=1)

        # Normalize user embeddings
        user_final_emb = F.normalize(user_inter_emb, p=2, dim=1)

        return user_final_emb, item_final_emb

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        user_final_emb, item_final_emb = self.forward()

        user_emb = user_final_emb[users]
        pos_item_emb = item_final_emb[pos_items]
        neg_item_emb = item_final_emb[neg_items]

        # BPR Loss
        pos_scores = torch.sum(user_emb * pos_item_emb, dim=1)
        neg_scores = torch.sum(user_emb * neg_item_emb, dim=1)
        mf_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))

        # Regularization
        reg_loss = (
            self.reg_weight
            * (
                user_emb.norm(2).pow(2)
                + pos_item_emb.norm(2).pow(2)
                + neg_item_emb.norm(2).pow(2)
            )
            / 2
        ) / user_emb.shape[0]

        # Contrastive Loss
        ssl_loss = self.ssl_loss(user_final_emb, item_final_emb)

        return mf_loss + reg_loss + self.ssl_reg * ssl_loss

    def ssl_loss(self, user_emb, item_emb):
        # Align embeddings from different modalities
        item_visual_emb = self.image_trs(self.image_embedding.weight)
        item_visual_emb = F.normalize(item_visual_emb, dim=1)

        item_textual_emb = self.text_trs(self.text_embedding.weight)
        item_textual_emb = F.normalize(item_textual_emb, dim=1)

        item_inter_emb = item_emb

        # Contrastive loss between interaction and visual embeddings
        loss_vi = self._info_nce_loss(item_inter_emb, item_visual_emb)
        # Contrastive loss between interaction and textual embeddings
        loss_ti = self._info_nce_loss(item_inter_emb, item_textual_emb)
        # Contrastive loss between visual and textual embeddings
        loss_vt = self._info_nce_loss(item_visual_emb, item_textual_emb)

        return (loss_vi + loss_ti + loss_vt) / 3

    def _info_nce_loss(self, emb1, emb2):
        batch_size = emb1.shape[0]
        emb1 = emb1 / emb1.norm(dim=1, keepdim=True)
        emb2 = emb2 / emb2.norm(dim=1, keepdim=True)
        logits = torch.mm(emb1, emb2.t()) / self.ssl_temp
        labels = torch.arange(batch_size).to(self.device)
        loss = self.ssl_criterion(logits, labels)
        return loss

    def full_sort_predict(self, interaction):
        users = interaction[0]
        user_final_emb, item_final_emb = self.forward()
        user_emb = user_final_emb[users]
        scores = torch.matmul(user_emb, item_final_emb.t())
        return scores

    def predict(self, interaction):
        user = interaction[0]
        item = interaction[1]
        user_final_emb, item_final_emb = self.forward()
        user_emb = user_final_emb[user]
        item_emb = item_final_emb[item]
        scores = torch.sum(user_emb * item_emb, dim=1)
        return scores

    def forward_only_embedding(self):
        # Used for precomputing embeddings if needed
        user_final_emb, item_final_emb = self.forward()
        return user_final_emb, item_final_emb


class LightGCNConv(MessagePassing):
    def __init__(self):
        super(LightGCNConv, self).__init__(aggr="add")

    def forward(self, x, edge_index):
        # Compute normalization
        row, col = edge_index
        deg = degree(row, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        return aggr_out
