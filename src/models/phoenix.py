# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from torch_geometric.nn import GATConv
from torch_geometric.utils import add_self_loops, remove_self_loops

from common.abstract_recommender import GeneralRecommender


class PHOENIX(GeneralRecommender):
    def __init__(self, config, dataset):
        super(PHOENIX, self).__init__(config, dataset)

        # Hyperparameters
        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.n_layers = config['n_layers']
        self.alpha = config['alpha']  # Contrastive loss weight
        self.dropout_rate = config['dropout_rate']
        self.num_heads = config['num_heads']  # For multi-head attention

        # Build unified graph
        self.build_unified_graph(dataset)

        # Embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)

        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        # Feature transformations
        if self.v_feat is not None:
            self.v_feat = F.normalize(self.v_feat, dim=1)
            self.v_linear = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)
            nn.init.xavier_uniform_(self.v_linear.weight)

        if self.t_feat is not None:
            self.t_feat = F.normalize(self.t_feat, dim=1)
            self.t_linear = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)
            nn.init.xavier_uniform_(self.t_linear.weight)

        # Attention-based GNN layers
        self.gnn_layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.gnn_layers.append(GATConv(self.embedding_dim, self.embedding_dim, heads=self.num_heads, dropout=self.dropout_rate))

        # Modality attention
        self.modality_attention = nn.Linear(self.feat_embed_dim, 1)

        # Loss functions
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.contrastive_loss = nn.CrossEntropyLoss()

        # Device
        self.device = config['device']

    def build_unified_graph(self, dataset):
        # Build user-item interaction graph
        user_item_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.num_nodes = self.n_users + self.n_items

        # Edges
        user_item_edges = np.vstack((user_item_matrix.row, user_item_matrix.col + self.n_users))
        item_user_edges = np.vstack((user_item_matrix.col + self.n_users, user_item_matrix.row))
        self.edge_index = np.hstack((user_item_edges, item_user_edges))

        # Add self-loops
        self.edge_index, _ = add_self_loops(torch.tensor(self.edge_index, dtype=torch.long), num_nodes=self.num_nodes)
        self.edge_index = self.edge_index.to(self.device)

    def forward(self):
        x = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)  # (num_nodes, embedding_dim)

        # Apply GNN layers with attention
        for gnn in self.gnn_layers:
            x = gnn(x, self.edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

        # Split embeddings back into users and items
        user_embeddings = x[:self.n_users]
        item_embeddings = x[self.n_users:]

        # Modality feature fusion with attention
        if self.v_feat is not None and self.t_feat is not None:
            v_emb = self.v_linear(self.v_feat)
            t_emb = self.t_linear(self.t_feat)

            # Modality attention
            v_attn = self.modality_attention(v_emb)
            t_attn = self.modality_attention(t_emb)
            attn_weights = F.softmax(torch.cat([v_attn, t_attn], dim=1), dim=1)

            fused_item_embeddings = attn_weights[:, 0].unsqueeze(1) * v_emb + attn_weights[:, 1].unsqueeze(1) * t_emb
            item_embeddings += fused_item_embeddings

        elif self.v_feat is not None:
            v_emb = self.v_linear(self.v_feat)
            item_embeddings += v_emb

        elif self.t_feat is not None:
            t_emb = self.t_linear(self.t_feat)
            item_embeddings += t_emb

        return user_embeddings, item_embeddings

    def calculate_loss(self, interaction):
        user = interaction[0]
        pos_item = interaction[1]
        neg_item = interaction[2]

        user_embeddings, item_embeddings = self.forward()

        u_emb = user_embeddings[user]
        pos_i_emb = item_embeddings[pos_item]
        neg_i_emb = item_embeddings[neg_item]

        # BPR Loss
        pos_scores = torch.sum(u_emb * pos_i_emb, dim=1)
        neg_scores = torch.sum(u_emb * neg_i_emb, dim=1)
        bpr_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))

        # Contrastive Loss
        # Generate positive and negative pairs for contrastive learning
        # For simplicity, we use embeddings before and after dropout as positive pairs
        u_emb_view1 = F.dropout(u_emb, p=self.dropout_rate, training=True)
        u_emb_view2 = F.dropout(u_emb, p=self.dropout_rate, training=True)

        pos_cosine = F.cosine_similarity(u_emb_view1, u_emb_view2)
        neg_cosine = torch.matmul(u_emb_view1, u_emb_view2.T)

        labels = torch.arange(u_emb.size(0)).to(self.device)
        contrastive_loss = self.contrastive_loss(neg_cosine, labels)

        total_loss = bpr_loss + self.alpha * contrastive_loss

        return total_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_embeddings, item_embeddings = self.forward()
        u_emb = user_embeddings[user]

        scores = torch.matmul(u_emb, item_embeddings.T)
        return scores

