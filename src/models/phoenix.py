import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree, remove_self_loops

from common.abstract_recommender import GeneralRecommender


class PHOENIX(GeneralRecommender):
    def __init__(self, config, dataset):
        super(PHOENIX, self).__init__(config, dataset)

        # Model configurations
        self.embedding_dim = config["embedding_size"]
        self.feat_embed_dim = config["feat_embed_dim"]
        self.n_layers = config["n_layers"]
        self.knn_k = config["knn_k"]
        self.reg_weight = config["reg_weight"]
        self.attention_heads = config["attention_heads"]
        self.dropout = config["dropout"]
        self.device = config["device"]

        # Dataset attributes
        self.interaction_matrix = dataset.inter_matrix(form="coo").astype(np.float32)
        self.norm_adj = self.get_norm_adj_mat().to(self.device)

        # Embedding layers
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        if dataset.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(dataset.v_feat, freeze=False)
            self.image_transform = nn.Linear(dataset.v_feat.shape[1], self.feat_embed_dim)
            nn.init.xavier_normal_(self.image_transform.weight)

        if dataset.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(dataset.t_feat, freeze=False)
            self.text_transform = nn.Linear(dataset.t_feat.shape[1], self.feat_embed_dim)
            nn.init.xavier_normal_(self.text_transform.weight)

        # Graph Learning Layers
        self.graph_layers = nn.ModuleList([GraphLayer(self.embedding_dim) for _ in range(self.n_layers)])
        self.attention_layer = MultiHeadAttention(self.embedding_dim, self.attention_heads)

    def get_norm_adj_mat(self):
        A = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        for key, value in data_dict:
            A[key] = value
 
        sumArr = (A > 0).sum(axis=1)
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D @ A @ D
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)

        return torch.sparse.FloatTensor(i, data, torch.Size((self.n_users + self.n_items, self.n_users + self.n_items)))

    def forward(self, interaction):
        user_nodes, pos_item_nodes, neg_item_nodes = interaction[0], interaction[1], interaction[2]
        pos_item_nodes += self.n_users
        neg_item_nodes += self.n_users

        # Initial Embedding
        user_emb = self.user_embedding(user_nodes)
        pos_item_emb = self.item_embedding(pos_item_nodes)
        neg_item_emb = self.item_embedding(neg_item_nodes)

        # Graph Propagation
        ego_embeddings = torch.cat((self.user_embedding.weight, self.item_embedding.weight), dim=0)
        all_embeddings = [ego_embeddings]

        for layer in self.graph_layers:
            ego_embeddings = layer(self.norm_adj, ego_embeddings)
            all_embeddings.append(ego_embeddings)

        all_embeddings = torch.stack(all_embeddings, dim=1).mean(dim=1)
        user_graph_emb, item_graph_emb = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)

        # Attention Fusion
        user_out = self.attention_layer(user_emb, user_graph_emb)
        pos_item_out = self.attention_layer(pos_item_emb, item_graph_emb)
        neg_item_out = self.attention_layer(neg_item_emb, item_graph_emb)

        return user_out, pos_item_out, neg_item_out

    def calculate_loss(self, interaction):
        users, pos_items, neg_items = interaction[0], interaction[1], interaction[2]
        user_out, pos_out, neg_out = self.forward(interaction)

        pos_scores = torch.sum(user_out * pos_out, dim=1)
        neg_scores = torch.sum(user_out * neg_out, dim=1)

        mf_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        reg_loss = self.reg_weight * (self.user_embedding.weight.norm(2).pow(2) + self.item_embedding.weight.norm(2).pow(2))

        return mf_loss + reg_loss

    def full_sort_predict(self, interaction):
        user_nodes = interaction[0]
        user_emb = self.user_embedding(user_nodes)

        all_embeddings = torch.cat((self.user_embedding.weight, self.item_embedding.weight), dim=0)
        for layer in self.graph_layers:
            all_embeddings = layer(self.norm_adj, all_embeddings)

        user_graph_emb, item_graph_emb = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        user_out = self.attention_layer(user_emb, user_graph_emb)

        scores = torch.matmul(user_out, item_graph_emb.t())
        return scores


class GraphLayer(MessagePassing):
    def __init__(self, embedding_dim):
        super(GraphLayer, self).__init__(aggr="add")
        self.linear = nn.Linear(embedding_dim, embedding_dim)
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, edge_index, x):
        x = self.linear(x)
        return self.propagate(edge_index, x=x)

    def message(self, x_j, edge_index, size):
        row, col = edge_index
        deg = degree(row, size[0], dtype=x_j.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        return norm.view(-1, 1) * x_j


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, heads):
        super(MultiHeadAttention, self).__init__()
        self.heads = heads
        self.attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=heads, dropout=0.1)

    def forward(self, query, key):
        query = query.unsqueeze(0)  # Add batch dimension
        key = key.unsqueeze(0)
        attn_output, _ = self.attention(query, key, key)
        return attn_output.squeeze(0)  # Remove batch dimension
