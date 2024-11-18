import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
import scipy.sparse as sp
import numpy as np

# Proposed SOTA Model for Multimodal Recommendation
class PHOENIX(nn.Module):
    def __init__(self, config, dataset):
        super(PHOENIX, self).__init__()
        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.knn_k = config['knn_k']
        self.lambda_coeff = config['lambda_coeff']
        self.cf_model = config['cf_model']
        self.n_layers = config['n_layers']
        self.reg_weight = config['reg_weight']
        self.device = config['device']

        # Load dataset information
        self.n_users = dataset.n_users
        self.n_items = dataset.n_items
        self.n_nodes = self.n_users + self.n_items
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.norm_adj = self.get_norm_adj_mat(self.interaction_matrix).to(self.device)

        # Embedding layers
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        if dataset.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(dataset.v_feat, freeze=False)
            self.image_trs = nn.Linear(dataset.v_feat.shape[1], self.feat_embed_dim)
            nn.init.xavier_normal_(self.image_trs.weight)

        if dataset.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(dataset.t_feat, freeze=False)
            self.text_trs = nn.Linear(dataset.t_feat.shape[1], self.feat_embed_dim)
            nn.init.xavier_normal_(self.text_trs.weight)

        # Graph learning components
        self.graph_layers = nn.ModuleList([GraphLayer(self.embedding_dim) for _ in range(self.n_layers)])
        self.cross_modal_correction = CrossModalCorrection(self.embedding_dim)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.embedding_dim, nhead=8, dropout=config['dropout']),
            num_layers=config['n_layers']
        )

    def get_norm_adj_mat(self, interaction_matrix):
        A = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        inter_M = interaction_matrix
        inter_M_t = interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        for key, value in data_dict.items():
            A[key] = value
        sumArr = (A > 0).sum(axis=1)
        diag = np.array(sumArr.flatten()).astype(np.float32) + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        return torch.sparse.FloatTensor(i, data, torch.Size((self.n_nodes, self.n_nodes)))

    def forward(self, users, items):
        user_emb = self.user_embedding(users)
        item_emb = self.item_embedding(items)

        # Graph learning and propagation
        ego_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        for layer in self.graph_layers:
            ego_embeddings = layer(self.norm_adj, ego_embeddings)
        user_graph_emb, item_graph_emb = torch.split(ego_embeddings, [self.n_users, self.n_items], dim=0)

        # Cross-modal correction mechanism
        user_corrected, item_corrected = self.cross_modal_correction(user_graph_emb, item_graph_emb)

        # Transformer encoder for final user-item representation
        user_final = self.transformer_encoder(user_corrected.unsqueeze(0)).squeeze(0)
        item_final = self.transformer_encoder(item_corrected.unsqueeze(0)).squeeze(0)

        user_out = user_final[users]
        item_out = item_final[items]

        return torch.sum(user_out * item_out, dim=1)

    def calculate_loss(self, interaction):
        users, pos_items, neg_items = interaction
        pos_scores = self.forward(users, pos_items)
        neg_scores = self.forward(users, neg_items)
        loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        reg_loss = self.reg_weight * (
            self.user_embedding.weight.norm(2).pow(2) +
            self.item_embedding.weight.norm(2).pow(2)
        )
        return loss + reg_loss

class GraphLayer(MessagePassing):
    def __init__(self, in_channels):
        super(GraphLayer, self).__init__(aggr='add')
        self.linear = nn.Linear(in_channels, in_channels)
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

class CrossModalCorrection(nn.Module):
    def __init__(self, embedding_dim):
        super(CrossModalCorrection, self).__init__()
        self.transform = nn.Linear(embedding_dim, embedding_dim)
        nn.init.xavier_uniform_(self.transform.weight)

    def forward(self, user_emb, item_emb):
        corrected_user = self.transform(user_emb)
        corrected_item = self.transform(item_emb)
        return corrected_user, corrected_item
