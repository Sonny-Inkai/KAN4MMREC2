# coding: utf-8
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, degree
from common.abstract_recommender import GeneralRecommender

class MIXED(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MIXED, self).__init__(config, dataset)

        num_user = self.n_users
        num_item = self.n_items
        self.dim_latent = config["embedding_size"]
        self.feat_embed_dim = config["feat_embed_dim"]
        self.n_layers = config["n_mm_layers"]
        self.knn_k = config["knn_k"]
        self.mm_image_weight = config["mm_image_weight"]
        self.reg_weight = config["reg_weight"]
        self.drop_rate = 0.1

        # Embedding layers for modalities
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)

        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)

        # Load or create adjacency matrices
        self.mm_adj = self.load_or_create_mm_adj()

        # Pack training interactions into edge_index
        train_interactions = dataset.inter_matrix(form="coo").astype(np.float32)
        self.edge_index = self.pack_edge_index(train_interactions)

        # Initialize user and item weights
        self.weight_u = nn.Parameter(torch.randn(num_user, 2, 1))
        self.weight_i = nn.Parameter(torch.randn(num_item, 2, 1))

        self.user_graph = User_Graph_sample(num_user, "softmax", self.dim_latent)
        self.result_embed = nn.Parameter(torch.randn(num_user + num_item, self.dim_latent))

    def load_or_create_mm_adj(self):
        dataset_path = os.path.abspath(config["data_path"] + config["dataset"])
        mm_adj_file = os.path.join(dataset_path, "mm_adj_{}.pt".format(self.knn_k))

        if os.path.exists(mm_adj_file):
            return torch.load(mm_adj_file)

        if self.v_feat is not None:
            indices, image_adj = self.get_knn_adj_mat(self.image_embedding.weight.detach())
            mm_adj = image_adj
        if self.t_feat is not None:
            indices, text_adj = self.get_knn_adj_mat(self.text_embedding.weight.detach())
            mm_adj = text_adj
        if self.v_feat is not None and self.t_feat is not None:
            mm_adj = (self.mm_image_weight * image_adj + (1.0 - self.mm_image_weight) * text_adj)

        torch.save(mm_adj, mm_adj_file)
        return mm_adj

    def pack_edge_index(self, inter_mat):
        rows = inter_mat.row
        cols = inter_mat.col + self.n_users
        return np.column_stack((rows, cols))

    def forward(self, interaction):
        user_nodes, pos_item_nodes, neg_item_nodes = interaction
        pos_item_nodes += self.n_users
        neg_item_nodes += self.n_users

        # Obtain representations for each modality
        v_rep = self.get_image_representation()
        t_rep = self.get_text_representation()

        # Combine representations
        representation = self.combine_representations(v_rep, t_rep)

        # User and item embeddings
        user_rep = representation[:self.n_users]
        item_rep = representation[self.n_users:]

        # Compute scores
        user_tensor = user_rep[user_nodes]
        pos_item_tensor = item_rep[pos_item_nodes]
        neg_item_tensor = item_rep[neg_item_nodes]
        pos_scores = torch.sum(user_tensor * pos_item_tensor, dim=1)
        neg_scores = torch.sum(user_tensor * neg_item_tensor, dim=1)
        return pos_scores, neg_scores

    def get_image_representation(self):
        if self.v_feat is not None:
            v_rep = self.image_embedding.weight
            return v_rep
        return None

    def get_text_representation(self):
        if self.t_feat is not None:
            t_rep = self.text_embedding.weight
            return t_rep
        return None

    def combine_representations(self, v_rep, t_rep):
        if v_rep is not None and t_rep is not None:
            return torch.cat((v_rep, t_rep), dim=1)
        elif v_rep is not None:
            return v_rep
        elif t_rep is not None:
            return t_rep
        return None

    def calculate_loss(self, interaction):
        user = interaction[0]
        pos_scores, neg_scores = self.forward(interaction)
        loss_value = -torch.mean(torch.log2(torch.sigmoid(pos_scores - neg_scores)))
        reg_loss = self.reg_weight * (self.weight_u.norm() + self.weight_i.norm())
        return loss_value + reg_loss

    def full_sort_predict(self, interaction):
        representation = self.get_image_representation() + self.get_text_representation()
        user_tensor = representation[:self.n_users]
        item_tensor = representation[self.n_users:]

        score_matrix = torch.matmul(user_tensor, item_tensor.t())
        return score_matrix

    def get_knn_adj_mat(self, mm_embeddings):
        context_norm = mm_embeddings.div(torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True))
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        _, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
        adj_size = sim.size()
        del sim
        indices0 = torch.arange(knn_ind.shape[0]).to(self.device)
        indices0 = torch.unsqueeze(indices0, 1).expand(-1, self.knn_k)
        indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)
        return indices, self.compute_normalized_laplacian(indices, adj_size)

    def compute_normalized_laplacian(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return torch.sparse.FloatTensor(indices, values, adj_size)

class User_Graph_sample(torch.nn.Module):
    def __init__(self, num_user, aggr_mode, dim_latent):
        super(User_Graph_sample, self).__init__()
        self.num_user = num_user
        self.dim_latent = dim_latent
        self.aggr_mode = aggr_mode

    def forward(self, features, user_graph, user_matrix):
        index = user_graph
        u_features = features[index]
        user_matrix = user_matrix.unsqueeze(1)
        u_pre = torch.matmul(user_matrix, u_features)
        return u_pre

class GCN(torch.nn.Module):
    def __init__(self, datasets, batch_size, num_user, num_item, dim_id, aggr_mode, num_layer, has_id, dropout, dim_latent=None, device=None, features=None):
        super(GCN, self).__init__()
        self.batch_size = batch_size
        self.num_user = num_user
        self.num_item = num_item
        self.datasets = datasets
        self.dim_id = dim_id
        self.dim_feat = features.size(1)
        self.dim_latent = dim_latent
        self.aggr_mode = aggr_mode
        self.num_layer = num_layer
        self.has_id = has_id
        self.dropout = dropout
        self.device = device

        if self.dim_latent:
            self.preference = nn.Parameter(nn.init.xavier_normal_(torch.tensor(np.random.randn(num_user, self.dim_latent), dtype=torch.float32, requires_grad=True), gain=1).to(self.device))
            self.MLP = nn.Linear(self.dim_feat, 4 * self.dim_latent)
            self.MLP_1 = nn.Linear(4 * self.dim_latent, self.dim_latent)
            self.conv_embed_1 = Base_gcn(self.dim_latent, self.dim_latent, aggr=self.aggr_mode)
        else:
            self.preference = nn.Parameter(nn.init.xavier_normal_(torch.tensor(np.random.randn(num_user, self.dim_feat), dtype=torch.float32, requires_grad=True), gain=1).to(self.device))
            self.conv_embed_1 = Base_gcn(self.dim_latent, self.dim_latent, aggr=self.aggr_mode)

    def forward(self, edge_index_drop, edge_index, features):
        temp_features = (self.MLP_1(F.leaky_relu(self.MLP(features))) if self.dim_latent else features)
        x = torch.cat((self.preference, temp_features), dim=0).to(self.device)
        x = F.normalize(x).to(self.device)
        h = self.conv_embed_1(x, edge_index)  # equation 1
        h_1 = self.conv_embed_1(h, edge_index)
        x_hat = h + x + h_1
        return x_hat, self.preference

class Base_gcn(MessagePassing):
    def __init__(self, in_channels, out_channels, normalize=True, bias=True, aggr="add", **kwargs):
        super(Base_gcn, self).__init__(aggr=aggr, **kwargs)
        self.aggr = aggr
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x, edge_index, size=None):
        if size is None:
            edge_index, _ = remove_self_loops(edge_index)
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j, edge_index, size):
        if self.aggr == "add":
            row, col = edge_index
            deg = degree(row, size[0], dtype=x_j.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
            return norm.view(-1, 1) * x_j
        return x_j

    def update(self, aggr_out):
        return aggr_out

    def __repr__(self):
        return "{}({},{})".format(self.__class__.__name__, self.in_channels, self.out_channels)
