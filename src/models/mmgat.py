# coding: utf-8
import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from common.abstract_recommender import GeneralRecommender

class ModalGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.gat = GATConv(in_dim, out_dim, heads=1)
        self.norm = nn.LayerNorm(out_dim)
        
    def forward(self, x, edge_index):
        x = self.gat(x, edge_index)
        return self.norm(x)

class MMGAT(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MMGAT, self).__init__(config, dataset)
        
        # Basic parameters
        self.embedding_dim = config["embedding_size"]
        self.n_layers = config["n_mm_layers"]
        self.dropout = config["dropout"]
        self.reg_weight = config["reg_weight"]
        
        # Embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        
        # Image modality
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_fc = nn.Linear(self.v_feat.shape[1], self.embedding_dim)
            self.image_gat = ModalGATLayer(self.embedding_dim, self.embedding_dim)
            
        # Text modality
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_fc = nn.Linear(self.t_feat.shape[1], self.embedding_dim)
            self.text_gat = ModalGATLayer(self.embedding_dim, self.embedding_dim)
            
        # User-Item graph
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.norm_adj = self.build_graph().to(self.device)
        self.to(self.device)
        
    def build_graph(self):
        # Build adjacency matrix
        adj = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj = adj.tolil()
        R = self.interaction_matrix.tolil()
        adj[:self.n_users, self.n_users:] = R
        adj[self.n_users:, :self.n_users] = R.T
        
        # Normalize
        rowsum = np.array(adj.sum(axis=1))
        d_inv = np.power(rowsum + 1e-9, -0.5).flatten()
        d_mat = sp.diags(d_inv)
        norm_adj = d_mat.dot(adj).dot(d_mat)
        
        # Convert to tensor
        norm_adj = norm_adj.tocoo()
        indices = np.vstack((norm_adj.row, norm_adj.col))
        i = torch.LongTensor(indices).to(self.device)
        v = torch.FloatTensor(norm_adj.data).to(self.device)
        shape = torch.Size(norm_adj.shape)
        
        return torch.sparse_coo_tensor(i, v, shape)

    def get_modal_graph(self, features, k=10):
        norm_feat = F.normalize(features, dim=1)
        sim = torch.mm(norm_feat, norm_feat.t())
        _, idx = sim.topk(k=k, dim=1)
        rows = torch.arange(features.size(0), device=self.device).repeat_interleave(k)
        edge_index = torch.stack([rows, idx.flatten()])
        return edge_index

    def forward(self):
        # Process modalities
        image_emb = text_emb = None
        
        if self.v_feat is not None:
            img_feat = self.image_fc(self.image_embedding.weight)
            img_edge = self.get_modal_graph(img_feat)
            image_emb = self.image_gat(img_feat, img_edge)
            
        if self.t_feat is not None:
            txt_feat = self.text_fc(self.text_embedding.weight)
            txt_edge = self.get_modal_graph(txt_feat)
            text_emb = self.text_gat(txt_feat, txt_edge)
            
        # Combine modalities
        if image_emb is not None and text_emb is not None:
            modal_emb = (image_emb + text_emb) / 2
        else:
            modal_emb = image_emb if image_emb is not None else text_emb
            
        # Graph convolution
        ego_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight])
        all_embeddings = [F.normalize(ego_embeddings, p=2, dim=1)]
        
        for _ in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.norm_adj, ego_embeddings)
            all_embeddings.append(F.normalize(ego_embeddings, p=2, dim=1))
            
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        
        user_emb, item_emb = torch.split(all_embeddings, [self.n_users, self.n_items])
        
        if modal_emb is not None:
            item_emb = item_emb + F.normalize(modal_emb, p=2, dim=1)
            
        return user_emb, item_emb

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        
        user_emb, item_emb = self.forward()
        
        u_emb = user_emb[users]
        pos_emb = item_emb[pos_items]
        neg_emb = item_emb[neg_items]
        
        # BPR loss
        pos_scores = torch.sum(u_emb * pos_emb, dim=1)
        neg_scores = torch.sum(u_emb * neg_emb, dim=1)
        mf_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        
        # Regularization
        reg_loss = self.reg_weight * (
            torch.norm(u_emb) +
            torch.norm(pos_emb) +
            torch.norm(neg_emb)
        )
        
        return mf_loss + reg_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_emb, item_emb = self.forward()
        scores = torch.matmul(user_emb[user], item_emb.transpose(0, 1))
        return scores