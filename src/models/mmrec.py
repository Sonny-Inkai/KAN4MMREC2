# coding: utf-8
# @email: enoche.chow@gmail.com
r"""
MMRec: Multimodal Recommendation with Modality-Aware Graph Neural Networks
"""

import os
import random
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter
import math

from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss, L2Loss
from utils.utils import build_sim, compute_normalized_laplacian


class MMREC(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MMREC, self).__init__(config, dataset)

        self.embedding_dim = config["embedding_size"]
        self.feat_embed_dim = config["feat_embed_dim"]
        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.reg_weight = config["reg_weight"]
        self.dropout = config["dropout"]
        self.modality_agg = config["modality_agg"]

        # Move these initializations up before they're used
        self.modal_specific = config["modal_specific"] if "modal_specific" in config else True
        self.fusion_layer = config["fusion_layer"] if "fusion_layer" in config else "early"  # early, late, hybrid
        self.use_modal_routing = config["use_modal_routing"] if "use_modal_routing" in config else True

        self.n_nodes = self.n_users + self.n_items
        
        # Load dataset info
        self.interaction_matrix = dataset.inter_matrix(form="coo").astype(np.float32)
        self.norm_adj = self.get_norm_adj_mat().to(self.device)

        # Initialize embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        # Initialize feature embeddings and transformations
        self.v_feat_embedding = None
        self.t_feat_embedding = None
        if self.v_feat is not None:
            self.v_feat_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.v_feat_trs = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)
            self.v_attn = MultiHeadAttention(self.feat_embed_dim, self.n_heads)
            
        if self.t_feat is not None:
            self.t_feat_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.t_feat_trs = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)
            self.t_attn = MultiHeadAttention(self.feat_embed_dim, self.n_heads)

        # Modality fusion
        if self.modality_agg == "weighted":
            self.modal_weights = nn.Parameter(torch.ones(2)/2)
        
        # GNN layers
        self.gnn_layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.gnn_layers.append(MMGNNLayer(
                embedding_dim=self.embedding_dim, 
                n_heads=self.n_heads, 
                dropout=self.dropout,
                fusion_layer=self.fusion_layer
            ))

        # Cross-modal contrastive learning
        self.temperature = 0.2
        self.modal_fusion = ModalFusion(self.feat_embed_dim)

        if self.modal_specific:
            self.modal_transform = ModalSpecificTransform(self.feat_embed_dim)
        
        if self.use_modal_routing:
            self.modal_router = ModalRouter(self.feat_embed_dim)

        # Add residual connections
        self.use_residual = True
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(self.embedding_dim) 
            for _ in range(self.n_layers)
        ])

    def get_norm_adj_mat(self):
        A = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        for key, value in data_dict.items():
            A[key] = value
        
        sumArr = (A > 0).sum(axis=1)
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse_coo_tensor(i, data, torch.Size(L.shape))
        return SparseL

    def forward(self):
        ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
        
        # Process modalities with routing
        modal_embeddings = []
        if self.v_feat_embedding is not None:
            v_feat = self.v_feat_trs(self.v_feat_embedding.weight)
            if self.modal_specific:
                v_feat = self.modal_transform(v_feat, 'visual')
            v_feat = self.v_attn(v_feat, v_feat, v_feat)
            v_feat = torch.cat((torch.zeros(self.n_users, self.feat_embed_dim).to(self.device), v_feat), dim=0)
            modal_embeddings.append(v_feat)
            
        if self.t_feat_embedding is not None:
            t_feat = self.t_feat_trs(self.t_feat_embedding.weight)
            if self.modal_specific:
                t_feat = self.modal_transform(t_feat, 'text')
            t_feat = self.t_attn(t_feat, t_feat, t_feat)
            t_feat = torch.cat((torch.zeros(self.n_users, self.feat_embed_dim).to(self.device), t_feat), dim=0)
            modal_embeddings.append(t_feat)

        # Early fusion with modal routing
        if len(modal_embeddings) > 0:
            if self.use_modal_routing:
                fused_modal = self.modal_router(modal_embeddings)
            else:
                if self.modality_agg == "concat":
                    fused_modal = torch.cat(modal_embeddings, dim=1)
                else:
                    fused_modal = self.modal_fusion(modal_embeddings)
            
            if self.fusion_layer == "early":
                ego_embeddings = ego_embeddings + fused_modal
            elif self.fusion_layer == "hybrid":
                ego_embeddings = torch.cat([ego_embeddings, fused_modal], dim=1)

        # Enhanced GNN with residual connections and layer normalization
        all_embeddings = [ego_embeddings]
        for i, gnn in enumerate(self.gnn_layers):
            gnn_out = gnn(ego_embeddings, self.norm_adj)
            if self.use_residual:
                gnn_out = gnn_out + ego_embeddings
            gnn_out = self.layer_norms[i](gnn_out)
            all_embeddings.append(gnn_out)
            ego_embeddings = gnn_out

        # Late fusion
        if len(modal_embeddings) > 0 and self.fusion_layer == "late":
            all_embeddings.append(fused_modal)
            
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        return u_g_embeddings, i_g_embeddings

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        u_g_embeddings, i_g_embeddings = self.forward()
        
        u_embeddings = u_g_embeddings[users]
        pos_embeddings = i_g_embeddings[pos_items]
        neg_embeddings = i_g_embeddings[neg_items]

        # BPR Loss
        pos_scores = torch.sum(u_embeddings * pos_embeddings, dim=1)
        neg_scores = torch.sum(u_embeddings * neg_embeddings, dim=1)
        mf_loss = -torch.mean(torch.log2(torch.sigmoid(pos_scores - neg_scores)))

        # Contrastive Loss for multimodal features
        contrastive_loss = 0.0
        if self.v_feat_embedding is not None and self.t_feat_embedding is not None:
            v_feat = self.v_feat_trs(self.v_feat_embedding.weight[pos_items])
            t_feat = self.t_feat_trs(self.t_feat_embedding.weight[pos_items])
            
            sim_matrix = torch.matmul(v_feat, t_feat.T) / self.temperature
            labels = torch.arange(sim_matrix.size(0)).to(self.device)
            contrastive_loss = F.cross_entropy(sim_matrix, labels) + F.cross_entropy(sim_matrix.T, labels)
            contrastive_loss = contrastive_loss * 0.2

        # L2 regularization
        reg_loss = self.reg_weight * (
            torch.norm(u_embeddings) +
            torch.norm(pos_embeddings) +
            torch.norm(neg_embeddings)
        )

        return mf_loss + reg_loss + contrastive_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]

        u_g_embeddings, i_g_embeddings = self.forward()
        u_embeddings = u_g_embeddings[user]

        scores = torch.matmul(u_embeddings, i_g_embeddings.transpose(0, 1))
        return scores


class MultiHeadAttention(nn.Module):
    def __init__(self, dim_model, num_heads):
        super().__init__()
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.head_dim = dim_model // num_heads
        
        self.q_linear = nn.Linear(dim_model, dim_model)
        self.k_linear = nn.Linear(dim_model, dim_model)
        self.v_linear = nn.Linear(dim_model, dim_model)
        self.out = nn.Linear(dim_model, dim_model)
        
    def forward(self, q, k, v, mask=None):
        # Add batch dimension if not present
        if q.dim() == 2:
            q = q.unsqueeze(0)
            k = k.unsqueeze(0)
            v = v.unsqueeze(0)
            
        batch_size = q.size(0)
        seq_len = q.size(1)
        
        # Linear transformations
        q = self.q_linear(q)
        k = self.k_linear(k)
        v = self.v_linear(v)
        
        # Split into heads
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, v)
        
        # Concatenate heads and put through final linear layer
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, self.dim_model)
        output = self.out(context)
        
        # Remove batch dimension if it was added
        if output.size(0) == 1:
            output = output.squeeze(0)
            
        return output


class MMGNNLayer(nn.Module):
    def __init__(self, embedding_dim, n_heads, dropout, fusion_layer):
        super().__init__()
        self.dropout = dropout
        self.n_heads = n_heads
        
        # For hybrid fusion, input dim will be larger
        self.input_dim = embedding_dim * 2 if fusion_layer == "hybrid" else embedding_dim
        self.output_dim = embedding_dim
        
        # Project input to correct dimension if needed
        self.input_proj = nn.Linear(self.input_dim, embedding_dim) if self.input_dim != embedding_dim else nn.Identity()
        
        self.attentions = nn.ModuleList([
            GATLayer(embedding_dim, embedding_dim//n_heads, dropout=dropout)
            for _ in range(n_heads)
        ])
        self.edge_weight_attention = EdgeWeightAttention(embedding_dim)
        
    def forward(self, x, adj):
        # Project input if dimensions don't match
        x = self.input_proj(x)
        x = F.dropout(x, self.dropout, training=self.training)
        
        # Apply edge weight attention
        adj = self.edge_weight_attention(x, adj)
        
        # Multi-head attention
        out = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        return out


class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0, alpha=0.2):
        super(GATLayer, self).__init__()
        self.dropout = dropout
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.alpha = alpha

        self.W = nn.Parameter(torch.zeros(size=(in_dim, out_dim)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        self.a_src = nn.Parameter(torch.zeros(size=(1, out_dim)))
        self.a_dst = nn.Parameter(torch.zeros(size=(1, out_dim)))
        nn.init.xavier_uniform_(self.a_src.data, gain=1.414)
        nn.init.xavier_uniform_(self.a_dst.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, x, adj):
        h = torch.mm(x, self.W)
        N = h.size()[0]

        e_src = h @ self.a_src.t()
        e_dst = h @ self.a_dst.t()
        e = self.leakyrelu(e_src.expand(-1, N) + e_dst.expand(N, -1).t())

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        h_prime = torch.matmul(attention, h)
        return h_prime


class ModalFusion(nn.Module):
    def __init__(self, dim):
        super(ModalFusion, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(dim, dim//2),
            nn.ReLU(),
            nn.Linear(dim//2, 1)
        )
        
    def forward(self, modal_list):
        if len(modal_list) == 1:
            return modal_list[0]
            
        modal_stack = torch.stack(modal_list, dim=1)
        weights = self.attention(modal_stack)
        weights = F.softmax(weights, dim=1)
        
        fused = torch.sum(modal_stack * weights, dim=1)
        return fused 


class ModalSpecificTransform(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.visual_transform = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        self.text_transform = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(), 
            nn.Linear(dim, dim)
        )
        
    def forward(self, x, modality):
        if modality == 'visual':
            return self.visual_transform(x)
        else:
            return self.text_transform(x)


class ModalRouter(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.routing_weights = Parameter(torch.randn(2, dim))
        self.routing_bias = Parameter(torch.zeros(dim))
        self.temperature = 0.1
        
    def forward(self, modal_list):
        modal_stack = torch.stack(modal_list, dim=1)  # [N, num_modalities, dim]
        
        # Compute routing weights
        routing_logits = torch.matmul(modal_stack, self.routing_weights.t())
        routing_weights = F.softmax(routing_logits / self.temperature, dim=1)
        
        # Route and aggregate
        routed_features = torch.sum(modal_stack * routing_weights.unsqueeze(-1), dim=1)
        return routed_features + self.routing_bias


class EdgeWeightAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, 1)
        )
        
    def forward(self, x, adj):
        N = x.size(0)
        row, col = adj._indices()
        
        # Compute attention scores
        edge_features = torch.cat([x[row], x[col]], dim=1)
        attention_weights = torch.sigmoid(self.attention(edge_features))
        
        # Update adjacency matrix
        new_values = adj._values() * attention_weights.squeeze()
        return torch.sparse.FloatTensor(adj._indices(), new_values, adj.size()) 