# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np

from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss

class EnhancedMMRec(GeneralRecommender):
    def __init__(self, config, dataset):
        super(EnhancedMMRec, self).__init__(config, dataset)
        
        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.n_layers = config['n_layers']
        self.mm_layers = config['n_mm_layers']
        self.dropout = config['dropout']
        self.reg_weight = config['reg_weight']
        self.temp = config.get('temperature', 0.2)
        self.knn_k = config['knn_k']
        self.n_nodes = self.n_users + self.n_items
        
        # Load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.norm_adj = self.get_norm_adj_mat().to(self.device)
        
        # User and item embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)
        
        # Multi-modal feature processors
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_encoder = nn.Sequential(
                nn.Linear(self.v_feat.shape[1], self.feat_embed_dim),
                nn.LayerNorm(self.feat_embed_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout)
            )
            self.image_projector = nn.Linear(self.feat_embed_dim, self.embedding_dim)
            
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_encoder = nn.Sequential(
                nn.Linear(self.t_feat.shape[1], self.feat_embed_dim),
                nn.LayerNorm(self.feat_embed_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout)
            )
            self.text_projector = nn.Linear(self.feat_embed_dim, self.embedding_dim)
        
        # Cross-modal fusion
        self.modal_fusion = nn.MultiheadAttention(
            embed_dim=self.embedding_dim,
            num_heads=4,
            dropout=self.dropout
        )
        
        # Graph attention layers
        self.gat_layers = nn.ModuleList([
            GraphAttentionLayer(self.embedding_dim, self.embedding_dim, dropout=self.dropout)
            for _ in range(self.n_layers)
        ])
        
        # Adaptive aggregation
        self.adaptive_weight = nn.Parameter(torch.ones(3) / 3)  # For CF, visual and textual
        self.softmax = nn.Softmax(dim=0)
        
        # Contrastive learning head
        self.cl_projector = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, self.embedding_dim)
        )
        
    def get_norm_adj_mat(self):
        # Convert to sparse adjacency matrix
        adj_mat = sp.dok_matrix((self.n_nodes, self.n_nodes), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.interaction_matrix.tolil()
        
        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()
        
        # Normalize adjacency matrix
        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        
        norm_adj = d_mat_inv.dot(adj_mat.dot(d_mat_inv))
        return self._sparse_mx_to_torch_sparse_tensor(norm_adj.tocoo())
    
    def _sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col))).long()
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)
    
    def forward(self, adj=None):
        if adj is None:
            adj = self.norm_adj
            
        # Process multi-modal features
        modal_embeds = []
        if self.v_feat is not None:
            image_feats = self.image_encoder(self.image_embedding.weight)
            image_embeds = self.image_projector(image_feats)
            modal_embeds.append(image_embeds)
            
        if self.t_feat is not None:
            text_feats = self.text_encoder(self.text_embedding.weight)
            text_embeds = self.text_projector(text_feats)
            modal_embeds.append(text_embeds)
        
        # Cross-modal fusion with self-attention
        if len(modal_embeds) > 1:
            modal_embeds = torch.stack(modal_embeds, dim=0)
            fused_embeds, _ = self.modal_fusion(
                modal_embeds, modal_embeds, modal_embeds
            )
            item_modal_embeds = torch.mean(fused_embeds, dim=0)
        elif len(modal_embeds) == 1:
            item_modal_embeds = modal_embeds[0]
        else:
            item_modal_embeds = torch.zeros_like(self.item_id_embedding.weight)
        
        # Collaborative filtering path
        ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
        cf_embeddings = [ego_embeddings]
        
        # Multi-layer graph convolution with attention
        for gat in self.gat_layers:
            ego_embeddings = gat(ego_embeddings, adj)
            cf_embeddings.append(ego_embeddings)
            
        cf_embeddings = torch.stack(cf_embeddings, dim=1)
        cf_embeddings = torch.mean(cf_embeddings, dim=1)
        
        # Split user and item embeddings
        u_g_embeddings, i_g_embeddings = torch.split(cf_embeddings, [self.n_users, self.n_items], dim=0)
        
        # Adaptive fusion of different signals
        weights = self.softmax(self.adaptive_weight)
        i_embeddings = weights[0] * i_g_embeddings + weights[1] * item_modal_embeds + weights[2] * self.item_id_embedding.weight
        
        return u_g_embeddings, i_embeddings
    
    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        
        u_embeddings, i_embeddings = self.forward()
        
        # BPR Loss
        u_embeddings = u_embeddings[users]
        pos_embeddings = i_embeddings[pos_items]
        neg_embeddings = i_embeddings[neg_items]
        
        pos_scores = torch.sum(u_embeddings * pos_embeddings, dim=1)
        neg_scores = torch.sum(u_embeddings * neg_embeddings, dim=1)
        bpr_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        
        # Contrastive Learning Loss
        u_cl = self.cl_projector(u_embeddings)
        pos_cl = self.cl_projector(pos_embeddings)
        neg_cl = self.cl_projector(neg_embeddings)
        
        pos_sim = torch.exp(torch.sum(u_cl * pos_cl, dim=1) / self.temp)
        neg_sim = torch.exp(torch.sum(u_cl * neg_cl, dim=1) / self.temp)
        cl_loss = -torch.mean(torch.log(pos_sim / (pos_sim + neg_sim)))
        
        # L2 regularization
        l2_loss = self.reg_weight * (
            torch.norm(u_embeddings) +
            torch.norm(pos_embeddings) +
            torch.norm(neg_embeddings)
        )
        
        return bpr_loss + 0.1 * cl_loss + l2_loss
    
    def full_sort_predict(self, interaction):
        user = interaction[0]
        
        u_embeddings, i_embeddings = self.forward()
        u_embeddings = u_embeddings[user]
        
        scores = torch.matmul(u_embeddings, i_embeddings.transpose(0, 1))
        return scores

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha=0.2):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)
    
    def forward(self, h, adj):
        Wh = torch.mm(h, self.W)
        e = self._prepare_attentional_mechanism_input(Wh)
        
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)
        
        return F.elu(h_prime)
    
    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)