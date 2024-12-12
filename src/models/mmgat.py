import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss

class MMGAT(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MMGAT, self).__init__(config, dataset)
        
        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.n_layers = config['n_layers']
        self.knn_k = config['knn_k']
        self.reg_weight = config['reg_weight']
        self.dropout = config['dropout']
        self.temperature = config['temperature']
        self.n_heads = config['n_heads']
        self.ssl_weight = config['ssl_weight']
        self.modal_fusion = config['modal_fusion']
        
        # User-Item interaction graph
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.norm_adj = self.get_norm_adj_mat().to(self.device)
        
        # Embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        # Modal-specific components
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)
            self.image_attention = MultiHeadAttention(self.feat_embed_dim, self.n_heads)
            
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)
            self.text_attention = MultiHeadAttention(self.feat_embed_dim, self.n_heads)
        
        # Cross-modal fusion
        if self.modal_fusion == 'gate':
            self.modal_gate = ModalGate(self.feat_embed_dim)
        elif self.modal_fusion == 'concat':
            self.modal_proj = nn.Linear(self.feat_embed_dim * 2, self.feat_embed_dim)
            
        # Graph convolution layers
        self.gc_layers = nn.ModuleList([
            GraphConvolution(self.embedding_dim) for _ in range(self.n_layers)
        ])
        
        # Contrastive learning projector
        self.projector = nn.Sequential(
            nn.Linear(self.feat_embed_dim, self.feat_embed_dim),
            nn.ReLU(),
            nn.Linear(self.feat_embed_dim, self.feat_embed_dim)
        )
        
    def get_norm_adj_mat(self):
        # Convert to sparse adjacency matrix
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.interaction_matrix.tolil()
        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()
        
        # Normalize adjacency matrix
        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj = d_mat_inv.dot(adj_mat).dot(d_mat_inv)
        
        # Convert to torch sparse tensor
        norm_adj = norm_adj.tocoo()
        indices = torch.LongTensor([norm_adj.row, norm_adj.col])
        values = torch.FloatTensor(norm_adj.data)
        shape = torch.Size(norm_adj.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def forward(self, users, items=None, training=True):
        # Get modal embeddings
        modal_emb = None
        if self.v_feat is not None and self.t_feat is not None:
            image_feat = self.image_trs(self.image_embedding.weight)
            text_feat = self.text_trs(self.text_embedding.weight)
            
            # Apply self-attention on each modality
            image_feat = self.image_attention(image_feat)
            text_feat = self.text_attention(text_feat)
            
            # Modal fusion
            if self.modal_fusion == 'gate':
                modal_emb = self.modal_gate(image_feat, text_feat)
            elif self.modal_fusion == 'concat':
                modal_emb = self.modal_proj(torch.cat([image_feat, text_feat], dim=-1))
            
        # Graph convolution
        user_emb = self.user_embedding.weight
        item_emb = self.item_id_embedding.weight
        
        if modal_emb is not None:
            item_emb = item_emb + modal_emb
            
        all_emb = torch.cat([user_emb, item_emb])
        embs = [all_emb]
        
        for gc_layer in self.gc_layers:
            all_emb = gc_layer(self.norm_adj, all_emb)
            if training:
                all_emb = F.dropout(all_emb, p=self.dropout)
            embs.append(all_emb)
            
        embs = torch.stack(embs, dim=1)
        embs = torch.mean(embs, dim=1)
        
        user_emb, item_emb = torch.split(embs, [self.n_users, self.n_items])
        
        if items is not None:
            user_emb = user_emb[users]
            item_emb = item_emb[items]
            
        return user_emb, item_emb, modal_emb

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        
        user_emb, all_item_emb, modal_emb = self.forward(users)
        pos_emb = all_item_emb[pos_items]
        neg_emb = all_item_emb[neg_items]
        
        # BPR loss
        pos_scores = torch.sum(user_emb * pos_emb, dim=1)
        neg_scores = torch.sum(user_emb * neg_emb, dim=1)
        bpr_loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores)))
        
        # Contrastive loss for modal representations
        if modal_emb is not None:
            z1 = self.projector(modal_emb)
            z2 = self.projector(all_item_emb)
            ssl_loss = self.info_nce_loss(z1, z2)
        else:
            ssl_loss = 0.0
        
        # L2 regularization
        reg_loss = self.reg_weight * (
            torch.norm(user_emb) +
            torch.norm(pos_emb) +
            torch.norm(neg_emb)
        )
        
        return bpr_loss + self.ssl_weight * ssl_loss + reg_loss

    def info_nce_loss(self, z1, z2):
        # Normalized features
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        logits = torch.matmul(z1, z2.T) / self.temperature
        labels = torch.arange(z1.shape[0], device=self.device)
        
        loss = F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
        return loss / 2

    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_emb, item_emb, _ = self.forward(user, training=False)
        scores = torch.matmul(user_emb, item_emb.transpose(0, 1))
        return scores

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        q = self.q_proj(x).reshape(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).reshape(batch_size, -1, self.dim)
        
        return self.out_proj(out)

class ModalGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, 2),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x1, x2):
        gate_input = torch.cat([x1, x2], dim=-1)
        gate_weights = self.gate(gate_input)
        return gate_weights[:, 0].unsqueeze(-1) * x1 + gate_weights[:, 1].unsqueeze(-1) * x2

class GraphConvolution(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, adj, features):
        return torch.sparse.mm(adj, features)