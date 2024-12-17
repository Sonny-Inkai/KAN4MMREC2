# coding: utf-8
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp

from common.abstract_recommender import GeneralRecommender
from common.loss import EmbLoss

class MMGAT(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MMGAT, self).__init__(config, dataset)
        
        # Basic parameters
        self.embedding_dim = 64
        self.feat_embed_dim = 64
        self.n_layers = 2
        self.knn_k = 10
        self.mm_image_weight = 0.5
        self.dropout = 0.1
        self.n_heads = 4
        self.alpha = 0.2  # LeakyReLU slope
        self.lambda_coeff = 0.5
        self.reg_weight = 1e-4
        self.temp = 0.2
        
        self.n_nodes = self.n_users + self.n_items
        self.build_item_graph = True
        self.mm_adj = None
        
        # Load interaction matrix and create normalized adjacency matrix
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.norm_adj = self.get_norm_adj_mat().to(self.device)
        
        # Initialize embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)
        
        # Flash attention layers
        self.flash_attn = nn.ModuleList([
            FlashMultiHeadAttention(self.embedding_dim, self.n_heads) 
            for _ in range(self.n_layers)
        ])
        
        # Feature transformations
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)
            nn.init.xavier_uniform_(self.image_trs.weight)
            
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)
            nn.init.xavier_uniform_(self.text_trs.weight)
            
        # Modal fusion
        self.modal_fusion = ModalFusionLayer(self.feat_embed_dim)
        
        # Initialize multimodal graph
        self._init_mm_graph()
        
        # Prediction layers
        self.predictor = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
            nn.LeakyReLU(self.alpha),
            nn.Dropout(self.dropout),
            nn.Linear(self.embedding_dim, self.embedding_dim)
        )
        
        self.reg_loss = EmbLoss()

    def _init_mm_graph(self):
        """Initialize multimodal graph with kNN structure"""
        if self.v_feat is not None:
            indices, image_adj = self.get_knn_adj_mat(self.image_embedding.weight.detach())
            self.mm_adj = image_adj
            
        if self.t_feat is not None:
            indices, text_adj = self.get_knn_adj_mat(self.text_embedding.weight.detach())
            self.mm_adj = text_adj
            
        if self.v_feat is not None and self.t_feat is not None:
            self.mm_adj = self.mm_image_weight * image_adj + (1.0 - self.mm_image_weight) * text_adj

    def get_norm_adj_mat(self):
        """Get normalized adjacency matrix"""
        A = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        for key, value in data_dict.items():
            A[key] = value
        
        # Symmetric normalization
        sumArr = (A > 0).sum(axis=1)
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        
        # Convert to tensor
        L = sp.coo_matrix(L)
        indices = torch.LongTensor([L.row, L.col])
        values = torch.FloatTensor(L.data)
        return torch.sparse.FloatTensor(indices, values, torch.Size([self.n_nodes, self.n_nodes]))

    def get_knn_adj_mat(self, embeddings):
        """Create kNN adjacency matrix from embeddings"""
        # Normalize embeddings
        norm_embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Compute similarity matrix
        sim = torch.mm(norm_embeddings, norm_embeddings.t())
        
        # Get top-k neighbors
        _, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
        
        # Create sparse adjacency matrix
        rows = torch.arange(knn_ind.size(0)).view(-1, 1).repeat(1, self.knn_k).flatten()
        adj_size = sim.size()
        
        # Compute normalized Laplacian
        indices = torch.stack([rows, knn_ind.flatten()])
        values = torch.ones_like(indices[0].float())
        adj = torch.sparse.FloatTensor(indices, values, adj_size)
        
        # Symmetric normalization
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        
        return indices, torch.sparse.FloatTensor(indices, values, adj_size)

    def forward(self):
        """Forward propagation"""
        # Get initial embeddings
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_id_embedding.weight
        
        # Process modal features
        modal_embedding = None
        if self.v_feat is not None:
            image_feats = self.image_trs(self.image_embedding.weight)
            modal_embedding = image_feats
            
        if self.t_feat is not None:
            text_feats = self.text_trs(self.text_embedding.weight)
            if modal_embedding is None:
                modal_embedding = text_feats
            else:
                modal_embedding = self.modal_fusion(modal_embedding, text_feats)
        
        # Combine embeddings
        all_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        embeddings_list = [all_embeddings]
        
        # Graph attention layers
        for i in range(self.n_layers):
            # Modal graph convolution for items
            if modal_embedding is not None:
                item_embeddings = item_embeddings + torch.sparse.mm(self.mm_adj, modal_embedding)
            
            # Update user-item graph
            all_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
            layer_embeddings = torch.sparse.mm(self.norm_adj, all_embeddings)
            
            # Flash attention
            attn_out = self.flash_attn[i](
                layer_embeddings.unsqueeze(0), 
                layer_embeddings.unsqueeze(0), 
                layer_embeddings.unsqueeze(0)
            ).squeeze(0)
            
            # Residual connection and layer norm
            layer_embeddings = F.layer_norm(
                layer_embeddings + attn_out, 
                [self.embedding_dim]
            )
            
            # Split users and items
            user_embeddings, item_embeddings = torch.split(
                layer_embeddings, 
                [self.n_users, self.n_items]
            )
            
            embeddings_list.append(layer_embeddings)
            
        # Aggregate embeddings from all layers
        final_embeddings = torch.stack(embeddings_list, dim=1)
        final_embeddings = torch.mean(final_embeddings, dim=1)
        
        user_embeddings, item_embeddings = torch.split(
            final_embeddings, 
            [self.n_users, self.n_items]
        )
        
        return user_embeddings, item_embeddings

    def calculate_loss(self, interaction):
        """Calculate model loss with mirror gradient"""
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        
        user_embeddings, item_embeddings = self.forward()
        
        # First forward pass
        user_all = user_embeddings[users]
        pos_all = item_embeddings[pos_items]
        neg_all = item_embeddings[neg_items]
        
        pos_scores = torch.sum(user_all * pos_all, dim=1)
        neg_scores = torch.sum(user_all * neg_all, dim=1)
        
        bpr_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        
        # Mirror gradient with momentum
        momentum = 0.9
        if hasattr(self, 'prev_grad'):
            grad_direction = torch.autograd.grad(bpr_loss, user_all, retain_graph=True)[0]
            mirror_direction = grad_direction + momentum * self.prev_grad
            self.prev_grad = grad_direction.detach()
        else:
            grad_direction = torch.autograd.grad(bpr_loss, user_all, retain_graph=True)[0]
            mirror_direction = grad_direction
            self.prev_grad = grad_direction.detach()
            
        # Second forward pass with mirrored gradients
        user_mirror = user_all - 0.1 * mirror_direction
        pos_scores_mirror = torch.sum(user_mirror * pos_all, dim=1)
        neg_scores_mirror = torch.sum(user_mirror * neg_all, dim=1)
        
        mirror_loss = -torch.mean(F.logsigmoid(pos_scores_mirror - neg_scores_mirror))
        
        # Contrastive loss
        modal_loss = 0.0
        if self.v_feat is not None:
            image_feats = self.image_trs(self.image_embedding.weight)
            pos_sim = torch.exp(torch.sum(user_all * image_feats[pos_items], dim=1) / self.temp)
            neg_sim = torch.exp(torch.sum(user_all * image_feats[neg_items], dim=1) / self.temp)
            modal_loss += -torch.log(pos_sim / (pos_sim + neg_sim)).mean()
            
        if self.t_feat is not None:
            text_feats = self.text_trs(self.text_embedding.weight)
            pos_sim = torch.exp(torch.sum(user_all * text_feats[pos_items], dim=1) / self.temp)
            neg_sim = torch.exp(torch.sum(user_all * text_feats[neg_items], dim=1) / self.temp)
            modal_loss += -torch.log(pos_sim / (pos_sim + neg_sim)).mean()
        
        # L2 regularization
        reg_loss = self.reg_loss(user_all, pos_all, neg_all)
        
        # Total loss with weighted components
        loss = bpr_loss + 0.1 * mirror_loss + 0.1 * modal_loss + self.reg_weight * reg_loss
        
        return loss

    def full_sort_predict(self, interaction):
        """Predict scores for evaluation"""
        user = interaction[0]
        
        user_embeddings, item_embeddings = self.forward()
        
        u_embeddings = user_embeddings[user]
        scores = torch.matmul(u_embeddings, item_embeddings.t())
        
        return scores

class FlashMultiHeadAttention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scaling = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
    def forward(self, q, k, v, mask=None):
        batch_size, seq_len, _ = q.shape
        
        # Project and reshape
        q = self.q_proj(q).view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(k).view(batch_size, seq_len, self.n_heads, self.head_dim)
        v = self.v_proj(v).view(batch_size, seq_len, self.n_heads, self.head_dim)
        
        # Transpose for attention
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Scaled dot product attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
            
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention
        output = torch.matmul(attn, v)
        
        # Reshape and project output
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        output = self.out_proj(output)
        
        return output

class ModalFusionLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x1, x2):
        """Modal fusion with attention mechanism"""
        # Concatenate features
        cat_features = torch.cat([x1, x2], dim=-1)
        
        # Calculate attention weights
        weights = self.attention(cat_features)
        
        # Weighted sum
        fused = weights * x1 + (1 - weights) * x2
        
        return fused.to(x1.device)

class InfoNCE(nn.Module):
    """InfoNCE contrastive loss"""
    def __init__(self, temp=0.2):
        super().__init__()
        self.temp = temp
        
    def forward(self, query, positive_key, negative_keys=None):
        positive_score = torch.sum(query * positive_key, dim=-1) / self.temp
        positive_score = torch.exp(positive_score)
        
        if negative_keys is not None:
            negative_score = torch.exp(torch.matmul(query, negative_keys.t()) / self.temp)
            denominator = negative_score.sum(dim=-1) + positive_score
        else:
            denominator = positive_score
            
        loss = -torch.log(positive_score / denominator)
        return loss.mean()

class GraphAttentionLayer(nn.Module):
    """Graph attention layer"""
    def __init__(self, in_features, out_features, dropout=0.1, alpha=0.2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data)
        
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data)
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        
    def forward(self, h, adj):
        Wh = torch.mm(h, self.W)
        e = self._prepare_attentional_mechanism_input(Wh)
        
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout)
        
        h_prime = torch.matmul(attention, Wh)
        return h_prime
    
    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        
        e = Wh1 + Wh2.transpose(0,1)
        return self.leakyrelu(e)

class LayerNorm(nn.Module):
    """Layer normalization module"""
    def __init__(self, dim, eps=1e-7):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))
        self.b = nn.Parameter(torch.zeros(dim))
        
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        
        x = (x - mean) / (std + self.eps)
        x = x * self.g + self.b
        
        return x