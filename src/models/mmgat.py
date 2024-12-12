import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity
import numpy as np
import scipy.sparse as sp
from common.abstract_recommender import GeneralRecommender

class MMGAT(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MMGAT, self).__init__()
        self.device = config['device']
        
        # Core parameters
        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.n_layers = config['n_layers']
        self.dropout = config['dropout']
        self.reg_weight = config['reg_weight']
        self.knn_k = config['knn_k']
        
        # Initialize embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        # Multi-modal feature processing
        if dataset.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(dataset.v_feat, freeze=False)
            self.image_transform = nn.Sequential(
                nn.Linear(dataset.v_feat.shape[1], self.feat_embed_dim),
                nn.LayerNorm(self.feat_embed_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout)
            )
            
        if dataset.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(dataset.t_feat, freeze=False)
            self.text_transform = nn.Sequential(
                nn.Linear(dataset.t_feat.shape[1], self.feat_embed_dim),
                nn.LayerNorm(self.feat_embed_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout)
            )

        # Cross-modal attention
        self.modal_attention = MultiHeadAttention(self.feat_embed_dim, 4, self.dropout)
        
        # Graph neural network layers
        self.gnn_layers = nn.ModuleList([
            GraphAttentionLayer(self.embedding_dim, self.embedding_dim, dropout=self.dropout)
            for _ in range(self.n_layers)
        ])

        # Adaptive fusion mechanism
        self.fusion_gate = nn.Sequential(
            nn.Linear(self.embedding_dim * 3, 3),
            nn.Softmax(dim=-1)
        )

        # Initialize adjacency matrix
        self.norm_adj = self.get_norm_adj_mat(dataset).to(self.device)

    def get_norm_adj_mat(self, dataset):
        # Build normalized adjacency matrix
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items))
        inter_mat = dataset.inter_matrix(form='coo').astype(np.float32)
        adj_mat = self._convert_sp_mat_to_sp_tensor(self._normalize_adj(adj_mat + sp.eye(adj_mat.shape[0])))
        return adj_mat

    def _normalize_adj(self, adj):
        rowsum = np.array(adj.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        return d_mat_inv.dot(adj).dot(d_mat_inv)

    def forward(self, user_indices, item_indices=None):
        # Get initial embeddings
        user_emb = self.user_embedding(user_indices)
        
        # Process multimodal features
        if hasattr(self, 'image_embedding'):
            img_feats = self.image_transform(self.image_embedding.weight)
        if hasattr(self, 'text_embedding'):
            txt_feats = self.text_transform(self.text_embedding.weight)

        # Cross-modal attention
        if hasattr(self, 'image_embedding') and hasattr(self, 'text_embedding'):
            modal_enhanced = self.modal_attention(img_feats, txt_feats)
        else:
            modal_enhanced = img_feats if hasattr(self, 'image_embedding') else txt_feats

        # Graph neural network propagation
        ego_embeddings = torch.cat([user_emb, modal_enhanced], dim=0)
        all_embeddings = [ego_embeddings]
        
        for gnn in self.gnn_layers:
            ego_embeddings = gnn(ego_embeddings, self.norm_adj)
            all_embeddings.append(ego_embeddings)
        
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)

        # Adaptive fusion
        if item_indices is not None:
            item_emb = i_g_embeddings[item_indices]
            modal_emb = modal_enhanced[item_indices]
            
            fusion_weights = self.fusion_gate(torch.cat([user_emb, item_emb, modal_emb], dim=-1))
            final_user_emb = fusion_weights[:, 0:1] * user_emb
            final_item_emb = fusion_weights[:, 1:2] * item_emb + fusion_weights[:, 2:3] * modal_emb
            
            return final_user_emb, final_item_emb
        
        return u_g_embeddings, i_g_embeddings

    def calculate_loss(self, users, pos_items, neg_items):
        user_emb, all_item_emb = self.forward(users)
        pos_emb = all_item_emb[pos_items]
        neg_emb = all_item_emb[neg_items]

        # BPR loss
        pos_scores = torch.sum(user_emb * pos_emb, dim=1)
        neg_scores = torch.sum(user_emb * neg_emb, dim=1)
        bpr_loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores)))

        # Contrastive loss for multimodal features
        modal_loss = 0
        if hasattr(self, 'image_embedding'):
            img_contrast = self.compute_contrastive_loss(
                self.image_transform(self.image_embedding.weight[pos_items]),
                self.image_transform(self.image_embedding.weight[neg_items])
            )
            modal_loss += img_contrast

        if hasattr(self, 'text_embedding'):
            txt_contrast = self.compute_contrastive_loss(
                self.text_transform(self.text_embedding.weight[pos_items]),
                self.text_transform(self.text_embedding.weight[neg_items])
            )
            modal_loss += txt_contrast

        # L2 regularization
        reg_loss = self.reg_weight * (
            torch.norm(user_emb) +
            torch.norm(pos_emb) +
            torch.norm(neg_emb)
        )

        return bpr_loss + 0.1 * modal_loss + reg_loss

    def compute_contrastive_loss(self, pos_feats, neg_feats, temperature=0.07):
        pos_sim = torch.exp(cosine_similarity(pos_feats, neg_feats) / temperature)
        neg_sim = torch.exp(torch.mm(pos_feats, neg_feats.t()) / temperature)
        return -torch.log(pos_sim / neg_sim.sum(dim=1))

    def predict(self, user_indices):
        user_emb, item_emb = self.forward(user_indices)
        return torch.matmul(user_emb, item_emb.t())

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.1):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Linear(2 * out_features, 1, bias=False)
        
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, x, adj):
        Wh = self.W(x)
        e = self._prepare_attentional_mechanism_input(Wh)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout)
        h_prime = torch.matmul(attention, Wh)
        
        return F.elu(h_prime)

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a.weight.T[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a.weight.T[self.out_features:, :])
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value=None):
        if value is None:
            value = key
            
        batch_size = query.size(0)
        
        # Linear transformations
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)
        
        # Split into heads
        Q = Q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, V)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # Final linear transformation
        output = self.W_o(context)
        
        return output