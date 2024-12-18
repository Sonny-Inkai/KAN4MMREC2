import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from common.abstract_recommender import GeneralRecommender
import scipy.sparse as sp
import numpy as np

class MMGAT(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MMGAT, self).__init__(config, dataset)
        
        # Basic parameters
        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.n_layers = config['n_layers']
        self.num_heads = config['num_heads']
        self.dropout = config['dropout']
        self.reg_weight = config['reg_weight']
        self.temperature = config['temperature']
        self.n_ui_layers = config['n_ui_layers']
        self.knn_k = config['knn_k']
        self.lambda_coeff = config['lambda_coeff']
        
        # Number of nodes
        self.n_nodes = self.n_users + self.n_items
        
        # Load interaction matrix and create normalized adjacency matrix
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.norm_adj = self.get_norm_adj_mat().to(self.device)
        
        # Get edge information
        self.edge_indices, self.edge_values = self.get_edge_info()
        self.edge_indices = self.edge_indices.to(self.device)
        self.edge_values = self.edge_values.to(self.device)
        
        # Initialize embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        # Modal-specific components
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)
            self.image_attention = MultiHeadAttention(self.feat_embed_dim, self.num_heads, self.dropout)
        
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)
            self.text_attention = MultiHeadAttention(self.feat_embed_dim, self.num_heads, self.dropout)

        # Initialize modal adjacency matrix
        self.mm_adj = None
        self.initialize_mm_adj()

    def get_norm_adj_mat(self):
        """Create normalized adjacency matrix with proper handling of edge cases."""
        A = sp.dok_matrix((self.n_nodes, self.n_nodes), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        
        # Create dictionary of interactions
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        A._update(data_dict)

        # Add self-loops and normalize
        A = A.tocoo().tocsr()
        A = A + sp.eye(self.n_nodes)
        degree = np.array(A.sum(axis=1)).squeeze()
        degree = np.maximum(degree, np.finfo(float).eps)  # Avoid division by zero
        d_inv_sqrt = np.power(degree, -0.5)
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        norm_adj = d_mat_inv_sqrt.dot(A).dot(d_mat_inv_sqrt)
        
        # Convert to torch sparse tensor
        norm_adj = norm_adj.tocoo()
        indices = np.vstack((norm_adj.row, norm_adj.col))
        indices = torch.from_numpy(indices).long()
        values = torch.from_numpy(norm_adj.data).float()
        shape = torch.Size(norm_adj.shape)
        
        return torch.sparse_coo_tensor(indices, values, shape)

    def get_edge_info(self):
        """Get edge information with proper tensor types."""
        rows = torch.from_numpy(self.interaction_matrix.row).long()
        cols = torch.from_numpy(self.interaction_matrix.col).long()
        edges = torch.stack([rows, cols])
        values = self._normalize_adj_values(edges)
        return edges, values

    def _normalize_adj_values(self, edges):
        """Normalize adjacency values with proper handling of edge cases."""
        values = torch.ones(edges.size(1), dtype=torch.float32)
        adj = torch.sparse_coo_tensor(
            edges, 
            values,
            size=(self.n_users, self.n_items)
        )
        
        # Add epsilon for numerical stability
        epsilon = 1e-7
        
        # Calculate degrees with safe handling of zero values
        row_sum = epsilon + torch.sparse.sum(adj, dim=-1).to_dense()
        col_sum = epsilon + torch.sparse.sum(adj.t(), dim=-1).to_dense()
        
        # Calculate normalized values
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        c_inv_sqrt = torch.pow(col_sum, -0.5)
        
        norm_values = r_inv_sqrt[edges[0]] * c_inv_sqrt[edges[1]]
        return norm_values

    def initialize_mm_adj(self):
        if self.v_feat is not None:
            image_feats = self.image_trs(self.image_embedding.weight)
            image_adj = self.build_knn_graph(image_feats)
            self.mm_adj = image_adj
            
        if self.t_feat is not None:
            text_feats = self.text_trs(self.text_embedding.weight)
            text_adj = self.build_knn_graph(text_feats)
            if self.mm_adj is None:
                self.mm_adj = text_adj
            else:
                self.mm_adj = 0.5 * (self.mm_adj + text_adj)

    def build_knn_graph(self, features):
        sim = torch.mm(features, features.t())
        values, indices = sim.topk(k=self.knn_k, dim=1)
        
        row_idx = torch.arange(features.size(0)).view(-1, 1).repeat(1, self.knn_k)
        adj = torch.zeros_like(sim)
        adj[row_idx.flatten(), indices.flatten()] = values.flatten()
        
        # Symmetric normalization
        deg = torch.sum(adj, dim=1)
        deg_inv_sqrt = torch.pow(deg + 1e-7, -0.5)
        norm_adj = deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(0)
        
        return norm_adj

    def forward(self, adj):
        # Process multimodal features
        modal_embeddings = []
        
        if self.v_feat is not None:
            image_feats = self.image_trs(self.image_embedding.weight)
            image_attn = self.image_attention(image_feats, image_feats, image_feats)
            modal_embeddings.append(image_attn)
            
        if self.t_feat is not None:
            text_feats = self.text_trs(self.text_embedding.weight)
            text_attn = self.text_attention(text_feats, text_feats, text_feats)
            modal_embeddings.append(text_attn)
        
        # Cross-modal fusion
        if len(modal_embeddings) > 1:
            item_enhanced = self.modal_fusion(modal_embeddings[0], modal_embeddings[1])
        else:
            item_enhanced = modal_embeddings[0]
            
        # Graph convolution on item features
        h = item_enhanced
        for _ in range(self.n_layers):
            h = torch.mm(self.mm_adj.to(self.device), h)
            
        # User-item graph convolution
        ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
        all_embeddings = [ego_embeddings]
        
        for _ in range(self.n_ui_layers):
            ego_embeddings = torch.sparse.mm(adj, ego_embeddings)
            all_embeddings.append(ego_embeddings)
            
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        i_g_embeddings = i_g_embeddings + F.normalize(h, p=2, dim=1)
        
        return u_g_embeddings, i_g_embeddings

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        # Mirror gradient logic
        ua_embeddings, ia_embeddings = self.forward(self.norm_adj)
        
        u_embeddings = ua_embeddings[users]
        pos_embeddings = ia_embeddings[pos_items]
        neg_embeddings = ia_embeddings[neg_items]
        
        # InfoNCE loss with temperature scaling
        pos_scores = torch.sum(u_embeddings * pos_embeddings, dim=1)
        neg_scores = torch.sum(u_embeddings * neg_embeddings, dim=1)
        
        pos_scores = torch.exp(pos_scores / self.temperature)
        neg_scores = torch.exp(neg_scores / self.temperature)
        
        nce_loss = -torch.log(pos_scores / (pos_scores + neg_scores))
        nce_loss = torch.mean(nce_loss)

        # Modal alignment loss
        modal_loss = 0.0
        if self.v_feat is not None and self.t_feat is not None:
            image_feats = self.image_trs(self.image_embedding.weight[pos_items])
            text_feats = self.text_trs(self.text_embedding.weight[pos_items])
            modal_loss = 1 - F.cosine_similarity(image_feats, text_feats, dim=1).mean()

        # L2 regularization
        l2_loss = self.reg_weight * (
            torch.norm(u_embeddings) +
            torch.norm(pos_embeddings) +
            torch.norm(neg_embeddings)
        )

        return nce_loss + 0.1 * modal_loss + l2_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        
        restore_user_e, restore_item_e = self.forward(self.norm_adj)
        u_embeddings = restore_user_e[user]
        
        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
        return scores

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product(self, Q, K, V):
        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores = scores / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        output = torch.matmul(scores, V)
        return output
        
    def forward(self, Q, K, V):
        batch_size = Q.size(0)
        
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        output = self.scaled_dot_product(Q, K, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        return self.W_o(output)

class CrossModalFusion(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()
        )
        
    def forward(self, x1, x2):
        gate = self.gate(torch.cat([x1, x2], dim=-1))
        return gate * x1 + (1 - gate) * x2