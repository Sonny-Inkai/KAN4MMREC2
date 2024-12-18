import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np
import copy

from common.abstract_recommender import GeneralRecommender

class MMGAT(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MMGAT, self).__init__(config, dataset)
        
        # Hyperparameters from config
        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.n_layers = config['n_layers']
        self.dropout = config['dropout']
        self.reg_weight = config['reg_weight']
        self.n_heads = 4
        self.temperature = 0.2
        self.n_nodes = self.n_users + self.n_items
        
        # Load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.norm_adj = self.get_norm_adj_mat().to(self.device)
        
        # Initialize embeddings with smaller variance
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        self.apply(self._init_weights)
        
        # Simplified modal encoders with layer normalization
        if self.v_feat is not None:
            self.v_feat = self.v_feat.to(self.device)
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_encoder = nn.Sequential(
                nn.Linear(self.v_feat.shape[1], self.feat_embed_dim),
                nn.LayerNorm(self.feat_embed_dim),
                nn.ReLU(),
                nn.Linear(self.feat_embed_dim, self.feat_embed_dim),
                nn.LayerNorm(self.feat_embed_dim)
            ).to(self.device)
            
        if self.t_feat is not None:
            self.t_feat = self.t_feat.to(self.device)
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_encoder = nn.Sequential(
                nn.Linear(self.t_feat.shape[1], self.feat_embed_dim),
                nn.LayerNorm(self.feat_embed_dim),
                nn.ReLU(),
                nn.Linear(self.feat_embed_dim, self.feat_embed_dim),
                nn.LayerNorm(self.feat_embed_dim)
            ).to(self.device)

        # Attention-based fusion
        self.modal_attention = nn.Sequential(
            nn.Linear(self.feat_embed_dim, self.feat_embed_dim),
            nn.Tanh(),
            nn.Linear(self.feat_embed_dim, 1, bias=False)
        )
        
        # Gradient reversal layer scale
        self.grl_lambda = 0.1
        
        self.to(self.device)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self):
        # Modal feature extraction and fusion
        modal_features = []
        if self.v_feat is not None:
            v_features = self.image_encoder(self.image_embedding.weight)
            modal_features.append(v_features)
            
        if self.t_feat is not None:
            t_features = self.text_encoder(self.text_embedding.weight)
            modal_features.append(t_features)
            
        # Attention-based modal fusion
        if len(modal_features) > 0:
            if len(modal_features) > 1:
                features_stack = torch.stack(modal_features, dim=1)
                attention_weights = F.softmax(self.modal_attention(features_stack).squeeze(-1), dim=1)
                fused_features = (features_stack * attention_weights.unsqueeze(-1)).sum(dim=1)
            else:
                fused_features = modal_features[0]
        else:
            fused_features = self.item_id_embedding.weight

        # Graph convolution with residual connections and layer normalization
        ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
        all_embeddings = [F.normalize(ego_embeddings, p=2, dim=-1)]
        
        for layer in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.norm_adj, ego_embeddings)
            ego_embeddings = F.normalize(ego_embeddings, p=2, dim=-1)
            ego_embeddings = F.dropout(ego_embeddings, self.dropout, training=self.training)
            all_embeddings.append(ego_embeddings)
            
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        
        # Feature enhancement with gradient reversal
        if self.training:
            i_g_embeddings = i_g_embeddings + self.grl_lambda * fused_features.detach()
        else:
            i_g_embeddings = i_g_embeddings + fused_features
            
        return u_g_embeddings, i_g_embeddings

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        u_embeddings, i_embeddings = self.forward()
        
        # InfoNCE loss with hard negative mining
        u_e = u_embeddings[users]
        pos_e = i_embeddings[pos_items]
        neg_e = i_embeddings[neg_items]
        
        # L2 normalization
        u_e = F.normalize(u_e, dim=-1)
        pos_e = F.normalize(pos_e, dim=-1)
        neg_e = F.normalize(neg_e, dim=-1)
        
        # Positive logits with temperature scaling
        pos_logits = torch.sum(u_e * pos_e, dim=-1) / self.temperature
        
        # Hard negative mining
        with torch.no_grad():
            neg_candidates = torch.matmul(u_e, neg_e.transpose(-2, -1))
            neg_candidates = F.softmax(neg_candidates / self.temperature, dim=-1)
        
        # Weighted negative logits
        neg_logits = torch.sum(neg_candidates * torch.matmul(u_e, neg_e.transpose(-2, -1)), dim=-1)
        
        loss = -torch.mean(pos_logits) + torch.mean(neg_logits)
        
        # Adaptive regularization
        reg_loss = self.reg_weight * (
            torch.mean(torch.norm(u_e, p=2, dim=-1)) +
            torch.mean(torch.norm(pos_e, p=2, dim=-1)) +
            torch.mean(torch.norm(neg_e, p=2, dim=-1))
        )
        
        # Curriculum learning weight
        cur_weight = min(float(self.global_step) / 1000.0, 1.0)
        
        return loss * cur_weight + reg_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        
        u_embeddings, i_embeddings = self.forward()
        
        # Normalize embeddings for cosine similarity
        u_embeddings = F.normalize(u_embeddings, dim=-1)
        i_embeddings = F.normalize(i_embeddings, dim=-1)
        
        scores = torch.matmul(u_embeddings[user], i_embeddings.transpose(0, 1))
        return scores

    def get_norm_adj_mat(self):
        """Get normalized adjacency matrix"""
        A = sp.dok_matrix((self.n_nodes, self.n_nodes), dtype=np.float32)
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
        indices = torch.LongTensor(np.array([L.row, L.col]))
        values = torch.FloatTensor(L.data)
        return torch.sparse_coo_tensor(indices, values, torch.Size((self.n_nodes, self.n_nodes)))

    def get_knn_adj_mat(self, mm_embeddings):
        """Get KNN adjacency matrix"""
        context_norm = F.normalize(mm_embeddings, p=2, dim=-1)
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        _, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
        adj_size = sim.size()
        del sim
        
        indices0 = torch.arange(knn_ind.shape[0], device=self.device)
        indices0 = torch.unsqueeze(indices0, 1)
        indices0 = indices0.expand(-1, self.knn_k)
        indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)
        
        return indices, self.compute_normalized_laplacian(indices, adj_size)

    def compute_normalized_laplacian(self, indices, adj_size):
        """Compute normalized Laplacian"""
        adj = torch.sparse_coo_tensor(indices, torch.ones_like(indices[0], device=self.device), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return torch.sparse_coo_tensor(indices, values, adj_size)