import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from torch.nn.init import xavier_normal_
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity

from common.abstract_recommender import GeneralRecommender
from common.loss import EmbLoss

class MMGAT(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MMGAT, self).__init__(config, dataset)

        # Basic setup
        self.n_nodes = self.n_users + self.n_items

        # Hyperparameters
        self.embedding_dim = 64
        self.feat_embed_dim = 64 
        self.n_heads = 2
        self.dropout = 0.1
        self.n_layers = 3
        self.alpha = 0.2
        self.reg_weight = 0.001
        self.mirror_coeff = 0.5
        
        # Load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.norm_adj = self.get_norm_adj_mat().to(self.device)

        # Embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        # Feature transformation layers
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)
            nn.init.xavier_normal_(self.image_trs.weight)
        
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)
            nn.init.xavier_normal_(self.text_trs.weight)

        # Attention layers
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(self.feat_embed_dim, self.n_heads, dropout=self.dropout)
            for _ in range(self.n_layers)
        ])

        # MLP layers
        self.predictor = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.embedding_dim, self.embedding_dim)
        )

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
        return torch.sparse.FloatTensor(i, data, torch.Size((self.n_nodes, self.n_nodes)))

    def laplacian_normalize(self, features):
        D = torch.sum(features, dim=1)
        D_inv_sqrt = torch.pow(D + 1e-7, -0.5)
        D_inv_sqrt = torch.diag_embed(D_inv_sqrt)
        norm_features = torch.matmul(torch.matmul(D_inv_sqrt, features), D_inv_sqrt)
        return norm_features

    def mirror_gradient_update(self, features, alpha=0.5):
        forward_features = features.detach().clone()
        backward_features = features.detach().clone()
        
        mirror_features = alpha * forward_features + (1 - alpha) * backward_features
        mirror_features.requires_grad_(True)
        
        return mirror_features

    def forward(self):
        # Basic embeddings
        h = self.item_id_embedding.weight
        ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
        all_embeddings = [ego_embeddings]

        # Process multimodal features
        if self.v_feat is not None:
            v_feat = self.image_trs(self.image_embedding.weight)
            v_feat = self.laplacian_normalize(v_feat)
            v_feat = self.mirror_gradient_update(v_feat)

        if self.t_feat is not None:
            t_feat = self.text_trs(self.text_embedding.weight)
            t_feat = self.laplacian_normalize(t_feat)
            t_feat = self.mirror_gradient_update(t_feat)

        # Multi-modal fusion
        if self.v_feat is not None and self.t_feat is not None:
            fused_features = self.mirror_coeff * v_feat + (1 - self.mirror_coeff) * t_feat
        elif self.v_feat is not None:
            fused_features = v_feat
        elif self.t_feat is not None:
            fused_features = t_feat
        else:
            fused_features = h

        # Apply attention mechanism
        for attention in self.attention_layers:
            fused_features = attention(fused_features, fused_features, fused_features)[0]
            all_embeddings.append(ego_embeddings + fused_features)

        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        return u_g_embeddings, i_g_embeddings + h

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        u_embeddings, i_embeddings = self.forward()
        
        u_embeddings = u_embeddings[users]
        pos_embeddings = i_embeddings[pos_items]
        neg_embeddings = i_embeddings[neg_items]

        # BPR Loss
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        
        mf_loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores)))

        # Regularization Loss
        reg_loss = self.reg_weight * (
            torch.norm(u_embeddings) +
            torch.norm(pos_embeddings) +
            torch.norm(neg_embeddings)
        )

        # Mirror Gradient Loss
        if self.v_feat is not None:
            v_embeddings = self.image_trs(self.image_embedding.weight[pos_items])
            mirror_v_loss = self.mirror_coeff * F.mse_loss(pos_embeddings, v_embeddings)
            reg_loss += mirror_v_loss

        if self.t_feat is not None:
            t_embeddings = self.text_trs(self.text_embedding.weight[pos_items])
            mirror_t_loss = self.mirror_coeff * F.mse_loss(pos_embeddings, t_embeddings)
            reg_loss += mirror_t_loss

        return mf_loss + reg_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        u_embeddings, i_embeddings = self.forward()
        u_embeddings = u_embeddings[user]
        scores = torch.matmul(u_embeddings, i_embeddings.transpose(0, 1))
        return scores