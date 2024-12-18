import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np
import copy
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity

from common.abstract_recommender import GeneralRecommender
from common.loss import EmbLoss


class HierarchicalAttention(nn.Module):
    def __init__(self, dim, n_heads=8):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
        
    def forward(self, x, mask=None):
        B, N, D = x.shape
        
        # Multi-head attention
        q = self.query(x).reshape(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).reshape(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).reshape(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = F.softmax(attn, dim=-1)
        
        x = torch.matmul(attn, v).transpose(1, 2).reshape(B, N, D)
        x = self.proj(x)
        
        # Gating mechanism
        gate = self.gate(torch.cat([x, v.transpose(1, 2).reshape(B, N, D)], dim=-1))
        return x * gate

class ModalityFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attention = HierarchicalAttention(dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(4 * dim, dim)
        )
        
    def forward(self, x):
        x = self.norm1(x + self.attention(x))
        x = self.norm2(x + self.mlp(x))
        return x

class MMGAT(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MMGAT, self).__init__(config, dataset)
        
        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.n_layers = config['n_layers']
        self.dropout = config['dropout']
        self.reg_weight = config['reg_weight']
        self.temperature = 0.07
        self.n_nodes = self.n_users + self.n_items
        
        # Load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.norm_adj = self.get_norm_adj_mat().to(self.device)
        
        # Core embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        self.apply(self._init_weights)
        
        # Enhanced modal encoders with hierarchical structure
        if self.v_feat is not None:
            self.v_feat = self.v_feat.to(self.device)
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_encoder = nn.Sequential(
                nn.Linear(self.v_feat.shape[1], self.feat_embed_dim),
                nn.LayerNorm(self.feat_embed_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(self.feat_embed_dim, self.feat_embed_dim)
            ).to(self.device)
            
        if self.t_feat is not None:
            self.t_feat = self.t_feat.to(self.device)
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_encoder = nn.Sequential(
                nn.Linear(self.t_feat.shape[1], self.feat_embed_dim),
                nn.LayerNorm(self.feat_embed_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(self.feat_embed_dim, self.feat_embed_dim)
            ).to(self.device)

        # Advanced fusion module
        self.modality_fusion = ModalityFusion(self.feat_embed_dim)
        
        # Momentum encoders for contrastive learning
        self.momentum_encoders = {
            'image': copy.deepcopy(self.image_encoder) if self.v_feat is not None else None,
            'text': copy.deepcopy(self.text_encoder) if self.t_feat is not None else None
        }
        
        # Projection heads
        self.projector = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
            nn.GELU(),
            nn.Linear(self.embedding_dim, self.embedding_dim)
        )
        
        self.predictor = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
            nn.GELU(),
            nn.Linear(self.embedding_dim, self.embedding_dim)
        )
        
        # Initialize momentum parameters
        self.momentum = 0.999
        self.global_step = 0
        
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

    def _update_momentum_encoder(self):
        """Update momentum encoders using exponential moving average"""
        for modal in ['image', 'text']:
            if self.momentum_encoders[modal] is not None:
                for param_q, param_k in zip(
                    getattr(self, f'{modal}_encoder').parameters(),
                    self.momentum_encoders[modal].parameters()
                ):
                    param_k.data = param_k.data * self.momentum + param_q.data * (1 - self.momentum)

    def forward(self):
        # Multi-modal feature extraction
        modal_features = []
        if self.v_feat is not None:
            v_features = self.image_encoder(self.image_embedding.weight)
            modal_features.append(v_features)
            
        if self.t_feat is not None:
            t_features = self.text_encoder(self.text_embedding.weight)
            modal_features.append(t_features)
            
        # Hierarchical fusion
        if len(modal_features) > 0:
            if len(modal_features) > 1:
                features_stack = torch.stack(modal_features, dim=1)
                fused_features = self.modality_fusion(features_stack)
                fused_features = fused_features.mean(dim=1)
            else:
                fused_features = modal_features[0]
        else:
            fused_features = self.item_id_embedding.weight

        # Enhanced graph convolution
        ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
        all_embeddings = [F.normalize(ego_embeddings, p=2, dim=-1)]
        
        for layer in range(self.n_layers):
            # Graph convolution with attention
            ego_embeddings = torch.sparse.mm(self.norm_adj, ego_embeddings)
            ego_embeddings = F.normalize(ego_embeddings + all_embeddings[-1], p=2, dim=-1)
            ego_embeddings = F.dropout(ego_embeddings, self.dropout, training=self.training)
            all_embeddings.append(ego_embeddings)
            
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        
        # Feature enhancement
        i_g_embeddings = i_g_embeddings + fused_features
            
        return u_g_embeddings, i_g_embeddings

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        # Update momentum encoders
        self._update_momentum_encoder()
        
        u_embeddings, i_embeddings = self.forward()
        
        # Online and target networks
        online_u = self.predictor(self.projector(u_embeddings))
        online_i = self.predictor(self.projector(i_embeddings))
        
        with torch.no_grad():
            target_u = self.projector(u_embeddings)
            target_i = self.projector(i_embeddings)
        
        # Extract embeddings for loss calculation
        u_e = online_u[users]
        pos_e = online_i[pos_items]
        neg_e = online_i[neg_items]
        
        # Target embeddings
        u_target = target_u[users]
        i_target = target_i[pos_items]
        
        # InfoNCE loss with hard negative mining
        pos_scores = torch.sum(u_e * pos_e, dim=-1) / self.temperature
        neg_scores = torch.matmul(u_e, neg_e.t()) / self.temperature
        
        # Hard negative mining
        with torch.no_grad():
            neg_weights = F.softmax(neg_scores, dim=-1)
        
        # Weighted InfoNCE loss
        pos_loss = -torch.mean(F.logsigmoid(pos_scores))
        neg_loss = torch.mean(torch.sum(neg_weights * F.logsigmoid(neg_scores), dim=-1))
        
        # Contrastive loss
        u_loss = -torch.mean(F.cosine_similarity(u_e, i_target.detach(), dim=-1))
        i_loss = -torch.mean(F.cosine_similarity(pos_e, u_target.detach(), dim=-1))
        
        # Multi-modal contrastive loss
        modal_loss = 0
        if self.v_feat is not None and self.t_feat is not None:
            v_feat = self.image_encoder(self.image_embedding.weight)[pos_items]
            t_feat = self.text_encoder(self.text_embedding.weight)[pos_items]
            modal_loss = -torch.mean(F.cosine_similarity(v_feat, t_feat, dim=-1))
        
        # Total loss with curriculum learning
        self.global_step += 1
        cur_weight = min(float(self.global_step) / 1000.0, 1.0)
        
        loss = (pos_loss + neg_loss + u_loss + i_loss + modal_loss) * cur_weight
        
        # L2 regularization
        reg_loss = self.reg_weight * (
            torch.norm(u_e) / len(users) +
            torch.norm(pos_e) / len(pos_items) +
            torch.norm(neg_e) / len(neg_items)
        )
        
        return loss + reg_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        
        u_embeddings, i_embeddings = self.forward()
        
        # Project embeddings for prediction
        u_embeddings = self.predictor(self.projector(u_embeddings))
        i_embeddings = self.predictor(self.projector(i_embeddings))
        
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