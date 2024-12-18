import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np

from common.abstract_recommender import GeneralRecommender

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, x):
        batch_size = x.size(0)
        q = self.query(x).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.head_dim)
        return self.proj(out)

class ModalFusionLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attention = MultiHeadAttention(dim, 8)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim)
        )
        
    def forward(self, x):
        x = self.norm1(x + self.attention(x))
        x = self.norm2(x + self.ffn(x))
        return x

class MMGAT(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MMGAT, self).__init__(config, dataset)
        
        self.embedding_dim = 64
        self.feat_embed_dim = 64
        self.n_layers = 3
        self.n_mm_layers = 2
        self.dropout = 0.5
        self.reg_weight = 1e-4
        self.knn_k = 20
        self.lambda_coeff = 0.5
        self.cl_weight = 0.1
        self.mm_image_weight = 0.5
        self.temperature = 0.2
        self.n_nodes = self.n_users + self.n_items
        
        # Load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.norm_adj = self.get_norm_adj_mat().to(self.device)
        
        # User and item embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        # Modal fusion layers
        self.modal_fusion = ModalFusionLayer(self.feat_embed_dim)
        
        # Multi-modal feature processors with enhanced architecture
        if self.v_feat is not None:
            self.v_feat = self.v_feat.to(self.device)
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_encoder = nn.Sequential(
                nn.Linear(self.v_feat.shape[1], self.feat_embed_dim),
                nn.LayerNorm(self.feat_embed_dim),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.feat_embed_dim, self.feat_embed_dim)
            ).to(self.device)
            
        if self.t_feat is not None:
            self.t_feat = self.t_feat.to(self.device)
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_encoder = nn.Sequential(
                nn.Linear(self.t_feat.shape[1], self.feat_embed_dim),
                nn.LayerNorm(self.feat_embed_dim),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.feat_embed_dim, self.feat_embed_dim)
            ).to(self.device)

        # Initialize multimodal adjacency matrix
        self.mm_adj = None
        if self.v_feat is not None:
            v_feat_map = self.image_encoder(self.image_embedding.weight)
            indices, image_adj = self.get_knn_adj_mat(v_feat_map)
            self.mm_adj = image_adj.to(self.device)
            
        if self.t_feat is not None:
            t_feat_map = self.text_encoder(self.text_embedding.weight)
            indices, text_adj = self.get_knn_adj_mat(t_feat_map)
            text_adj = text_adj.to(self.device)
            self.mm_adj = text_adj if self.mm_adj is None else self.mm_adj
            
        if self.v_feat is not None and self.t_feat is not None:
            self.mm_adj = self.mm_image_weight * image_adj + (1.0 - self.mm_image_weight) * text_adj

        # Advanced predictors for mirror gradient
        self.predictor = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
            nn.GELU(),
            nn.Linear(self.embedding_dim, self.embedding_dim)
        )
        
        # Ensure all components are on the same device
        self.to(self.device)

    def get_norm_adj_mat(self):
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
        return torch.sparse_coo_tensor(indices, values, torch.Size((self.n_nodes, self.n_nodes))).to(self.device)

    def get_knn_adj_mat(self, mm_embeddings):
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
        adj = torch.sparse_coo_tensor(indices, torch.ones_like(indices[0], device=self.device), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return torch.sparse_coo_tensor(indices, values, adj_size)

    def forward(self):
        # Enhanced feature fusion
        modal_features = []
        if self.v_feat is not None:
            v_features = self.image_encoder(self.image_embedding.weight)
            modal_features.append(v_features)
            
        if self.t_feat is not None:
            t_features = self.text_encoder(self.text_embedding.weight)
            modal_features.append(t_features)
            
        if len(modal_features) > 0:
            if len(modal_features) > 1:
                modal_features = torch.stack(modal_features, dim=1)
                fused_features = self.modal_fusion(modal_features).mean(dim=1)
            else:
                fused_features = modal_features[0]
        else:
            fused_features = self.item_id_embedding.weight

        # Graph convolution
        ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
        all_embeddings = [ego_embeddings]
        
        for _ in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.norm_adj, ego_embeddings)
            all_embeddings.append(ego_embeddings)
            
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        
        # Multimodal enhancement
        if self.mm_adj is not None:
            mm_embeddings = i_g_embeddings + fused_features
            for _ in range(self.n_mm_layers):
                mm_embeddings = torch.sparse.mm(self.mm_adj, mm_embeddings)
            i_g_embeddings = i_g_embeddings + mm_embeddings
            
        return u_g_embeddings, i_g_embeddings

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        u_embeddings, i_embeddings = self.forward()
        
        # Multi-view contrastive learning with momentum encoder
        with torch.no_grad():
            # Momentum update for target networks
            self.momentum = min(1 - 1/(self.n_layers + 1), 0.999)
            
            u_target = u_embeddings.detach()
            i_target = i_embeddings.detach()
            
            # Stronger stochastic augmentation
            u_target = F.dropout(u_target, self.dropout * 1.5)
            i_target = F.dropout(i_target, self.dropout * 1.5)
            
            # Feature augmentation for modalities
            if self.v_feat is not None:
                v_feat_target = self.image_encoder(self.image_embedding.weight).detach()
                v_feat_target = F.dropout(v_feat_target, self.dropout)
                
            if self.t_feat is not None:
                t_feat_target = self.text_encoder(self.text_embedding.weight).detach()
                t_feat_target = F.dropout(t_feat_target, self.dropout)

        # Online network predictions
        u_online = self.predictor(u_embeddings)
        i_online = self.predictor(i_embeddings)
        
        # Adaptive temperature scaling
        batch_size = len(users)
        temp = self.temperature * (1 + torch.exp(-torch.tensor(batch_size/500.0))).item()

        # Multi-view InfoNCE loss with hard negative mining
        u_e = u_online[users]
        pos_e = i_online[pos_items]
        neg_e = i_online[neg_items]
        
        # Distance-based negative sampling weights
        neg_weights = torch.exp(-torch.sum((u_e.unsqueeze(1) - neg_e) ** 2, dim=-1) / temp)
        neg_weights = F.softmax(neg_weights, dim=-1)
        
        # Compute scores with weighted negatives
        pos_scores = torch.sum(u_e * pos_e, dim=1) / temp
        neg_scores = torch.sum(u_e * neg_e * neg_weights.unsqueeze(-1), dim=1) / temp

        # Bidirectional cross-entropy loss
        pos_loss = -torch.mean(F.logsigmoid(pos_scores))
        neg_loss = -torch.mean(F.logsigmoid(-neg_scores))
        bpr_loss = pos_loss + neg_loss

        # Enhanced multi-modal contrastive learning
        u_target = u_target[users]
        i_target = i_target[pos_items]
        
        # L2 normalization for embeddings
        norm_u_e = F.normalize(u_e, dim=-1)
        norm_pos_e = F.normalize(pos_e, dim=-1)
        norm_u_target = F.normalize(u_target, dim=-1)
        norm_i_target = F.normalize(i_target, dim=-1)
        
        # Multi-modal alignment loss
        modal_loss = 0.0
        if self.v_feat is not None:
            v_feat_online = self.predictor(self.image_encoder(self.image_embedding.weight))
            v_feat_online = v_feat_online[pos_items]
            v_feat_target = v_feat_target[pos_items]
            modal_loss += -torch.mean(F.cosine_similarity(
                F.normalize(v_feat_online, dim=-1),
                F.normalize(v_feat_target, dim=-1).detach(),
                dim=-1
            ))
            
        if self.t_feat is not None:
            t_feat_online = self.predictor(self.text_encoder(self.text_embedding.weight))
            t_feat_online = t_feat_online[pos_items]
            t_feat_target = t_feat_target[pos_items]
            modal_loss += -torch.mean(F.cosine_similarity(
                F.normalize(t_feat_online, dim=-1),
                F.normalize(t_feat_target, dim=-1).detach(),
                dim=-1
            ))
        
        # Bidirectional contrastive loss with dynamic temperature
        loss_ui = -torch.mean(F.cosine_similarity(norm_u_e, norm_i_target.detach(), dim=-1)) / temp
        loss_iu = -torch.mean(F.cosine_similarity(norm_pos_e, norm_u_target.detach(), dim=-1)) / temp
        
        # Dynamic weight adjustment based on training progress
        cl_weight = self.cl_weight * (1 - torch.exp(-torch.tensor(batch_size/500.0))).item()
        cl_loss = (loss_ui + loss_iu + modal_loss) * cl_weight

        # Adaptive L2 regularization
        l2_loss = self.reg_weight * (
            torch.norm(norm_u_e) / (len(users) * self.embedding_dim) +
            torch.norm(norm_pos_e) / (len(pos_items) * self.embedding_dim) +
            torch.norm(F.normalize(neg_e, dim=-1)) / (len(neg_items) * self.embedding_dim)
        )

        # Gradient scaling for stable optimization
        total_loss = bpr_loss + cl_loss + l2_loss
        scale = torch.exp(-total_loss.detach()).clamp(min=0.5, max=2.0)
        
        return total_loss * scale

    def full_sort_predict(self, interaction):
        user = interaction[0]
        
        u_embeddings, i_embeddings = self.forward()
        u_embeddings = self.predictor(u_embeddings)
        i_embeddings = self.predictor(i_embeddings)
        
        scores = torch.matmul(u_embeddings[user], i_embeddings.transpose(0, 1))
        return scores