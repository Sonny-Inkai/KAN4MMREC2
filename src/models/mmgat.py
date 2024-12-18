import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np
import copy

from common.abstract_recommender import GeneralRecommender

class DynamicRoutingLayer(nn.Module):
    def __init__(self, dim, n_heads=4, n_iter=3):
        super().__init__()
        self.n_heads = n_heads
        self.n_iter = n_iter
        self.scale = dim ** -0.5
        self.routing_weights = nn.Parameter(torch.randn(n_heads, dim, dim))
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, x):
        # x shape: [N, num_modalities, D]
        N, M, D = x.shape
        
        # Initialize routing logits
        routing_logits = torch.zeros(N, self.n_heads, M, M, device=x.device)
        
        # Iterative dynamic routing
        for _ in range(self.n_iter):
            # Softmax over modality dimension
            routing_weights = F.softmax(routing_logits, dim=-1)  # [N, heads, M, M]
            
            # Transform features for each head
            transformed_features = []
            for h in range(self.n_heads):
                # Transform input: [N, M, D] -> [N, M, D]
                x_transformed = torch.matmul(x, self.routing_weights[h])
                # Compute attention scores
                scores = torch.matmul(x_transformed, x.transpose(-2, -1)) * self.scale
                routing_logits[:, h] = scores
            
        # Final routing weights
        final_weights = F.softmax(routing_logits.mean(dim=1), dim=-1)  # [N, M, M]
        
        # Apply routing weights to features
        out = torch.matmul(final_weights, x)  # [N, M, D]
        
        # Average over modalities
        out = out.mean(dim=1)  # [N, D]
        
        return self.proj(out)

class ModalityGatingUnit(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
        
    def forward(self, x1, x2):
        gate = self.gate(torch.cat([x1, x2], dim=-1))
        return gate * x1 + (1 - gate) * x2

class MMGAT(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MMGAT, self).__init__(config, dataset)
        
        # Core hyperparameters
        self.embedding_dim = 64
        self.feat_embed_dim = 64
        self.n_layers = 3
        self.n_heads = 4
        self.n_routing_iter = 3
        self.dropout = 0.5
        self.reg_weight = 1e-4
        self.knn_k = 20
        self.cl_weight = 0.1
        self.temperature = 0.2
        self.n_nodes = self.n_users + self.n_items
        
        # Load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.norm_adj = self.get_norm_adj_mat().to(self.device)
        
        # Core embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        # Advanced modal fusion components
        self.dynamic_routing = DynamicRoutingLayer(self.feat_embed_dim, self.n_heads, self.n_routing_iter)
        self.modality_gate = ModalityGatingUnit(self.feat_embed_dim)
        
        # Enhanced modal encoders
        if self.v_feat is not None:
            self.v_feat = self.v_feat.to(self.device)
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_encoder = nn.Sequential(
                nn.Linear(self.v_feat.shape[1], self.feat_embed_dim),
                nn.LayerNorm(self.feat_embed_dim),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.feat_embed_dim, self.feat_embed_dim),
                nn.LayerNorm(self.feat_embed_dim)
            ).to(self.device)
            
        if self.t_feat is not None:
            self.t_feat = self.t_feat.to(self.device)
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_encoder = nn.Sequential(
                nn.Linear(self.t_feat.shape[1], self.feat_embed_dim),
                nn.LayerNorm(self.feat_embed_dim),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.feat_embed_dim, self.feat_embed_dim),
                nn.LayerNorm(self.feat_embed_dim)
            ).to(self.device)

        # Momentum encoders for target networks
        self.momentum_image_encoder = copy.deepcopy(self.image_encoder) if self.v_feat is not None else None
        self.momentum_text_encoder = copy.deepcopy(self.text_encoder) if self.t_feat is not None else None

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
            
        # Advanced predictors with residual connections
        self.predictor = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim)
        )
        
        # Initialize EMA parameters
        self.ema = 0.999
        self.global_step = 0
        
        self.to(self.device)

    def forward(self):
        # Enhanced feature fusion with dynamic routing
        modal_features = []
        if self.v_feat is not None:
            v_features = self.image_encoder(self.image_embedding.weight)
            modal_features.append(v_features)
            
        if self.t_feat is not None:
            t_features = self.text_encoder(self.text_embedding.weight)
            modal_features.append(t_features)
            
        if len(modal_features) > 0:
            if len(modal_features) > 1:
                # Stack modalities: [N, num_modalities, D]
                modal_features = torch.stack(modal_features, dim=1)
                # Apply dynamic routing
                fused_features = self.dynamic_routing(modal_features)
                
                # Additional gating between modalities
                v_gate = self.modality_gate(modal_features[:, 0], modal_features[:, 1])
                t_gate = self.modality_gate(modal_features[:, 1], modal_features[:, 0])
                gated_features = v_gate * modal_features[:, 0] + t_gate * modal_features[:, 1]
                
                # Combine routed and gated features
                fused_features = fused_features + gated_features.mean(dim=0)
            else:
                fused_features = modal_features[0]
        else:
            fused_features = self.item_id_embedding.weight

        # Enhanced graph convolution with skip connections
        ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
        all_embeddings = [ego_embeddings]
        
        for _ in range(self.n_layers):
            # Graph convolution
            ego_embeddings = torch.sparse.mm(self.norm_adj, ego_embeddings)
            # Skip connection
            ego_embeddings = ego_embeddings + all_embeddings[-1]
            all_embeddings.append(ego_embeddings)
            
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        
        # Multimodal enhancement with residual connection
        if self.mm_adj is not None:
            mm_embeddings = i_g_embeddings + fused_features
            for _ in range(self.n_layers):
                mm_embeddings = torch.sparse.mm(self.mm_adj, mm_embeddings)
            i_g_embeddings = i_g_embeddings + mm_embeddings
            
        return u_g_embeddings, i_g_embeddings

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        u_embeddings, i_embeddings = self.forward()
        
        # Update momentum encoders
        self.global_step += 1
        momentum = min(1 - 1/(self.global_step + 1), self.ema)
        
        with torch.no_grad():
            # Momentum update for target networks
            if self.v_feat is not None:
                for param_q, param_k in zip(self.image_encoder.parameters(), 
                                          self.momentum_image_encoder.parameters()):
                    param_k.data = param_k.data * momentum + param_q.data * (1 - momentum)
                    
            if self.t_feat is not None:
                for param_q, param_k in zip(self.text_encoder.parameters(),
                                          self.momentum_text_encoder.parameters()):
                    param_k.data = param_k.data * momentum + param_q.data * (1 - momentum)
            
            # Target embeddings
            u_target = u_embeddings.clone()
            i_target = i_embeddings.clone()
            
            # Strong augmentation
            u_target = F.dropout(u_target, self.dropout * 1.5)
            i_target = F.dropout(i_target, self.dropout * 1.5)
            
        # Online predictions
        u_online = self.predictor(u_embeddings)
        i_online = self.predictor(i_embeddings)
        
        # Adaptive temperature
        batch_size = len(users)
        temp = self.temperature * (1 + torch.exp(-torch.tensor(batch_size/500.0))).item()

        # Multi-view contrastive learning
        u_e = u_online[users]
        pos_e = i_online[pos_items]
        neg_e = i_online[neg_items]
        
        # Hard negative mining with similarity-based weights
        neg_weights = torch.exp(-torch.sum((u_e.unsqueeze(1) - neg_e) ** 2, dim=-1) / temp)
        neg_weights = F.softmax(neg_weights, dim=-1)
        
        # InfoNCE loss with weighted negatives
        pos_scores = torch.sum(u_e * pos_e, dim=1) / temp
        neg_scores = torch.sum(u_e * neg_e * neg_weights.unsqueeze(-1), dim=1) / temp

        # Bidirectional cross-entropy
        pos_loss = -torch.mean(F.logsigmoid(pos_scores))
        neg_loss = -torch.mean(F.logsigmoid(-neg_scores))
        bpr_loss = pos_loss + neg_loss

        # Enhanced contrastive learning
        u_target = u_target[users]
        i_target = i_target[pos_items]
        
        norm_u_e = F.normalize(u_e, dim=-1)
        norm_pos_e = F.normalize(pos_e, dim=-1)
        norm_u_target = F.normalize(u_target, dim=-1)
        norm_i_target = F.normalize(i_target, dim=-1)
        
        # Multi-modal alignment with momentum features
        modal_loss = 0.0
        if self.v_feat is not None:
            with torch.no_grad():
                v_feat_target = self.momentum_image_encoder(self.image_embedding.weight)
                v_feat_target = v_feat_target[pos_items]
                
            v_feat_online = self.predictor(self.image_encoder(self.image_embedding.weight))
            v_feat_online = v_feat_online[pos_items]
            
            modal_loss += -torch.mean(F.cosine_similarity(
                F.normalize(v_feat_online, dim=-1),
                F.normalize(v_feat_target, dim=-1),
                dim=-1
            ))
            
        if self.t_feat is not None:
            with torch.no_grad():
                t_feat_target = self.momentum_text_encoder(self.text_embedding.weight)
                t_feat_target = t_feat_target[pos_items]
                
            t_feat_online = self.predictor(self.text_encoder(self.text_embedding.weight))
            t_feat_online = t_feat_online[pos_items]
            
            modal_loss += -torch.mean(F.cosine_similarity(
                F.normalize(t_feat_online, dim=-1),
                F.normalize(t_feat_target, dim=-1),
                dim=-1
            ))
        
        # Bidirectional contrastive loss
        loss_ui = -torch.mean(F.cosine_similarity(norm_u_e, norm_i_target, dim=-1)) / temp
        loss_iu = -torch.mean(F.cosine_similarity(norm_pos_e, norm_u_target, dim=-1)) / temp
        
        # Dynamic weight adjustment
        cl_weight = self.cl_weight * (1 - torch.exp(-torch.tensor(batch_size/500.0))).item()
        cl_loss = (loss_ui + loss_iu + modal_loss) * cl_weight

        # Adaptive L2 regularization
        l2_loss = self.reg_weight * (
            torch.norm(norm_u_e) / (len(users) * self.embedding_dim) +
            torch.norm(norm_pos_e) / (len(pos_items) * self.embedding_dim) +
            torch.norm(F.normalize(neg_e, dim=-1)) / (len(neg_items) * self.embedding_dim)
        )

        # Gradient scaling
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