import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np

from common.abstract_recommender import GeneralRecommender
from common.loss import EmbLoss

class FlashAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1, scale_factor=0.125):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = (self.head_dim ** -0.5) * scale_factor
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
        # Add learnable temperature parameter
        self.temp = nn.Parameter(torch.ones(1, num_heads, 1, 1))
        
    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention with learnable temperature
        attn = (q @ k.transpose(-2, -1)) * self.scale * self.temp
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
            
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class ModalFusionBlock(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = FlashAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class AdaptiveModalFusion(nn.Module):
    def __init__(self, dim, num_modalities=2):
        super().__init__()
        self.modal_weights = nn.Parameter(torch.ones(num_modalities) / num_modalities)
        self.modal_norm = nn.LayerNorm(dim)
        self.fusion_layer = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )
        
    def forward(self, modal_features):
        # Normalize weights
        weights = F.softmax(self.modal_weights, dim=0)
        
        # Weighted sum of modalities
        fused = torch.zeros_like(modal_features[0])
        for i, features in enumerate(modal_features):
            fused += weights[i] * features
            
        # Apply normalization and fusion
        fused = self.modal_norm(fused)
        fused = self.fusion_layer(fused)
        return fused

class MMGAT(GeneralRecommender):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        
        # Initialize dimensions and parameters
        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.n_layers = config['n_layers']
        self.dropout = config['dropout']
        self.reg_weight = config['reg_weight']
        self.n_heads = config['n_heads']
        self.temperature = config['temperature']
        
        self.n_nodes = self.n_users + self.n_items
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.norm_adj = self.get_norm_adj_mat().to(self.device)
        
        # Enhanced embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        self.apply(self._init_weights)
        
        # Modal encoders with residual connections
        if self.v_feat is not None:
            self.v_feat = self.v_feat.to(self.device)
            self.image_encoder = nn.Sequential(
                nn.Linear(self.v_feat.shape[1], self.feat_embed_dim),
                nn.LayerNorm(self.feat_embed_dim),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.feat_embed_dim, self.feat_embed_dim)
            )
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            
        if self.t_feat is not None:
            self.t_feat = self.t_feat.to(self.device)
            self.text_encoder = nn.Sequential(
                nn.Linear(self.t_feat.shape[1], self.feat_embed_dim),
                nn.LayerNorm(self.feat_embed_dim),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.feat_embed_dim, self.feat_embed_dim)
            )
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
        
        # Enhanced modal fusion with Flash Attention
        self.modal_fusion_blocks = nn.ModuleList([
            ModalFusionBlock(self.feat_embed_dim, self.n_heads)
            for _ in range(2)
        ])
        
        # Add adaptive modal fusion
        self.adaptive_fusion = AdaptiveModalFusion(self.feat_embed_dim)
        
        # Graph attention layers
        self.gat_layers = nn.ModuleList([
            FlashAttention(self.embedding_dim, self.n_heads, self.dropout)
            for _ in range(self.n_layers)
        ])
        
        self.to(self.device)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.xavier_uniform_(m.weight)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    
    def forward(self):
        # Modal feature extraction and fusion
        modal_features = []
        if self.v_feat is not None:
            img_feat = self.image_encoder(self.image_embedding.weight)
            modal_features.append(img_feat)
            
        if self.t_feat is not None:
            txt_feat = self.text_encoder(self.text_embedding.weight)
            modal_features.append(txt_feat)
        
        # Enhanced modal fusion
        if len(modal_features) > 0:
            if len(modal_features) > 1:
                # Apply adaptive fusion
                fused_features = self.adaptive_fusion(modal_features)
                
                # Apply modal fusion blocks
                modal_stack = torch.stack(modal_features, dim=1)
                for fusion_layer in self.modal_fusion_blocks:
                    modal_stack = fusion_layer(modal_stack)
                fused_features = fused_features + torch.mean(modal_stack, dim=1)
            else:
                fused_features = modal_features[0]
        else:
            fused_features = None
        
        # Graph attention with residual connections
        x = torch.cat([self.user_embedding.weight, self.item_embedding.weight])
        all_embs = [x]
        
        for i in range(self.n_layers):
            # Graph attention
            x_res = x
            x = torch.sparse.mm(self.norm_adj, x)
            x = self.gat_layers[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + x_res  # Residual connection
            all_embs.append(x)
        
        x = torch.mean(torch.stack(all_embs, dim=1), dim=1)
        user_emb, item_emb = torch.split(x, [self.n_users, self.n_items])
        
        # Combine with modal features
        if fused_features is not None:
            item_emb = item_emb + fused_features
        
        return user_emb, item_emb
    
    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        
        user_emb, item_emb = self.forward()
        
        u_emb = user_emb[users]
        pos_emb = item_emb[pos_items]
        neg_emb = item_emb[neg_items]
        
        # InfoNCE loss with temperature scaling
        pos_scores = torch.sum(u_emb * pos_emb, dim=1) / self.temperature
        neg_scores = torch.sum(u_emb * neg_emb, dim=1) / self.temperature
        
        loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        
        # Modal alignment loss with improved temperature scaling
        if hasattr(self, 'image_embedding') and hasattr(self, 'text_embedding'):
            img_feat = self.image_encoder(self.image_embedding.weight)
            txt_feat = self.text_encoder(self.text_embedding.weight)
            
            img_feat = F.normalize(img_feat, dim=-1)
            txt_feat = F.normalize(txt_feat, dim=-1)
            
            modal_sim = torch.matmul(img_feat, txt_feat.t()) / self.temperature
            modal_loss = -torch.mean(torch.diag(F.log_softmax(modal_sim, dim=1)))
            loss = loss + 0.1 * modal_loss
        
        # L2 regularization with improved weighting
        reg_loss = self.reg_weight * (
            torch.norm(u_emb) / len(users) +
            torch.norm(pos_emb) / len(pos_items) +
            torch.norm(neg_emb) / len(neg_items)
        )
        
        return loss + reg_loss
    
    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_emb, item_emb = self.forward()
        
        # Get embeddings for the batch of users
        u_embeddings = user_emb[user]
        
        # Calculate scores for all items
        scores = torch.matmul(u_embeddings, item_emb.transpose(0, 1))
        
        return scores

    def get_norm_adj_mat(self):
        A = sp.dok_matrix((self.n_nodes, self.n_nodes), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        for key, value in data_dict.items():
            A[key] = value
        
        # Add self-connections and normalize
        A = A + sp.eye(self.n_nodes)
        rowsum = np.array(A.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        
        # Symmetric normalization
        norm_adj = d_mat.dot(A).dot(d_mat)
        norm_adj = norm_adj.tocoo()
        
        # Convert to torch sparse tensor
        indices = torch.LongTensor([norm_adj.row, norm_adj.col])
        values = torch.FloatTensor(norm_adj.data)
        shape = torch.Size(norm_adj.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def pre_epoch_processing(self):
        """
        Operations to perform before each epoch
        """
        pass

    def post_epoch_processing(self):
        """
        Operations to perform after each epoch
        """
        pass