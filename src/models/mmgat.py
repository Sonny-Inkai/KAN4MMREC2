import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np

from common.abstract_recommender import GeneralRecommender
from common.loss import EmbLoss

class FlashAttention(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.1, max_seq_len=2048):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.dropout = dropout
        
        # Chunked attention parameters
        self.chunk_size = 256  # Optimize for memory
        self.max_seq_len = max_seq_len
        
    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)  # B, N, H, D
        
        # Chunked attention computation
        out = torch.zeros_like(q)
        softmax_max = torch.full((B, self.num_heads, N), float('-inf'), device=x.device)
        softmax_sum = torch.zeros((B, self.num_heads, N), device=x.device)
        
        for chunk_idx in range(0, N, self.chunk_size):
            chunk_end = min(chunk_idx + self.chunk_size, N)
            cur_chunk = slice(chunk_idx, chunk_end)
            
            # Current chunk attention scores
            attn_weights = torch.matmul(q, k[:, cur_chunk].transpose(-2, -1)) * self.scale
            
            if mask is not None:
                attn_weights = attn_weights.masked_fill(mask[:, :, cur_chunk] == 0, float('-inf'))
            
            # Update running max and sum for stable softmax
            chunk_max = attn_weights.max(dim=-1, keepdim=True)[0]
            exp_weights = torch.exp(attn_weights - chunk_max)
            chunk_sum = exp_weights.sum(dim=-1, keepdim=True)
            
            # Update output with current chunk
            exp_values = torch.matmul(exp_weights, v[:, cur_chunk])
            out = out + exp_values * chunk_sum
            
            # Update running statistics
            softmax_max = torch.maximum(softmax_max, chunk_max.squeeze(-1))
            softmax_sum = softmax_sum + chunk_sum.squeeze(-1)
        
        out = out / softmax_sum.unsqueeze(-1)
        out = self.proj(out.reshape(B, N, -1))
        return out

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

class MMGAT(GeneralRecommender):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        
        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.n_layers = config['n_layers']
        self.dropout = config['dropout']
        self.reg_weight = config['reg_weight']
        self.n_heads = 4
        self.temperature = 0.07
        
        self.n_nodes = self.n_users + self.n_items
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.norm_adj = self.get_norm_adj_mat().to(self.device)
        
        # Embeddings with better initialization
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        self.apply(self._init_weights)
        
        # Enhanced modal encoders
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
        
        # Modal fusion with Flash Attention
        self.modal_fusion = nn.ModuleList([
            ModalFusionBlock(self.feat_embed_dim, self.n_heads)
            for _ in range(2)
        ])
        
        # Graph convolution
        self.gc_layers = nn.ModuleList([
            nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
            for _ in range(self.n_layers)
        ])
        
        self.to(self.device)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
            nn.init.trunc_normal_(m.weight, std=0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self):
        # Modal feature extraction and fusion
        modal_features = []
        if self.v_feat is not None:
            img_feat = self.image_encoder(self.image_embedding.weight)
            modal_features.append(img_feat)
            
        if self.t_feat is not None:
            txt_feat = self.text_encoder(self.text_embedding.weight)
            modal_features.append(txt_feat)
        
        # Enhanced modal fusion with Flash Attention
        if len(modal_features) > 0:
            if len(modal_features) > 1:
                modal_stack = torch.stack(modal_features, dim=1)
                for fusion_layer in self.modal_fusion:
                    modal_stack = fusion_layer(modal_stack)
                fused_features = torch.mean(modal_stack, dim=1)
            else:
                fused_features = modal_features[0]
        else:
            fused_features = None
        
        # Graph convolution with residual connections
        x = torch.cat([self.user_embedding.weight, self.item_embedding.weight])
        all_embs = [x]
        
        for i in range(self.n_layers):
            x = torch.sparse.mm(self.norm_adj, x)
            x = self.gc_layers[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
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
        
        # Modal alignment loss
        if hasattr(self, 'image_embedding') and hasattr(self, 'text_embedding'):
            img_feat = self.image_encoder(self.image_embedding.weight)
            txt_feat = self.text_encoder(self.text_embedding.weight)
            
            img_feat = F.normalize(img_feat, dim=-1)
            txt_feat = F.normalize(txt_feat, dim=-1)
            
            modal_sim = torch.matmul(img_feat, txt_feat.t()) / self.temperature
            modal_loss = -torch.mean(torch.diag(F.log_softmax(modal_sim, dim=1)))
            loss = loss + 0.1 * modal_loss
        
        # L2 regularization
        reg_loss = self.reg_weight * (
            torch.norm(u_emb) / len(users) +
            torch.norm(pos_emb) / len(pos_items) +
            torch.norm(neg_emb) / len(neg_items)
        )
        
        return loss + reg_loss
    
    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_emb, item_emb = self.forward()
        scores = torch.matmul(user_emb[user], item_emb.transpose(0, 1))
        return scores

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
        return torch.sparse_coo_tensor(indices, values, torch.Size((self.n_nodes, self.n_nodes)))