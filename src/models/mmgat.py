# coding: utf-8
import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from common.abstract_recommender import GeneralRecommender

class HybridAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        
    def attention(self, q, k, v, mask=None):
        B, H, L, D = q.shape
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(D)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)
        return output
        
    def forward(self, x, y=None):
        y = x if y is None else y
        B, L, D = x.shape
        
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(y).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(y).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_output = self.attention(q, k, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, D)
        attn_output = self.o_proj(attn_output)
        x = self.norm1(x + self.dropout(attn_output))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x

class MultimodalFusion(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.feat_embed_dim = config["feat_embed_dim"]
        
        self.image_encoder = nn.Sequential(
            nn.Linear(config["v_feat_dim"], self.feat_embed_dim),
            nn.LayerNorm(self.feat_embed_dim),
            nn.GELU()
        )
        
        self.text_encoder = nn.Sequential(
            nn.Linear(config["t_feat_dim"], self.feat_embed_dim),
            nn.LayerNorm(self.feat_embed_dim),
            nn.GELU()
        )
        
        self.self_attention = HybridAttention(self.feat_embed_dim)
        self.cross_attention = HybridAttention(self.feat_embed_dim)
        self.fusion = nn.Sequential(
            nn.Linear(self.feat_embed_dim * 2, self.feat_embed_dim),
            nn.LayerNorm(self.feat_embed_dim),
            nn.GELU()
        )
        
    def forward(self, image_feat, text_feat):
        # Process individual modalities
        image_emb = self.image_encoder(image_feat).unsqueeze(0)
        text_emb = self.text_encoder(text_feat).unsqueeze(0)
        
        # Self attention refinement
        image_refined = self.self_attention(image_emb)
        text_refined = self.self_attention(text_emb)
        
        # Cross modal attention
        image_enhanced = self.cross_attention(image_refined, text_refined)
        text_enhanced = self.cross_attention(text_refined, image_refined)
        
        # Combine enhanced features
        combined = torch.cat([image_enhanced, text_enhanced], dim=-1)
        fused = self.fusion(combined)
        return fused.squeeze(0), image_enhanced.squeeze(0), text_enhanced.squeeze(0)

class MMGAT(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MMGAT, self).__init__(config, dataset)
        
        self.embedding_dim = config["embedding_size"]
        self.feat_embed_dim = config["feat_embed_dim"]
        self.n_layers = config["n_mm_layers"]
        self.reg_weight = config["reg_weight"]
        self.temperature = 0.07
        
        # Load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        
        # Basic embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_normal_(self.user_embedding.weight)
        nn.init.xavier_normal_(self.item_embedding.weight)
        
        # Modality-specific components
        config["v_feat_dim"] = self.v_feat.shape[1] if self.v_feat is not None else 0
        config["t_feat_dim"] = self.t_feat.shape[1] if self.t_feat is not None else 0
        
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            
        self.fusion_module = MultimodalFusion(config)
        
        # Graph components
        self.norm_adj = self.build_graph().to(self.device)
        self.graph_attention = HybridAttention(self.embedding_dim)
        
        self.to(self.device)
        
    def build_graph(self):
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.interaction_matrix.tolil()
        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()
        
        row_sum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(row_sum + 1e-9, -0.5).flatten()
        d_mat = sp.diags(d_inv)
        norm_adj = d_mat.dot(adj_mat).dot(d_mat)
        
        coo = norm_adj.tocoo()
        values = coo.data
        indices = np.vstack((coo.row, coo.col))
        indices = torch.LongTensor(indices)
        values = torch.FloatTensor(values)
        shape = torch.Size(coo.shape)
        return torch.sparse_coo_tensor(indices, values, shape)
        
    def forward(self):
        # Process multimodal features
        if self.v_feat is not None and self.t_feat is not None:
            modal_embedding, img_enhanced, txt_enhanced = self.fusion_module(
                self.image_embedding.weight,
                self.text_embedding.weight
            )
        else:
            modal_embedding = None
            img_enhanced = None
            txt_enhanced = None
            
        # Graph propagation
        ego_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        all_embeddings = [ego_embeddings]
        
        for _ in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.norm_adj, ego_embeddings)
            refined_embeddings = self.graph_attention(ego_embeddings.unsqueeze(0)).squeeze(0)
            all_embeddings.append(refined_embeddings)
            
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        
        user_embeddings, item_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items])
        
        if modal_embedding is not None:
            item_embeddings = item_embeddings + modal_embedding
            
        return user_embeddings, item_embeddings, img_enhanced, txt_enhanced
        
    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        
        user_embeddings, item_embeddings, img_enhanced, txt_enhanced = self.forward()
        
        u_embeddings = user_embeddings[users]
        pos_embeddings = item_embeddings[pos_items]
        neg_embeddings = item_embeddings[neg_items]
        
        # BPR loss
        pos_scores = torch.sum(u_embeddings * pos_embeddings, dim=1)
        neg_scores = torch.sum(u_embeddings * neg_embeddings, dim=1)
        bpr_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        
        # Contrastive loss
        contrastive_loss = 0.0
        if img_enhanced is not None and txt_enhanced is not None:
            pos_sim = torch.cosine_similarity(
                F.normalize(img_enhanced[pos_items], dim=1),
                F.normalize(txt_enhanced[pos_items], dim=1)
            )
            contrastive_loss = -torch.mean(F.logsigmoid(pos_sim / self.temperature))
            
        # L2 regularization
        reg_loss = self.reg_weight * (
            torch.norm(u_embeddings) +
            torch.norm(pos_embeddings) +
            torch.norm(neg_embeddings)
        )
        
        return bpr_loss + 0.2 * contrastive_loss + reg_loss
        
    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_embeddings, item_embeddings, _, _ = self.forward()
        scores = torch.matmul(user_embeddings[user], item_embeddings.transpose(0, 1))
        return scores