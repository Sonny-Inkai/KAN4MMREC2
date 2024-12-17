import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity
import math
from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss

class MMGAT(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MMGAT, self).__init__(config, dataset)
        
        # Hyperparameters
        self.embedding_dim = 64  
        self.feat_embed_dim = 64
        self.num_heads = 4
        self.dropout = 0.2
        self.n_layers = 2
        self.temperature = 0.2
        self.reg_weight = 1e-4
        self.beta = 0.5
        self.mm_fusion_mode = 'gate'
        self.num_routing = 3
        
        self.n_nodes = self.n_users + self.n_items
        
        # Embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        # Feature transformations
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)
            self.image_gate = nn.Linear(self.feat_embed_dim * 2, 1)
            
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)
            self.text_gate = nn.Linear(self.feat_embed_dim * 2, 1)

        # Flash attention layers
        self.flash_attn = nn.ModuleList([
            FlashMultiHeadAttention(self.embedding_dim, self.num_heads, self.dropout)
            for _ in range(self.n_layers)
        ])
        
        # Graph attention layers
        self.graph_attn = nn.ModuleList([
            GraphAttentionLayer(self.embedding_dim, self.dropout)
            for _ in range(self.n_layers)
        ])
        
        # Capsule network for routing
        self.routing_weights = nn.Parameter(torch.randn(self.num_routing, self.feat_embed_dim, self.feat_embed_dim))
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(self.embedding_dim)
        self.layer_norm2 = nn.LayerNorm(self.feat_embed_dim)
        
        # MLP for fusion
        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.feat_embed_dim * 2, self.feat_embed_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.feat_embed_dim, self.feat_embed_dim)
        )
        
    def routing_mechanism(self, features):
        b = torch.zeros(features.size(0), self.num_routing, device=features.device)
        
        for i in range(self.num_routing):
            c = F.softmax(b, dim=1)
            c = c.unsqueeze(2)
            
            weighted_pred = torch.matmul(features.unsqueeze(1), self.routing_weights)
            weighted_sum = (c * weighted_pred).sum(dim=1)
            
            if i < self.num_routing - 1:
                b = b + (weighted_pred * features.unsqueeze(1)).sum(dim=-1)
                
        return weighted_sum

    def multimodal_fusion(self, visual_feat, text_feat):
        if self.mm_fusion_mode == 'gate':
            if visual_feat is not None and text_feat is not None:
                # Gating mechanism
                visual_gate = torch.sigmoid(self.image_gate(torch.cat([visual_feat, text_feat], dim=-1)))
                text_gate = torch.sigmoid(self.text_gate(torch.cat([text_feat, visual_feat], dim=-1)))
                
                # Normalized fusion
                fused_feat = visual_gate * visual_feat + text_gate * text_feat
                fused_feat = self.layer_norm2(fused_feat)
            elif visual_feat is not None:
                fused_feat = visual_feat
            else:
                fused_feat = text_feat
                
            return self.fusion_mlp(torch.cat([fused_feat, fused_feat], dim=-1))
        else:
            if visual_feat is not None and text_feat is not None:
                return (visual_feat + text_feat) / 2
            elif visual_feat is not None:
                return visual_feat
            else:
                return text_feat

    def forward(self, adj_matrix=None):
        # Process multimodal features
        visual_feat = None
        text_feat = None
        
        if self.v_feat is not None:
            visual_feat = self.image_trs(self.image_embedding.weight)
            visual_feat = self.routing_mechanism(visual_feat)
            
        if self.t_feat is not None:
            text_feat = self.text_trs(self.text_embedding.weight)
            text_feat = self.routing_mechanism(text_feat)
        
        # Fuse multimodal features
        mm_feat = self.multimodal_fusion(visual_feat, text_feat)
        
        # Process user-item graph
        user_embed = self.user_embedding.weight
        item_embed = self.item_id_embedding.weight
        
        all_embed = torch.cat([user_embed, item_embed], dim=0)
        all_embed = self.layer_norm1(all_embed)
        
        # Apply flash attention and graph attention
        for flash_layer, graph_layer in zip(self.flash_attn, self.graph_attn):
            all_embed = flash_layer(all_embed, all_embed, all_embed)
            if adj_matrix is not None:
                all_embed = graph_layer(all_embed, adj_matrix)
            
        user_embed, item_embed = torch.split(all_embed, [self.n_users, self.n_items])
        
        # Combine with multimodal features
        if mm_feat is not None:
            item_embed = item_embed + mm_feat
            
        return user_embed, item_embed
        
    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        
        user_embeddings, item_embeddings = self.forward()
        
        # Mirror gradient
        with torch.no_grad():
            user_embeddings_mirror = user_embeddings.clone()
            item_embeddings_mirror = item_embeddings.clone()
        
        user_e = user_embeddings[users]
        pos_e = item_embeddings[pos_items]
        neg_e = item_embeddings[neg_items]
        
        # First forward pass
        pos_scores = torch.sum(user_e * pos_e, dim=1)
        neg_scores = torch.sum(user_e * neg_e, dim=1)
        
        main_loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores)))
        
        # Mirror gradient second pass
        user_e_mirror = user_embeddings_mirror[users]
        pos_e_mirror = item_embeddings_mirror[pos_items]
        neg_e_mirror = item_embeddings_mirror[neg_items]
        
        pos_scores_mirror = torch.sum(user_e_mirror * pos_e_mirror, dim=1)
        neg_scores_mirror = torch.sum(user_e_mirror * neg_e_mirror, dim=1)
        
        mirror_loss = -torch.mean(torch.log(torch.sigmoid(pos_scores_mirror - neg_scores_mirror)))
        
        # Contrastive loss for multimodal features
        if self.v_feat is not None:
            v_feat = self.image_trs(self.image_embedding.weight[pos_items])
            v_loss = 1 - cosine_similarity(v_feat, pos_e, dim=-1).mean()
        else:
            v_loss = 0.0
            
        if self.t_feat is not None:
            t_feat = self.text_trs(self.text_embedding.weight[pos_items])
            t_loss = 1 - cosine_similarity(t_feat, pos_e, dim=-1).mean()
        else:
            t_loss = 0.0
            
        # L2 regularization
        reg_loss = self.reg_weight * (
            torch.norm(user_e) +
            torch.norm(pos_e) +
            torch.norm(neg_e)
        )
        
        return main_loss + self.beta * mirror_loss + reg_loss + v_loss + t_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        
        user_embeddings, item_embeddings = self.forward()
        
        user_e = user_embeddings[user]
        scores = torch.matmul(user_e, item_embeddings.transpose(0, 1))
        
        return scores

class FlashMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = dropout
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        
        self.scale = math.sqrt(self.head_dim)
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # Project and reshape
        q = self.q_proj(q).view(batch_size, -1, self.num_heads, self.head_dim)
        k = self.k_proj(k).view(batch_size, -1, self.num_heads, self.head_dim)
        v = self.v_proj(v).view(batch_size, -1, self.num_heads, self.head_dim)
        
        # Transpose for attention computation
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
            
        attn = F.softmax(attn, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        
        # Compute output
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous()
        out = out.view(batch_size, -1, self.d_model)
        
        return self.o_proj(out)

class GraphAttentionLayer(nn.Module):
    def __init__(self, input_dim, dropout):
        super().__init__()
        self.dropout = dropout
        
        # Edge attention parameters
        self.a = nn.Parameter(torch.empty(size=(2*input_dim, 1)))
        nn.init.xavier_uniform_(self.a.data)
        
        self.leakyrelu = nn.LeakyReLU()
        
    def forward(self, h, adj):
        N = h.size()[0]
        
        # Compute attention coefficients
        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, N, -1)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        
        # Mask attention coefficients
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        # Apply attention
        h_prime = torch.matmul(attention, h)
        
        return h_prime