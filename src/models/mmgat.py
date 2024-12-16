import torch
import torch.nn as nn
import torch.nn.functional as F
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
from torch_geometric.nn import GATConv
from common.abstract_recommender import GeneralRecommender

class MMGAT(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MMGAT, self).__init__(config, dataset)
        
        # Configuration
        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.num_heads = config['num_heads']
        self.dropout = config['dropout']
        self.n_layers = config['n_layers']
        self.num_gat_heads = config['num_gat_heads']
        self.temperature = config['temperature']
        self.reg_weight = config['reg_weight']
        
        # Base embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        
        # Modal-specific components
        if self.v_feat is not None:
            self.visual_encoder = ModalFlashEncoder(
                input_dim=self.v_feat.shape[1],
                hidden_dim=self.feat_embed_dim,
                num_heads=self.num_heads
            )
            self.visual_proj = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)
            
        if self.t_feat is not None:
            self.text_encoder = ModalFlashEncoder(
                input_dim=self.t_feat.shape[1],
                hidden_dim=self.feat_embed_dim,
                num_heads=self.num_heads
            )
            self.text_proj = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)

        # Cross-modal fusion
        self.modal_fusion = CrossModalFusion(
            dim=self.feat_embed_dim,
            num_heads=self.num_heads
        )
        
        # Graph attention layers
        self.gat_layers = nn.ModuleList([
            GraphAttentionLayer(
                in_dim=self.embedding_dim,
                out_dim=self.embedding_dim,
                num_heads=self.num_gat_heads
            ) for _ in range(self.n_layers)
        ])
        
        # Feature transformation
        self.feature_fusion = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
        self.init_weights()

    def init_weights(self):
        """Initialize weights using Xavier initialization"""
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.xavier_uniform_(module.weight)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    nn.init.zeros_(module.bias)

    def encode_modalities(self, items):
        """Encode items using both visual and textual modalities"""
        modal_embeddings = []
        
        if self.v_feat is not None:
            visual_features = self.v_feat[items]
            visual_emb = self.visual_encoder(visual_features)
            modal_embeddings.append(visual_emb)
            
        if self.t_feat is not None:
            text_features = self.t_feat[items]
            text_emb = self.text_encoder(text_features)
            modal_embeddings.append(text_emb)
            
        if len(modal_embeddings) > 1:
            # Fuse modalities using cross-attention
            fused_embedding = self.modal_fusion(modal_embeddings[0], modal_embeddings[1])
        else:
            fused_embedding = modal_embeddings[0]
            
        return fused_embedding, modal_embeddings

    def forward(self, users, items):
        """
        Forward pass of the model
        Args:
            users: User indices
            items: Item indices
        """
        user_emb = self.user_embedding(users)
        item_emb = self.item_embedding(items)
        
        # Get modal embeddings
        modal_emb, modal_features = self.encode_modalities(items)
        
        # Combine base embeddings with modal information
        item_emb = self.feature_fusion(torch.cat([item_emb, modal_emb], dim=-1))
        
        # Apply graph attention for message passing
        combined_emb = torch.cat([user_emb, item_emb], dim=0)
        for gat_layer in self.gat_layers:
            combined_emb = gat_layer(combined_emb)
        
        # Split back user and item embeddings
        user_final, item_final = torch.split(combined_emb, [user_emb.size(0), item_emb.size(0)])
        
        return user_final, item_final, modal_features

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        
        # Get embeddings for positive and negative interactions
        user_emb, pos_emb, modal_features = self.forward(users, pos_items)
        _, neg_emb, _ = self.forward(users, neg_items)
        
        # BPR loss
        pos_scores = torch.sum(user_emb * pos_emb, dim=-1)
        neg_scores = torch.sum(user_emb * neg_emb, dim=-1)
        bpr_loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores)))
        
        # Contrastive loss between modalities
        contra_loss = 0.0
        if len(modal_features) > 1:
            sim_matrix = torch.matmul(modal_features[0], modal_features[1].t())
            sim_matrix = sim_matrix / self.temperature
            labels = torch.arange(sim_matrix.size(0)).to(self.device)
            contra_loss = F.cross_entropy(sim_matrix, labels)
        
        # L2 regularization
        reg_loss = self.reg_weight * (
            torch.norm(user_emb) + torch.norm(pos_emb) + torch.norm(neg_emb)
        )
        
        return bpr_loss + 0.1 * contra_loss + reg_loss

    def predict(self, interaction):
        user = interaction[0]
        item = interaction[1]
        
        user_emb, item_emb, _ = self.forward(user, item)
        scores = torch.sum(user_emb * item_emb, dim=1)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_emb = self.user_embedding(user)
        
        # Calculate scores for all items
        all_items = torch.arange(self.n_items).to(self.device)
        all_item_emb = self.item_embedding(all_items)
        
        # Get modal embeddings for all items
        modal_emb, _ = self.encode_modalities(all_items)
        all_item_emb = self.feature_fusion(torch.cat([all_item_emb, modal_emb], dim=-1))
        
        scores = torch.matmul(user_emb, all_item_emb.t())
        return scores

class ModalFlashEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.qkv_proj = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Project input features
        x = self.input_proj(x)
        
        # Generate QKV matrices
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, -1, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, S, D)
        
        # Apply Flash Attention
        output = flash_attn_qkvpacked_func(qkv, dropout_p=0.0)
        output = output.reshape(batch_size, -1, self.num_heads * self.head_dim)
        
        return output

class CrossModalFusion(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.output_proj = nn.Linear(dim, dim)
        
    def forward(self, x1, x2):
        batch_size = x1.shape[0]
        
        # Project to Q, K, V
        q = self.q_proj(x1).reshape(batch_size, -1, self.num_heads, self.head_dim)
        k = self.k_proj(x2).reshape(batch_size, -1, self.num_heads, self.head_dim)
        v = self.v_proj(x2).reshape(batch_size, -1, self.num_heads, self.head_dim)
        
        # Transpose for attention
        q = q.transpose(1, 2)  # (B, H, S, D)
        k = k.transpose(1, 2)  # (B, H, S, D)
        v = v.transpose(1, 2)  # (B, H, S, D)
        
        # Apply Flash Attention
        output = flash_attn_func(q, k, v, dropout_p=0.0)
        
        # Reshape and project output
        output = output.transpose(1, 2).reshape(batch_size, -1, self.dim)
        output = self.output_proj(output)
        
        return output

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads):
        super().__init__()
        self.gat = GATConv(in_dim, out_dim // num_heads, heads=num_heads)
        
    def forward(self, x):
        return self.gat(x)