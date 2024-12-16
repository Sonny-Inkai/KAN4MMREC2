import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from common.abstract_recommender import GeneralRecommender

class MMGAT(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MMGAT, self).__init__(config, dataset)
        
        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.num_heads = config['n_heads']
        self.dropout = config['dropout']
        self.n_layers = config['n_layers']
        self.temperature = config['temperature']
        self.reg_weight = config['reg_weight']
        
        # Base embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        
        # Modal-specific components
        if self.v_feat is not None:
            self.visual_encoder = ModalEncoder(
                input_dim=self.v_feat.shape[1],
                hidden_dim=self.feat_embed_dim,
                num_heads=self.num_heads
            )
            self.visual_proj = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)
            
        if self.t_feat is not None:
            self.text_encoder = ModalEncoder(
                input_dim=self.t_feat.shape[1],
                hidden_dim=self.feat_embed_dim,
                num_heads=self.num_heads
            )
            self.text_proj = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)
            
        # Cross-modal fusion
        self.modal_fusion = CrossModalFusion(
            dim=self.feat_embed_dim,
            num_heads=self.num_heads,
            dropout=self.dropout
        )
        
        # Graph attention layers
        self.gat_layers = nn.ModuleList([
            GATConv(self.embedding_dim, self.embedding_dim, heads=self.num_heads, dropout=self.dropout)
            for _ in range(self.n_layers)
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
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.xavier_uniform_(module.weight)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    nn.init.zeros_(module.bias)

    def encode_modalities(self, items):
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
            fused_embedding = self.modal_fusion(modal_embeddings[0], modal_embeddings[1])
        else:
            fused_embedding = modal_embeddings[0]
            
        return fused_embedding, modal_embeddings

    def forward(self, users, items):
        user_emb = self.user_embedding(users)
        item_emb = self.item_embedding(items)
        
        # Get modal embeddings
        modal_emb, modal_features = self.encode_modalities(items)
        
        # Combine item embeddings with modal information
        item_emb = self.feature_fusion(torch.cat([item_emb, modal_emb], dim=-1))
        
        # Apply graph attention
        combined_emb = torch.cat([user_emb, item_emb], dim=0)
        for gat_layer in self.gat_layers:
            combined_emb = gat_layer(combined_emb)
            combined_emb = F.relu(combined_emb)
            combined_emb = F.dropout(combined_emb, p=self.dropout, training=self.training)
        
        user_final, item_final = torch.split(combined_emb, [user_emb.size(0), item_emb.size(0)])
        
        return user_final, item_final, modal_features

class ModalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        
    def forward(self, x):
        x = self.input_proj(x)
        x = x.unsqueeze(1)  # Add sequence dimension
        x = self.encoder(x)
        return x.squeeze(1)  # Remove sequence dimension

class CrossModalFusion(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim)
        )
        
    def forward(self, x1, x2):
        # Self attention
        x1 = x1.unsqueeze(1)
        x2 = x2.unsqueeze(1)
        
        # Cross attention
        attn_output, _ = self.attention(x1, x2, x2)
        x = x1 + attn_output
        x = self.norm1(x)
        
        # Feed forward
        ff_output = self.ffn(x)
        x = x + ff_output
        x = self.norm2(x)
        
        return x.squeeze(1)