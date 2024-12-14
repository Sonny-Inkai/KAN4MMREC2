import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss, L2Loss
from utils.utils import build_sim, compute_normalized_laplacian
class MMGAT(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MMGAT, self).__init__(config, dataset)
        
        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.n_layers = config['n_layers']
        self.dropout = config['dropout']
        self.reg_weight = config['reg_weight']
        
        # Dynamic preference modeling
        self.user_embedding = TimeAwareEmbedding(
            self.n_users, 
            self.embedding_dim,
            config['time_spans']
        )
        
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        
        # Hierarchical feature extractors
        if self.v_feat is not None:
            self.visual_encoder = HierarchicalEncoder(
                self.v_feat.shape[1],
                [self.feat_embed_dim * 2, self.feat_embed_dim],
                self.dropout
            )
            
        if self.t_feat is not None:
            self.text_encoder = HierarchicalEncoder(
                self.t_feat.shape[1],
                [self.feat_embed_dim * 2, self.feat_embed_dim],
                self.dropout
            )
        
        # Cross-modal attention
        self.modal_attention = CrossModalAttention(
            self.feat_embed_dim,
            num_heads=4
        )
        
        # Dynamic preference aggregator
        self.preference_aggregator = PreferenceAggregator(
            self.embedding_dim,
            self.feat_embed_dim,
            self.dropout
        )
        
        self.criterion = nn.BCEWithLogitsLoss()

    def extract_temporal_features(self, users, timestamps):
        # Extract time-aware user embeddings
        user_emb = self.user_embedding(users, timestamps)
        return user_emb
    
    def extract_modal_features(self, items):
        modal_features = []
        
        if self.v_feat is not None:
            visual_features = self.visual_encoder(self.v_feat[items])
            modal_features.append(visual_features)
            
        if self.t_feat is not None:
            text_features = self.text_encoder(self.t_feat[items])
            modal_features.append(text_features)
            
        if len(modal_features) > 1:
            # Apply cross-modal attention
            modal_features = self.modal_attention(modal_features[0], modal_features[1])
        else:
            modal_features = modal_features[0]
            
        return modal_features

    def forward(self, users, items, timestamps=None):
        # Get time-aware user embeddings
        user_emb = self.extract_temporal_features(users, timestamps)
        
        # Get item embeddings and modal features
        item_emb = self.item_embedding(items)
        modal_features = self.extract_modal_features(items)
        
        # Aggregate preferences
        output = self.preference_aggregator(
            user_emb,
            item_emb,
            modal_features
        )
        
        return output

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        timestamps = interaction[3] if len(interaction) > 3 else None

        # Forward pass for positive and negative samples
        pos_scores = self.forward(users, pos_items, timestamps)
        neg_scores = self.forward(users, neg_items, timestamps)

        # Calculate BPR loss
        loss = -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()

        # Add regularization for temporal embeddings
        temporal_reg = self.user_embedding.get_temporal_regularization()
        
        return loss + self.reg_weight * temporal_reg

    def full_sort_predict(self, interaction):
        users = interaction[0]
        timestamps = interaction[1] if len(interaction) > 1 else None
        
        # Get user embeddings
        user_emb = self.extract_temporal_features(users, timestamps)
        
        # Calculate scores for all items
        all_items = torch.arange(self.n_items).to(self.device)
        item_emb = self.item_embedding(all_items)
        modal_features = self.extract_modal_features(all_items)
        
        scores = self.preference_aggregator(
            user_emb.unsqueeze(1).expand(-1, self.n_items, -1),
            item_emb.unsqueeze(0).expand(len(users), -1, -1),
            modal_features.unsqueeze(0).expand(len(users), -1, -1)
        )
        
        return scores

class TimeAwareEmbedding(nn.Module):
    def __init__(self, num_users, embedding_dim, time_spans):
        super().__init__()
        self.base_embedding = nn.Embedding(num_users, embedding_dim)
        self.time_embedding = nn.Linear(1, embedding_dim)
        self.time_spans = time_spans
        
    def forward(self, users, timestamps=None):
        base_emb = self.base_embedding(users)
        
        if timestamps is not None:
            time_features = self.time_embedding(timestamps.unsqueeze(-1))
            return base_emb + time_features
        return base_emb
        
    def get_temporal_regularization(self):
        return torch.norm(self.time_embedding.weight)

class HierarchicalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.LayerNorm(dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = dim
            
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.layers(x)

class CrossModalAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        
    def forward(self, x1, x2):
        attn_output, _ = self.attention(x1, x2, x2)
        return attn_output

class PreferenceAggregator(nn.Module):
    def __init__(self, embedding_dim, feat_dim, dropout):
        super().__init__()
        
        self.attention = nn.Sequential(
            nn.Linear(embedding_dim + feat_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1)
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(embedding_dim + feat_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, user_emb, item_emb, modal_features):
        # Calculate attention weights
        combined = torch.cat([item_emb, modal_features], dim=-1)
        attention_weights = self.attention(combined).sigmoid()
        
        # Weighted combination
        weighted_features = attention_weights * modal_features
        
        # Final fusion
        output = self.fusion(torch.cat([user_emb, weighted_features], dim=-1))
        return (user_emb * output).sum(dim=-1)