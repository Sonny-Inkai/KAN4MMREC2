

import os
import copy
import random
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity

from common.abstract_recommender import GeneralRecommender
from common.loss import EmbLoss
class MMGAT(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MMGAT, self).__init__(config, dataset)
        
        # Hyperparameters
        self.embedding_dim = 64
        self.feat_embed_dim = 64
        self.dropout_rate = 0.2
        self.lr = 0.001
        self.weight_decay = 1e-4
        
        # Dataset-specific parameters
        self.n_users = dataset.num_users
        self.n_items = dataset.num_items
        
        # Embedding layers
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        
        # Attention mechanism
        self.attention_layer = nn.MultiheadAttention(embed_dim=self.embedding_dim, num_heads=4, dropout=self.dropout_rate)
        
        # Feature projection layers
        self.textual_proj = nn.Linear(self.feat_embed_dim, self.embedding_dim)
        self.visual_proj = nn.Linear(self.feat_embed_dim, self.embedding_dim)
        
        # Graph attention layer (GAT)
        self.gat_layer = nn.Linear(self.embedding_dim, self.embedding_dim)
        
        # Dropout
        self.dropout = nn.Dropout(self.dropout_rate)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
    
    def forward(self, user, item, textual_feat, visual_feat):
        # User and item embeddings
        user_emb = self.user_embedding(user)
        item_emb = self.item_embedding(item)
        
        # Project multimodal features
        text_emb = self.textual_proj(textual_feat)
        visual_emb = self.visual_proj(visual_feat)
        
        # Combine embeddings
        combined_emb = user_emb + item_emb + text_emb + visual_emb
        
        # Apply attention
        attn_output, _ = self.attention_layer(combined_emb.unsqueeze(0), combined_emb.unsqueeze(0), combined_emb.unsqueeze(0))
        attn_output = attn_output.squeeze(0)
        
        # Apply GAT
        gat_output = F.relu(self.gat_layer(attn_output))
        
        return gat_output
    
    def calculate_loss(self, user, item, label, textual_feat, visual_feat):
        # Forward pass
        logits = self.forward(user, item, textual_feat, visual_feat)
        
        # Binary cross-entropy loss
        bce_loss = F.binary_cross_entropy_with_logits(logits, label)
        
        # Regularization term (L2)
        l2_reg = self.weight_decay * (self.user_embedding.weight.norm(2) + self.item_embedding.weight.norm(2))
        
        # Final loss
        loss = bce_loss + l2_reg
        return loss
    
    def full_sort_predict(self, user, textual_feat, visual_feat):
        # Generate predictions for all items
        user_emb = self.user_embedding(user)
        all_items_emb = self.item_embedding.weight
        
        # Project multimodal features
        text_emb = self.textual_proj(textual_feat)
        visual_emb = self.visual_proj(visual_feat)
        
        # Combine embeddings
        combined_emb = user_emb.unsqueeze(1) + all_items_emb + text_emb + visual_emb
        
        # Apply attention
        attn_output, _ = self.attention_layer(combined_emb.unsqueeze(0), combined_emb.unsqueeze(0), combined_emb.unsqueeze(0))
        attn_output = attn_output.squeeze(0)
        
        # Apply GAT
        gat_output = F.relu(self.gat_layer(attn_output))
        scores = torch.matmul(gat_output, all_items_emb.T)
        
        return scores
