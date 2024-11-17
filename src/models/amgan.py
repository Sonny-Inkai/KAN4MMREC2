# coding: utf-8
# @email: enoche.chow@gmail.com
r"""
MHGT: Multimodal Hierarchical Graph Transformer for Multimedia Recommendations
# Update: 17/11/2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import Parameter
from torch_geometric.nn import GCNConv
from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss

class CrossModalTransformer(nn.Module):
    def __init__(self, embedding_dim, num_heads, num_layers):
        super(CrossModalTransformer, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, text_feat, image_feat, collaborative_feat):
        # Stack features along the sequence dimension for transformer input
        combined_feat = torch.stack((text_feat, image_feat, collaborative_feat), dim=0)
        # Pass through transformer encoder
        output = self.transformer_encoder(combined_feat)
        # Aggregate the output
        return torch.mean(output, dim=0)

class HierarchicalAttention(nn.Module):
    def __init__(self, embedding_dim):
        super(HierarchicalAttention, self).__init__()
        self.attention_layer = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=4)

    def forward(self, query, key, value):
        attn_output, _ = self.attention_layer(query, key, value)
        return attn_output

class GraphFeaturePropagation(nn.Module):
    def __init__(self, embedding_dim):
        super(GraphFeaturePropagation, self).__init__()
        self.gcn1 = GCNConv(embedding_dim, embedding_dim)
        self.gcn2 = GCNConv(embedding_dim, embedding_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.gcn1(x, edge_index))
        x = self.gcn2(x, edge_index)
        return x

class AMGAN(GeneralRecommender):
    def __init__(self, config, dataset):
        super(AMGAN, self).__init__(config, dataset)
        self.num_user = self.n_users
        self.num_item = self.n_items
        self.embedding_dim = config['embedding_size']
        self.reg_weight = config['reg_weight']
        self.contrastive_weight = config['contrastive_weight']

        # User and Item Embeddings
        self.user_embedding = nn.Embedding(self.num_user, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.num_item, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        # Multimodal Embeddings for Items
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.embedding_dim)
            nn.init.xavier_uniform_(self.image_trs.weight)
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.embedding_dim)
            nn.init.xavier_uniform_(self.text_trs.weight)

        # Cross-Modal Transformer Module
        self.cross_modal_transformer = CrossModalTransformer(self.embedding_dim, num_heads=8, num_layers=4)

        # Hierarchical Attention Module
        self.hierarchical_attention = HierarchicalAttention(self.embedding_dim)

        # Graph Feature Propagation Module
        self.graph_feature_propagation = GraphFeaturePropagation(self.embedding_dim)

    def forward(self, user_indices, item_indices, edge_index):
        user_embed = self.user_embedding(user_indices)
        item_embed = self.item_embedding(item_indices)

        # Get multimodal item embeddings
        if self.v_feat is not None:
            image_feat = F.relu(self.image_trs(self.image_embedding(item_indices)))
        else:
            image_feat = torch.zeros_like(item_embed)

        if self.t_feat is not None:
            text_feat = F.relu(self.text_trs(self.text_embedding(item_indices)))
        else:
            text_feat = torch.zeros_like(item_embed)

        # Cross-Modal Transformer Fusion
        combined_item_feat = self.cross_modal_transformer(text_feat, image_feat, item_embed)

        # Hierarchical Attention
        combined_item_feat = self.hierarchical_attention(combined_item_feat.unsqueeze(0), combined_item_feat.unsqueeze(0), combined_item_feat.unsqueeze(0)).squeeze(0)

        # Graph Feature Propagation
        combined_item_feat = self.graph_feature_propagation(combined_item_feat, edge_index)

        return user_embed, combined_item_feat

    def calculate_loss(self, interaction, edge_index):
        user_indices = interaction[0]
        pos_item_indices = interaction[1]
        neg_item_indices = interaction[2]

        # Forward Pass
        user_embed, pos_item_embed = self.forward(user_indices, pos_item_indices, edge_index)
        _, neg_item_embed = self.forward(user_indices, neg_item_indices, edge_index)

        # BPR Loss
        pos_scores = torch.sum(user_embed * pos_item_embed, dim=1)
        neg_scores = torch.sum(user_embed * neg_item_embed, dim=1)
        bpr_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))

        # Contrastive Loss
        contrastive_loss = torch.mean((user_embed - pos_item_embed) ** 2) - torch.mean((user_embed - neg_item_embed) ** 2)
        contrastive_loss = torch.clamp(contrastive_loss, min=0)

        # Regularization Loss
        reg_loss = self.reg_weight * ((self.user_embedding.weight ** 2).mean() + (self.item_embedding.weight ** 2).mean())

        return bpr_loss + self.contrastive_weight * contrastive_loss + reg_loss

    def full_sort_predict(self, interaction, edge_index):
        user_indices = interaction[0]
        user_embed = self.user_embedding(user_indices)
        item_embed = self.item_embedding.weight

        if self.v_feat is not None:
            image_feat = F.relu(self.image_trs(self.image_embedding.weight))
        else:
            image_feat = torch.zeros_like(item_embed)

        if self.t_feat is not None:
            text_feat = F.relu(self.text_trs(self.text_embedding.weight))
        else:
            text_feat = torch.zeros_like(item_embed)

        # Cross-Modal Transformer Fusion for all items
        combined_item_feat = self.cross_modal_transformer(text_feat, image_feat, item_embed)

        # Hierarchical Attention for all items
        combined_item_feat = self.hierarchical_attention(combined_item_feat.unsqueeze(0), combined_item_feat.unsqueeze(0), combined_item_feat.unsqueeze(0)).squeeze(0)

        # Graph Feature Propagation for all items
        combined_item_feat = self.graph_feature_propagation(combined_item_feat, edge_index)

        # Dot product between user and all items
        scores = torch.matmul(user_embed, combined_item_feat.t())
        return scores
