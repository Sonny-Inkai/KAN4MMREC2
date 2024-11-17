# coding: utf-8
# @email: enoche.chow@gmail.com
r"""
CMFFN: Contrastive Multi-Level Feature Fusion Network for Robust Multimedia Recommendation
# Update: 17/11/2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import Parameter
from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss

class MultiLevelFeatureFusion(nn.Module):
    def __init__(self, embedding_dim):
        super(MultiLevelFeatureFusion, self).__init__()
        self.embedding_dim = embedding_dim
        self.weight_text = Parameter(torch.Tensor(1))
        self.weight_image = Parameter(torch.Tensor(1))
        self.weight_collaborative = Parameter(torch.Tensor(1))
        nn.init.constant_(self.weight_text, 1.0)
        nn.init.constant_(self.weight_image, 1.0)
        nn.init.constant_(self.weight_collaborative, 1.0)

    def forward(self, text_feat, image_feat, collaborative_feat):
        # Weighted average fusion of different modalities
        combined_feat = (
            self.weight_text * text_feat +
            self.weight_image * image_feat +
            self.weight_collaborative * collaborative_feat
        ) / (self.weight_text + self.weight_image + self.weight_collaborative)
        return combined_feat


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

        # Feature Fusion Module
        self.fusion = MultiLevelFeatureFusion(self.embedding_dim)

    def forward(self, user_indices, item_indices):
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

        # Fusion of multimodal features
        combined_item_feat = self.fusion(text_feat, image_feat, item_embed)
        return user_embed, combined_item_feat

    def calculate_loss(self, interaction):
        user_indices = interaction[0]
        pos_item_indices = interaction[1]
        neg_item_indices = interaction[2]

        # Forward Pass
        user_embed, pos_item_embed = self.forward(user_indices, pos_item_indices)
        _, neg_item_embed = self.forward(user_indices, neg_item_indices)

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

    def full_sort_predict(self, interaction):
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

        # Fusion of multimodal features for all items
        combined_item_feat = self.fusion(text_feat, image_feat, item_embed)

        # Dot product between user and all items
        scores = torch.matmul(user_embed, combined_item_feat.t())
        return scores
