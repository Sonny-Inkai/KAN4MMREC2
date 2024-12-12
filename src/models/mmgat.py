import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from common.abstract_recommender import GeneralRecommender
from common.loss import EmbLoss

class MMGAT(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MMGAT, self).__init__(config, dataset)
        
        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.reg_weight = config['reg_weight']
        self.device = config['device']
        self.dropout = config['dropout']
        self.hidden_size = self.embedding_dim * 2
        
        # Core embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        
        # Initialize modality embeddings and projections
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_encoder = nn.Sequential(
                nn.Linear(self.v_feat.shape[1], self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_size, self.feat_embed_dim)
            )
            
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_encoder = nn.Sequential(
                nn.Linear(self.t_feat.shape[1], self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_size, self.feat_embed_dim)
            )

        # Modality fusion
        if self.v_feat is not None and self.t_feat is not None:
            self.modal_attention = nn.Sequential(
                nn.Linear(self.feat_embed_dim * 2, 64),
                nn.ReLU(),
                nn.Linear(64, 2),
                nn.Softmax(dim=1)
            )
            
            fusion_input_dim = self.feat_embed_dim * 2
        else:
            fusion_input_dim = self.feat_embed_dim
            
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_dim, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.embedding_dim)
        )
        
        # Item representation enhancement
        self.item_enhance = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.embedding_dim)
        )
        
        # User preference projection
        self.user_projection = nn.Sequential(
            nn.Linear(self.embedding_dim, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.embedding_dim)
        )

        # Loss functions
        self.criterion = nn.BCEWithLogitsLoss()
        self.reg_loss = EmbLoss()
        
    def encode_modalities(self):
        image_feat = text_feat = None
        
        if self.v_feat is not None:
            image_feat = self.image_encoder(self.image_embedding.weight)
            image_feat = F.normalize(image_feat, p=2, dim=1)
            
        if self.t_feat is not None:
            text_feat = self.text_encoder(self.text_embedding.weight)
            text_feat = F.normalize(text_feat, p=2, dim=1)
            
        if image_feat is not None and text_feat is not None:
            # Concatenate features for attention
            concat_feat = torch.cat([image_feat, text_feat], dim=1)
            attention_weights = self.modal_attention(concat_feat)
            
            # Apply attention weights
            modal_features = torch.stack([image_feat, text_feat], dim=1)
            attention_weights = attention_weights.unsqueeze(-1)
            fused_features = (modal_features * attention_weights).sum(dim=1)
            
            # Concatenate original features with attention output
            final_features = torch.cat([image_feat, text_feat], dim=1)
            
        elif image_feat is not None:
            final_features = image_feat
        else:
            final_features = text_feat
            
        return self.fusion_layer(final_features)
        
    def forward(self):
        # Get modal features
        item_modal_embeds = self.encode_modalities()
        
        # Process item embeddings
        item_id_embeds = self.item_embedding.weight
        enhanced_item_embeds = self.item_enhance(
            torch.cat([item_id_embeds, item_modal_embeds], dim=1)
        )
        
        # Process user embeddings
        user_embeds = self.user_projection(self.user_embedding.weight)
        
        return user_embeds, enhanced_item_embeds

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        
        user_embeds, item_embeds = self.forward()
        
        user_e = user_embeds[users]
        pos_e = item_embeds[pos_items]
        neg_e = item_embeds[neg_items]
        
        # BPR loss
        pos_scores = (user_e * pos_e).sum(dim=1)
        neg_scores = (user_e * neg_e).sum(dim=1)
        
        loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8))
        
        # Modal consistency loss
        modal_loss = 0
        if self.v_feat is not None and self.t_feat is not None:
            image_feat = self.image_encoder(self.image_embedding.weight)
            text_feat = self.text_encoder(self.text_embedding.weight)
            
            pos_modal_sim = F.cosine_similarity(
                image_feat[pos_items],
                text_feat[pos_items]
            )
            modal_loss = 1 - pos_modal_sim.mean()
        
        # L2 regularization
        reg_loss = self.reg_weight * (
            user_e.norm(2).pow(2) +
            pos_e.norm(2).pow(2) +
            neg_e.norm(2).pow(2)
        ) / float(len(users))
        
        return loss + 0.1 * modal_loss + reg_loss

    def predict(self, interaction):
        user = interaction[0]
        item = interaction[1]
        
        user_embeds, item_embeds = self.forward()
        
        u_embeddings = user_embeds[user]
        i_embeddings = item_embeds[item]
        
        return torch.mul(u_embeddings, i_embeddings).sum(dim=1)

    def full_sort_predict(self, interaction):
        user = interaction[0]
        
        user_embeds, item_embeds = self.forward()
        u_embeddings = user_embeds[user]
        
        scores = torch.matmul(u_embeddings, item_embeds.t())
        return scores