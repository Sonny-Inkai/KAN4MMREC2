
import torch
import torch.nn as nn
import torch.nn.functional as F
from common.abstract_recommender import GeneralRecommender

class LightMMRec(GeneralRecommender):
    def __init__(self, config, dataset):
        super(LightMMRec, self).__init__(config, dataset)
        
        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.n_layers = config['n_layers']
        self.reg_weight = config['reg_weight']
        self.dropout = config['dropout']
        self.cache_size = config['cache_size']
        self.tau = config['temperature']
        
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        self.available_modalities = []
        self.modality_dims = []
        
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_fcn = self.build_fcn(self.v_feat.shape[1])
            self.available_modalities.append('image')
            self.modality_dims.append(self.feat_embed_dim)
            
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_fcn = self.build_fcn(self.t_feat.shape[1])
            self.available_modalities.append('text')
            self.modality_dims.append(self.feat_embed_dim)

        self.n_modalities = len(self.available_modalities)
        
        self.modality_router = self.build_router()
        self.fusion_networks = nn.ModuleList([
            self.build_fusion_layer() for _ in range(self.n_modalities)
        ])
        
        self.feature_cache = {}

    def build_fcn(self, input_dim):
        return nn.Sequential(
            nn.Linear(input_dim, self.feat_embed_dim * 2),
            nn.LayerNorm(self.feat_embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.feat_embed_dim * 2, self.feat_embed_dim),
            nn.LayerNorm(self.feat_embed_dim)
        )

    def build_router(self):
        return nn.Sequential(
            nn.Linear(self.embedding_dim * 2, self.embedding_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.embedding_dim, self.n_modalities),
            nn.Softmax(dim=-1)
        )

    def build_fusion_layer(self):
        return nn.Sequential(
            nn.Linear(self.feat_embed_dim * 2, self.feat_embed_dim),
            nn.LayerNorm(self.feat_embed_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.feat_embed_dim, self.feat_embed_dim)
        )

    def process_modality(self, modal_name, indices):
        batch_size = indices.size(0)
        if modal_name == 'image':
            features = self.image_fcn(self.image_embedding(indices))
        else:
            features = self.text_fcn(self.text_embedding(indices))
        return features

    def progressive_fusion(self, user_emb, item_emb, item_indices):
        # Ensure all inputs have the same batch size
        batch_size = user_emb.size(0)
        modal_importance = self.modality_router(torch.cat([user_emb, item_emb], dim=-1))
        modal_order = torch.argsort(modal_importance, dim=-1, descending=True)
        
        # Initialize with item embeddings
        fused_features = item_emb
        
        for i in range(self.n_modalities):
            modal_idx = modal_order[:, i]
            modal_name = self.available_modalities[modal_idx[0].item()]
            
            # Process current modality
            modal_features = self.process_modality(modal_name, item_indices)
            
            # Ensure consistent dimensions for concatenation
            fused_features = self.fusion_networks[i](
                torch.cat([fused_features, modal_features], dim=-1)
            )
            
        return fused_features

    def forward(self, user, item):
        user_emb = self.user_embedding(user)
        item_emb = self.item_id_embedding(item)
        fused_item_features = self.progressive_fusion(user_emb, item_emb, item)
        return user_emb, fused_item_features

    def calculate_loss(self, interaction):
        user = interaction[0]
        pos_item = interaction[1]
        neg_item = interaction[2]

        user_emb, pos_item_emb = self.forward(user, pos_item)
        _, neg_item_emb = self.forward(user, neg_item)

        pos_scores = torch.mul(user_emb, pos_item_emb).sum(dim=1)
        neg_scores = torch.mul(user_emb, neg_item_emb).sum(dim=1)

        loss = -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()
        reg_loss = self.reg_weight * (
            user_emb.norm(2).pow(2) +
            pos_item_emb.norm(2).pow(2) +
            neg_item_emb.norm(2).pow(2)
        )

        return loss + reg_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_emb = self.user_embedding(user)
        
        # Process items in batches
        batch_size = 256  # Reduce batch size to manage memory
        scores = []
        
        for i in range(0, self.n_items, batch_size):
            end_idx = min(i + batch_size, self.n_items)
            batch_items = torch.arange(i, end_idx).to(self.device)
            
            # Get item embeddings for current batch
            batch_item_emb = self.item_id_embedding(batch_items)
            
            # Expand user embeddings to match batch size
            batch_user_emb = user_emb.unsqueeze(1).expand(-1, end_idx - i, -1)
            batch_item_emb = batch_item_emb.unsqueeze(0).expand(user_emb.size(0), -1, -1)
            
            # Process each user-item pair in the batch
            batch_scores = []
            for j in range(batch_user_emb.size(0)):
                user_j_emb = batch_user_emb[j]
                item_j_emb = batch_item_emb[j]
                fused_features = self.progressive_fusion(
                    user_j_emb,
                    item_j_emb,
                    batch_items
                )
                score = torch.mul(user_j_emb, fused_features).sum(dim=-1)
                batch_scores.append(score)
            
            batch_scores = torch.stack(batch_scores)
            scores.append(batch_scores)
        
        return torch.cat(scores, dim=1)