import torch
import torch.nn as nn
import torch.nn.functional as F
from common.abstract_recommender import GeneralRecommender

class LIGHTMMREC(GeneralRecommender):
    def __init__(self, config, dataset):
        super(LIGHTMMREC, self).__init__(config, dataset)
        
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
        self.cache_hits = {m: 0 for m in self.available_modalities}
        self.cache_misses = {m: 0 for m in self.available_modalities}

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

    def get_cached_features(self, modal_name, indices):
        if modal_name not in self.feature_cache:
            self.feature_cache[modal_name] = {}
            
        cache = self.feature_cache[modal_name]
        features = []
        uncached_indices = []
        
        for idx in indices:
            idx = idx.item()
            if idx in cache:
                features.append(cache[idx])
                self.cache_hits[modal_name] += 1
            else:
                uncached_indices.append(idx)
                self.cache_misses[modal_name] += 1
                
        return features, uncached_indices

    def update_cache(self, modal_name, indices, features):
        cache = self.feature_cache[modal_name]
        for idx, feat in zip(indices, features):
            if len(cache) >= self.cache_size:
                cache.pop(next(iter(cache)))
            cache[idx.item()] = feat.detach()

    def process_modality(self, modal_name, indices):
        cached_features, uncached_indices = self.get_cached_features(modal_name, indices)
        
        if uncached_indices:
            if modal_name == 'image':
                new_features = self.image_fcn(self.image_embedding(torch.tensor(uncached_indices).to(self.device)))
            else:
                new_features = self.text_fcn(self.text_embedding(torch.tensor(uncached_indices).to(self.device)))
            
            self.update_cache(modal_name, torch.tensor(uncached_indices), new_features)
            cached_features.extend([f.detach() for f in new_features])
            
        return torch.stack(cached_features)

    def progressive_fusion(self, user_emb, item_emb, item_indices):
        modal_importance = self.modality_router(torch.cat([user_emb, item_emb], dim=-1))
        modal_order = torch.argsort(modal_importance, dim=-1, descending=True)
        
        fused_features = item_emb
        confidence = torch.zeros(item_emb.size(0)).to(self.device)
        
        for i in range(self.n_modalities):
            modal_idx = modal_order[:, i]
            modal_name = self.available_modalities[modal_idx[0]]
            
            modal_features = self.process_modality(modal_name, item_indices)
            fused_features = self.fusion_networks[i](torch.cat([fused_features, modal_features], dim=-1))
            
            current_confidence = F.cosine_similarity(fused_features, item_emb)
            confidence = torch.maximum(confidence, current_confidence)
            
            if torch.all(confidence > 0.9):
                break
                
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

        bpr_loss = -(pos_scores - neg_scores).sigmoid().log().mean()
        reg_loss = self.reg_weight * (user_emb.norm(2).pow(2) + 
                                    pos_item_emb.norm(2).pow(2) + 
                                    neg_item_emb.norm(2).pow(2))

        return bpr_loss + reg_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_emb = self.user_embedding(user)
        
        all_items = torch.arange(self.n_items).to(self.device)
        item_emb = self.item_id_embedding(all_items)
        fused_item_features = self.progressive_fusion(user_emb.unsqueeze(1).expand(-1, self.n_items, -1).reshape(-1, self.embedding_dim),
                                                    item_emb.unsqueeze(0).expand(user_emb.size(0), -1, -1).reshape(-1, self.embedding_dim),
                                                    all_items)
        
        score = torch.mul(user_emb.unsqueeze(1), fused_item_features.view(user_emb.size(0), -1, self.embedding_dim)).sum(dim=-1)
        return score