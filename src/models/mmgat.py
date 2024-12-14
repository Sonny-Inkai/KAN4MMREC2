import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from common.abstract_recommender import GeneralRecommender
from common.loss import EmbLoss

class MMGAT(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MMGAT, self).__init__(config, dataset)
        
        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.n_layers = config['n_layers']
        self.dropout = config['dropout']
        self.reg_weight = config['reg_weight']
        
        # Core embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)

        # Feature processors
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_projector = nn.Sequential(
                nn.Linear(self.v_feat.shape[1], self.feat_embed_dim // 2),
                nn.Dropout(self.dropout),
                nn.ReLU()
            )
            
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_projector = nn.Sequential(
                nn.Linear(self.t_feat.shape[1], self.feat_embed_dim // 2),
                nn.Dropout(self.dropout),
                nn.ReLU()
            )

        # Progressive refinement layers
        self.refinement_layers = nn.ModuleList([
            RefinementLayer(self.embedding_dim, self.dropout)
            for _ in range(self.n_layers)
        ])

        # Final prediction layers
        self.predictor = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(self.embedding_dim // 2, self.embedding_dim)
        )

        self.criterion = nn.BCEWithLogitsLoss()
        self.reg_loss = EmbLoss()

    def get_sparse_adj_mat(self):
        # Convert to indices format for efficiency
        row = self.interaction_matrix.row
        col = self.interaction_matrix.col + self.n_users
        
        indices = np.vstack((row, col))
        indices = np.hstack((indices, np.vstack((col, row))))  # Add reverse edges
        
        values = np.ones(indices.shape[1])
        shape = (self.n_users + self.n_items, self.n_users + self.n_items)
        
        # Create sparse tensor directly
        indices = torch.LongTensor(indices)
        values = torch.FloatTensor(values)
        
        return torch.sparse_coo_tensor(indices, values, shape).to(self.device)

    def process_modal_features(self, batch_size=None):
        modal_features = []
        
        if self.v_feat is not None:
            img_feats = self.image_projector(self.image_embedding.weight)
            modal_features.append(img_feats)
            
        if self.t_feat is not None:
            txt_feats = self.text_projector(self.text_embedding.weight)
            modal_features.append(txt_feats)

        if modal_features:
            combined = torch.cat(modal_features, dim=-1)
            return F.normalize(combined, p=2, dim=-1)
        return None

    def forward(self, users, items=None):
        # Process modalities first
        modal_feats = self.process_modal_features()
        
        # Get base embeddings
        user_embeddings = self.user_embedding(users)
        if items is not None:
            item_embeddings = self.item_embedding(items)
        else:
            item_embeddings = self.item_embedding.weight

        # Progressive refinement
        for layer in self.refinement_layers:
            user_embeddings = layer(user_embeddings)
            item_embeddings = layer(item_embeddings)

        # Integrate modal features with item embeddings
        if modal_feats is not None:
            if items is not None:
                modal_feats = modal_feats[items]
            item_embeddings = item_embeddings + modal_feats

        # Final predictions
        user_embeddings = self.predictor(user_embeddings)
        item_embeddings = self.predictor(item_embeddings)
        
        return user_embeddings, item_embeddings

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        # Get embeddings
        user_emb, all_item_emb = self.forward(users)
        pos_emb = all_item_emb[pos_items]
        neg_emb = all_item_emb[neg_items]

        # Compute scores
        pos_scores = (user_emb * pos_emb).sum(dim=1)
        neg_scores = (user_emb * neg_emb).sum(dim=1)
        
        # BPR loss
        loss = -(pos_scores - neg_scores).sigmoid().log().mean()

        # Add regularization
        reg_loss = self.reg_weight * self.reg_loss(user_emb, pos_emb, neg_emb)
        
        return loss + reg_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_emb, item_emb = self.forward(user)
        scores = torch.matmul(user_emb, item_emb.t())
        return scores

class RefinementLayer(nn.Module):
    def __init__(self, embedding_dim, dropout):
        super().__init__()
        self.refiner = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        self.norm = nn.LayerNorm(embedding_dim)
        
    def forward(self, x):
        residual = x
        out = self.refiner(x)
        out = self.norm(out + residual)
        return out