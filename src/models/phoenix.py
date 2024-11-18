import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SAGEConv
from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss

class PHOENIX(GeneralRecommender):
    def __init__(self, config, dataset):
        super(PHOENIX, self).__init__(config, dataset)

        # Configuration Parameters
        self.embedding_dim = config["embedding_size"]
        self.feat_embed_dim = config["feat_embed_dim"]
        self.n_layers = config["n_mm_layers"]
        self.reg_weight = config["reg_weight"]
        self.cl_weight = config["cl_weight"]
        self.dropout = config["dropout"]

        # Embeddings for Users and Items
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        # Modality-specific Layers
        if self.v_feat is not None:
            self.visual_transform = nn.Linear(self.v_feat.size(1), self.feat_embed_dim)
        if self.t_feat is not None:
            self.textual_transform = nn.Linear(self.t_feat.size(1), self.feat_embed_dim)

        # Multi-scale Graph Layers
        self.graph_layers = nn.ModuleList([
            GATConv(self.embedding_dim, self.embedding_dim, heads=4, concat=False),
            SAGEConv(self.embedding_dim, self.embedding_dim)
        ])

        # Attention Mechanisms
        self.user_item_attention = nn.MultiheadAttention(self.embedding_dim, num_heads=4)
        self.modality_attention = nn.Sequential(
            nn.Linear(2 * self.feat_embed_dim, self.feat_embed_dim),
            nn.ReLU(),
            nn.Linear(self.feat_embed_dim, 1),
            nn.Softmax(dim=1)
        )

        # Self-Supervised MLP
        self.mlp_predictor = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, self.embedding_dim)
        )

        # Prediction Layers
        self.predictor = nn.Linear(self.embedding_dim, 1)
        self.reg_loss = EmbLoss()

    def forward(self, interaction):
        user = interaction[0]
        item_pos = interaction[1]
        item_neg = interaction[2]

        # User and Item Embeddings
        user_emb = self.user_embedding[user]
        item_pos_emb = self.item_embedding[item_pos]
        item_neg_emb = self.item_embedding[item_neg]

        # Modality-Specific Embeddings
        visual_emb = F.relu(self.visual_transform(self.v_feat)) if self.v_feat is not None else 0
        textual_emb = F.relu(self.textual_transform(self.t_feat)) if self.t_feat is not None else 0

        # Modality Fusion
        modality_features = torch.cat([visual_emb, textual_emb], dim=1)
        modality_weights = self.modality_attention(modality_features)
        fused_features = (modality_weights[:, :1] * visual_emb) + (modality_weights[:, 1:] * textual_emb)

        # Graph Propagation
        h = fused_features
        for layer in self.graph_layers:
            h = F.dropout(layer(h, self.mm_adj), p=self.dropout, training=self.training)

        # Attention on User-Item Interactions
        user_item_rep, _ = self.user_item_attention(
            query=user_emb.unsqueeze(1),
            key=torch.cat([item_pos_emb.unsqueeze(1), item_neg_emb.unsqueeze(1)], dim=1),
            value=h.unsqueeze(1)
        )
        user_item_rep = user_item_rep.squeeze(1)

        # Final Scores
        pos_scores = self.predictor(user_item_rep + item_pos_emb).squeeze(-1)
        neg_scores = self.predictor(user_item_rep + item_neg_emb).squeeze(-1)

        return pos_scores, neg_scores

    def calculate_loss(self, interaction):
        pos_scores, neg_scores = self.forward(interaction)

        # Recommendation Loss
        rec_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))

        # Modality Alignment Loss
        align_loss = 0
        if self.v_feat is not None and self.t_feat is not None:
            align_loss = self.cl_weight * (
                1 - F.cosine_similarity(self.v_feat, self.t_feat, dim=-1).mean()
            )

        # Regularization Loss
        reg_loss = self.reg_weight * self.reg_loss(self.user_embedding.weight, self.item_embedding.weight)

        return rec_loss + align_loss + reg_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_emb = self.user_embedding[user]
        item_emb = self.item_embedding.weight

        # Graph Propagation for Items
        h = item_emb
        for layer in self.graph_layers:
            h = F.dropout(layer(h, self.mm_adj), p=self.dropout, training=self.training)

        scores = torch.matmul(user_emb, h.t())
        return scores
