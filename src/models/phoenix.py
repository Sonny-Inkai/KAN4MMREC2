import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, MessagePassing

from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss

class PHOENIX(GeneralRecommender):
    def __init__(self, config, dataset):
        super(PHOENIX, self).__init__(config, dataset)

        self.embedding_dim = config["embedding_size"]
        self.feat_embed_dim = config["feat_embed_dim"]
        self.n_layers = config["n_mm_layers"]
        self.reg_weight = config["reg_weight"]
        self.cl_weight = config["cl_weight"]
        self.dropout = config["dropout"]

        # Modality-specific embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        if self.v_feat is not None:
            self.visual_embedding = nn.Linear(self.v_feat.size(1), self.feat_embed_dim)
        if self.t_feat is not None:
            self.textual_embedding = nn.Linear(self.t_feat.size(1), self.feat_embed_dim)

        # Graph attention layers
        self.gat_visual = GATConv(self.feat_embed_dim, self.embedding_dim, heads=4)
        self.gat_textual = GATConv(self.feat_embed_dim, self.embedding_dim, heads=4)
        self.gat_cross = GATConv(self.embedding_dim, self.embedding_dim, heads=4)

        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(self.embedding_dim, num_heads=4)

        # Predictor and regularization
        self.predictor = nn.Linear(self.embedding_dim, 1)
        self.reg_loss = EmbLoss()

    def forward(self, interaction):
        user = interaction[0]
        item_pos = interaction[1]
        item_neg = interaction[2]

        # Modality-specific embeddings
        visual_emb = F.relu(self.visual_embedding(self.v_feat)) if self.v_feat is not None else 0
        textual_emb = F.relu(self.textual_embedding(self.t_feat)) if self.t_feat is not None else 0

        # GAT propagation
        visual_rep = self.gat_visual(visual_emb, self.mm_adj)
        textual_rep = self.gat_textual(textual_emb, self.mm_adj)

        # Cross-modal aggregation
        cross_rep, _ = self.cross_attention(visual_rep.unsqueeze(1), textual_rep.unsqueeze(1), textual_rep.unsqueeze(1))
        cross_rep = cross_rep.squeeze(1)

        # Combine user-item interactions
        user_emb = self.user_embedding[user]
        item_pos_emb = self.item_embedding[item_pos]
        item_neg_emb = self.item_embedding[item_neg]

        # Final embeddings
        user_item_pos = user_emb * item_pos_emb + cross_rep
        user_item_neg = user_emb * item_neg_emb + cross_rep

        pos_scores = self.predictor(user_item_pos).squeeze(-1)
        neg_scores = self.predictor(user_item_neg).squeeze(-1)

        return pos_scores, neg_scores

    def calculate_loss(self, interaction):
        pos_scores, neg_scores = self.forward(interaction)

        # Recommendation loss
        rec_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))

        # Modality alignment loss
        align_loss = self.cl_weight * (1 - F.cosine_similarity(self.v_feat, self.t_feat, dim=-1).mean())

        # Regularization
        reg_loss = self.reg_weight * self.reg_loss(self.user_embedding.weight, self.item_embedding.weight)

        return rec_loss + align_loss + reg_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_emb = self.user_embedding[user]
        item_emb = self.item_embedding.weight
        scores = torch.matmul(user_emb, item_emb.t())
        return scores
