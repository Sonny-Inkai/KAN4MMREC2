# phoenix.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from common.abstract_recommender import GeneralRecommender

class PHOENIX(GeneralRecommender):
    def __init__(self, config, dataset):
        super(PHOENIX, self).__init__(config, dataset)
        
        # Model Configurations
        self.embedding_dim = config.get("embedding_size", 64)
        self.n_layers = config.get("n_gnn_layers", 3)
        self.knn_k = config.get("knn_k", 20)
        self.reg_weight = config.get("reg_weight", 1e-5)
        self.cl_weight = config.get("cl_weight", 0.1)
        self.dropout = config.get("dropout", 0.1)
        self.device = config.get("device", "cpu")
        
        # Embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        # Modality-Specific Transformation Layers
        if self.v_feat is not None:
            self.image_transform = nn.Linear(self.v_feat.shape[1], self.embedding_dim)
        if self.t_feat is not None:
            self.text_transform = nn.Linear(self.t_feat.shape[1], self.embedding_dim)
        
        # Multi-Head Graph Attention
        self.gat_layers = nn.ModuleList(
            [GATConv(self.embedding_dim, self.embedding_dim, heads=4, concat=False) for _ in range(self.n_layers)]
        )

        # Modality Attention
        self.modality_attention = nn.Linear(3 * self.embedding_dim, self.embedding_dim)

        # Feature Gating Layer
        self.feature_gate = nn.Linear(3 * self.embedding_dim, 1)

        # Contrastive Projection Layers
        self.user_projection = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.item_projection = nn.Linear(self.embedding_dim, self.embedding_dim)

        # Fusion Layer
        self.fusion_layer = nn.Linear(3 * self.embedding_dim, self.embedding_dim)

        # Loss Functions
        self.bpr_loss = nn.BCEWithLogitsLoss()
        self.cl_loss = nn.CosineEmbeddingLoss()

    def forward(self):
        # User and Item Embeddings
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight

        # Visual and Textual Feature Embeddings
        v_embeddings = F.normalize(self.image_transform(self.v_feat), dim=1) if self.v_feat is not None else torch.zeros_like(item_embeddings).to(self.device)
        t_embeddings = F.normalize(self.text_transform(self.t_feat), dim=1) if self.t_feat is not None else torch.zeros_like(item_embeddings).to(self.device)

        # Graph Attention Layers
        all_embeddings = torch.cat((user_embeddings, item_embeddings), dim=0)
        for gat_layer in self.gat_layers:
            all_embeddings = F.dropout(all_embeddings, p=self.dropout, training=self.training)
            all_embeddings = gat_layer(all_embeddings, self.edge_index)

        user_embeddings, item_embeddings = torch.split(
            all_embeddings, [self.n_users, self.n_items], dim=0
        )

        # Modality Fusion with Attention
        combined_item_embeddings = torch.cat(
            [item_embeddings, v_embeddings, t_embeddings], dim=1
        )
        fused_item_embeddings = F.relu(self.modality_attention(combined_item_embeddings))

        # Feature Gating
        modality_scores = torch.sigmoid(self.feature_gate(combined_item_embeddings))
        fused_item_embeddings = fused_item_embeddings * modality_scores

        # Final User and Item Representations
        return user_embeddings, fused_item_embeddings

    def calculate_loss(self, interaction):
        user_indices = interaction[0]
        pos_item_indices = interaction[1]
        neg_item_indices = interaction[2]

        user_embeddings, item_embeddings = self.forward()
        pos_scores = torch.mul(user_embeddings[user_indices], item_embeddings[pos_item_indices]).sum(dim=1)
        neg_scores = torch.mul(user_embeddings[user_indices], item_embeddings[neg_item_indices]).sum(dim=1)

        # BPR Loss
        bpr_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))

        # Contrastive Loss (Optional)
        if self.cl_weight > 0:
            pos_cl_loss = self.contrastive_loss(user_embeddings[user_indices], item_embeddings[pos_item_indices])
            neg_cl_loss = self.contrastive_loss(user_embeddings[user_indices], item_embeddings[neg_item_indices])
            cl_loss = (pos_cl_loss + neg_cl_loss) / 2
        else:
            cl_loss = 0.0

        return bpr_loss + self.cl_weight * cl_loss

    def contrastive_loss(self, user_embeddings, item_embeddings):
        user_proj = F.normalize(self.user_projection(user_embeddings), dim=1)
        item_proj = F.normalize(self.item_projection(item_embeddings), dim=1)
        return 1 - F.cosine_similarity(user_proj, item_proj).mean()

    def full_sort_predict(self, interaction):
        user_indices = interaction[0]
        user_embeddings, item_embeddings = self.forward()
        scores = torch.matmul(user_embeddings[user_indices], item_embeddings.T)
        return scores
