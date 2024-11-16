import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from common.abstract_recommender import GeneralRecommender
import numpy as np

class HOPE(GeneralRecommender):
    def __init__(self, config, dataset):
        super(HOPE, self).__init__(config, dataset)

        # Config parameters
        self.embedding_dim = config["embedding_size"]
        self.feat_embed_dim = config["embedding_size"]
        self.n_layers = config["n_mm_layers"]
        self.reg_weight = config["reg_weight"]
        self.dropout = config["dropout"]

        # Embedding Layers
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        # Modality-specific embeddings (Images, Text)
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_transform = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_transform = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)

        # Enhanced GNN layers for Graph Attention Networks
        self.gnn_image = GATConv(self.feat_embed_dim, self.embedding_dim, heads=4, concat=True, dropout=self.dropout)
        self.gnn_text = GATConv(self.feat_embed_dim, self.embedding_dim, heads=4, concat=True, dropout=self.dropout)

        # Cross-modal contrastive fusion layers
        self.cross_modal_fc = nn.Linear(4 * self.embedding_dim, self.embedding_dim)

        # Counterfactual reasoning for bias mitigation
        self.counterfactual_weight = nn.Parameter(torch.FloatTensor([0.5]))

        # Load edge_index from the dataset
        train_interactions = dataset.inter_matrix(form="coo").astype(np.float32)
        edge_index = self.pack_edge_index(train_interactions)

        # Ensuring edge indices are within range
        self.edge_index = self.validate_edge_index(edge_index, self.n_users + self.n_items)

    def pack_edge_index(self, inter_mat):
        rows = inter_mat.row
        cols = inter_mat.col + self.n_users  # Offset item nodes by the number of user nodes

        # Ensuring indices are valid and within range
        valid_mask = (rows >= 0) & (rows < self.n_users) & (cols >= self.n_users) & (cols < self.n_users + self.n_items)
        rows = rows[valid_mask]
        cols = cols[valid_mask]

        return np.column_stack((rows, cols))

    def validate_edge_index(self, edge_index, num_nodes):
        """Ensure all edge indices are within valid node range."""
        valid_mask = (edge_index[:, 0] < num_nodes) & (edge_index[:, 1] < num_nodes)
        edge_index = edge_index[valid_mask]
        
        edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(self.device)
        return edge_index_tensor

    def forward(self, modality='image'):
        if modality == 'image':
            x = F.relu(self.image_transform(self.image_embedding.weight))
            x = F.dropout(x, p=self.dropout, training=self.training)

            # Ensure edge_index is within range of x
            num_nodes = x.shape[0]
            edge_index = self.edge_index[:, (self.edge_index[0] < num_nodes) & (self.edge_index[1] < num_nodes)]

            x = self.gnn_image(x, edge_index)
        elif modality == 'text':
            x = F.relu(self.text_transform(self.text_embedding.weight))
            x = F.dropout(x, p=self.dropout, training=self.training)

            # Ensure edge_index is within range of x
            num_nodes = x.shape[0]
            edge_index = self.edge_index[:, (self.edge_index[0] < num_nodes) & (self.edge_index[1] < num_nodes)]

            x = self.gnn_text(x, edge_index)
        else:
            raise ValueError("Modality must be 'image' or 'text'")
        return x

    def cross_modal_fusion(self, image_features, text_features):
        # Adjust the combined_features shape to match the input requirements of cross_modal_fc
        if image_features.shape[0] != text_features.shape[0]:
            min_size = min(image_features.shape[0], text_features.shape[0])
            image_features = image_features[:min_size]
            text_features = text_features[:min_size]
        
        combined_features = torch.cat((image_features, text_features), dim=-1)
        combined_features = F.relu(self.cross_modal_fc(combined_features))
        return combined_features

    def counterfactual_reasoning(self, fused_features):
        counterfactual_features = fused_features * self.counterfactual_weight + (1 - self.counterfactual_weight) * fused_features.detach()
        return counterfactual_features

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        user_emb = self.user_embedding.weight[users]
        image_features = self.forward(modality='image')
        text_features = self.forward(modality='text')
        fused_features = self.cross_modal_fusion(image_features, text_features)
        final_representation = self.counterfactual_reasoning(fused_features)

        pos_item_emb = self.item_embedding(pos_items)
        neg_item_emb = self.item_embedding(neg_items)

        bpr_loss = self.bpr_loss(user_emb, pos_item_emb, neg_item_emb)
        reg_loss = self.reg_weight * (user_emb.norm(2).pow(2) + pos_item_emb.norm(2).pow(2) + neg_item_emb.norm(2).pow(2))
        return bpr_loss + reg_loss

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        return mf_loss

    def full_sort_predict(self, interaction):
        users = interaction[0]
        user_emb = self.user_embedding(users)
        image_features = self.forward(modality='image')
        text_features = self.forward(modality='text')
        fused_features = self.cross_modal_fusion(image_features, text_features)
        final_representation = self.counterfactual_reasoning(fused_features)

        scores = torch.matmul(user_emb, self.item_embedding.weight.transpose(0, 1))
        return scores
