import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GraphConv
from torch_geometric.utils import add_self_loops, remove_self_loops, degree
from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss

class KAN4MMREC(GeneralRecommender):
    def __init__(self, config, dataset):
        super(KAN4MMREC, self).__init__(config, dataset)

        # Config parameters
        self.embedding_dim = config["embedding_size"]
        self.feat_embed_dim = config["feat_embed_dim"]
        self.knn_k = config["knn_k"]
        self.lambda_coeff = config["lambda_coeff"]
        self.n_layers = config["n_mm_layers"]
        self.reg_weight = config["reg_weight"]
        self.dropout = config["dropout"]
        self.device = config["device"]
        
        # Defining Embedding Layers
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim*2)
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
        
        # GNN layers for modality-specific graphs
        self.gnn_image = GATConv(self.feat_embed_dim, self.embedding_dim, heads=2, concat=True)
        self.gnn_text = GATConv(self.feat_embed_dim, self.embedding_dim, heads=2, concat=True)
        
        # Cross-modal contrastive fusion layers
        self.cross_modal_fc = nn.Linear(2 * self.embedding_dim, self.embedding_dim)
        
        # Counterfactual reasoning for bias mitigation
        self.counterfactual_weight = nn.Parameter(torch.FloatTensor([0.5]))

    def forward(self, edge_index, modality='image'):
        if modality == 'image':
            x = F.relu(self.image_transform(self.image_embedding.weight))
            x = self.gnn_image(x, edge_index)
        elif modality == 'text':
            x = F.relu(self.text_transform(self.text_embedding.weight))
            x = self.gnn_text(x, edge_index)
        else:
            raise ValueError("Modality must be 'image' or 'text'")
        return x

    def cross_modal_fusion(self, image_features, text_features):
        # Perform contrastive learning to align the features of different modalities
        combined_features = torch.cat((image_features, text_features), dim=-1)
        combined_features = F.relu(self.cross_modal_fc(combined_features))
        return combined_features

    def counterfactual_reasoning(self, fused_features):
        # Apply counterfactual reasoning to mitigate bias from any particular modality
        counterfactual_features = fused_features * self.counterfactual_weight + (1 - self.counterfactual_weight) * fused_features.detach()
        return counterfactual_features

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        
        user_emb = self.user_embedding[users]
        image_features = self.forward(edge_index=self.edge_index, modality='image')
        text_features = self.forward(edge_index=self.edge_index, modality='text')
        fused_features = self.cross_modal_fusion(image_features, text_features)
        final_representation = self.counterfactual_reasoning(fused_features)
        
        pos_item_emb = final_representation[pos_items]
        neg_item_emb = final_representation[neg_items]
        
        bpr_loss = self.bpr_loss(user_emb, pos_item_emb, neg_item_emb)
        reg_loss = self.reg_weight * (user_emb.norm(2).pow(2) + pos_item_emb.norm(2).pow(2) + neg_item_emb.norm(2).pow(2))
        return bpr_loss + reg_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_embedding = self.user_embedding[user]
        image_features = self.forward(edge_index=self.edge_index, modality='image')
        text_features = self.forward(edge_index=self.edge_index, modality='text')
        fused_features = self.cross_modal_fusion(image_features, text_features)
        final_representation = self.counterfactual_reasoning(fused_features)

        scores = torch.matmul(user_embedding, final_representation.transpose(0,1))
        return scores

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        return mf_loss
