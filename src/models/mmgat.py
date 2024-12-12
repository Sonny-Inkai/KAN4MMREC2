import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from common.abstract_recommender import GeneralRecommender

class MMGAT(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MMGAT, self).__init__(config, dataset)
        
        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.dropout = config['dropout']
        self.reg_weight = config['reg_weight']
        self.temperature = config['temperature']

        self.n_nodes = self.n_users + self.n_items
        
        # Base embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        
        # Modal-specific processors
        if self.v_feat is not None:
            self.image_encoder = ModalityEncoder(
                self.v_feat.shape[1], 
                self.feat_embed_dim,
                self.n_heads,
                self.dropout
            )
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            
        if self.t_feat is not None:
            self.text_encoder = ModalityEncoder(
                self.t_feat.shape[1],
                self.feat_embed_dim, 
                self.n_heads,
                self.dropout
            )
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
        
        # Cross-modal fusion
        self.modal_fusion = CrossModalFusion(
            self.feat_embed_dim,
            self.n_heads,
            self.dropout
        )
        
        # Graph structure learning
        self.graph_learner = AdaptiveGraphLearner(
            self.embedding_dim,
            self.temperature
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            nn.init.xavier_normal_(module.weight)
            
    def forward(self, user_nodes, pos_item_nodes=None, neg_item_nodes=None):
        # Process modalities
        item_feats = []
        
        if self.v_feat is not None:
            image_feats = self.image_encoder(self.image_embedding.weight)
            item_feats.append(image_feats)
            
        if self.t_feat is not None:
            text_feats = self.text_encoder(self.text_embedding.weight)
            item_feats.append(text_feats)
        
        # Fuse modalities
        if len(item_feats) > 1:
            item_feats = self.modal_fusion(item_feats)
        else:
            item_feats = item_feats[0]
            
        # Learn graph structure
        base_embeddings = torch.cat([
            self.user_embedding.weight,
            self.item_embedding.weight
        ], dim=0)
        
        adj_matrix = self.graph_learner(base_embeddings)
        
        # Message passing
        all_embeddings = base_embeddings
        embeddings_list = [all_embeddings]
        
        for _ in range(self.n_layers):
            all_embeddings = torch.sparse.mm(adj_matrix, all_embeddings)
            all_embeddings = F.normalize(all_embeddings, p=2, dim=1)
            embeddings_list.append(all_embeddings)
            
        all_embeddings = torch.mean(torch.stack(embeddings_list, dim=1), dim=1)
        
        user_embeddings = all_embeddings[:self.n_users]
        item_embeddings = all_embeddings[self.n_users:] + item_feats
        
        users = user_embeddings[user_nodes]
        
        if pos_item_nodes is not None:
            pos_items = item_embeddings[pos_item_nodes]
            neg_items = item_embeddings[neg_item_nodes]
            return users, pos_items, neg_items
            
        return users, item_embeddings
        
    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1] 
        neg_items = interaction[2]
        
        user_e, pos_e, neg_e = self.forward(users, pos_items, neg_items)
        
        # BPR Loss
        pos_scores = torch.sum(user_e * pos_e, dim=1)
        neg_scores = torch.sum(user_e * neg_e, dim=1)
        
        bpr_loss = torch.mean(-F.logsigmoid(pos_scores - neg_scores))
        
        # Regularization Loss
        reg_loss = self.reg_weight * (
            user_e.norm(2).pow(2) + 
            pos_e.norm(2).pow(2) + 
            neg_e.norm(2).pow(2)
        ) / float(len(users))
        
        return bpr_loss + reg_loss

    def full_sort_predict(self, interaction):
        users = interaction[0]
        
        user_e, item_e = self.forward(users)
        scores = torch.matmul(user_e, item_e.transpose(0, 1))
        
        return scores

class ModalityEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_heads, dropout):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            hidden_dim, 
            n_heads,
            dropout=dropout
        )
        
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        x = self.fc(x)
        x = x.unsqueeze(0)
        x, _ = self.attention(x, x, x)
        return x.squeeze(0)

class CrossModalFusion(nn.Module):
    def __init__(self, hidden_dim, n_heads, dropout):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            n_heads, 
            dropout=dropout
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, modalities):
        x = torch.stack(modalities, dim=0)
        x, _ = self.attention(x, x, x)
        x = x.mean(dim=0)
        x = self.fc(x)
        return x

class AdaptiveGraphLearner(nn.Module):
    def __init__(self, hidden_dim, temperature):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, x):
        sim = torch.mm(x, x.transpose(0, 1))
        sim = F.softmax(sim / self.temperature, dim=1)
        return sim