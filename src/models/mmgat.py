import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import LightConv
from common.abstract_recommender import GeneralRecommender
import scipy.sparse as sp
import numpy as np

class MMGAT(GeneralRecommender):
    def __init__(self, config, dataset):
        super(KANMMRec, self).__init__(config, dataset)
        
        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.n_layers = config['n_layers']
        self.dropout = config['dropout']
        self.reg_weight = config['reg_weight']
        self.n_heads = 4
        self.temperature = 0.2
        
        self.n_users = dataset.user_num
        self.n_items = dataset.item_num
        
        # Get interaction matrix
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.norm_adj = self.get_norm_adj_mat().to(self.device)
        
        # ID Embeddings with position encoding
        self.user_embedding = PositionalEmbedding(self.n_users, self.embedding_dim)
        self.item_embedding = PositionalEmbedding(self.n_items, self.embedding_dim)
        
        # Modal-specific modules
        if self.v_feat is not None:
            self.image_encoder = ModalEncoder(
                self.v_feat.shape[1],
                self.feat_embed_dim,
                self.n_heads
            )
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            
        if self.t_feat is not None:
            self.text_encoder = ModalEncoder(
                self.t_feat.shape[1],
                self.feat_embed_dim,
                self.n_heads
            )
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
        
        # Cross-modal Interaction
        self.modal_interaction = CrossModalInteraction(
            self.feat_embed_dim,
            self.n_heads
        )
        
        # Collaborative knowledge distillation
        self.knowledge_distillation = CollaborativeDistillation(
            self.embedding_dim,
            self.feat_embed_dim
        )
        
        # Interest-aware aggregation
        self.interest_aggregation = InterestAwareAggregation(
            self.embedding_dim,
            self.feat_embed_dim
        )
        
        # Graph convolution layers
        self.light_gcn = nn.ModuleList([
            LightConv(self.embedding_dim, num_relations=2)
            for _ in range(self.n_layers)
        ])
        
        self.apply(self._init_weights)
        
    def get_norm_adj_mat(self):
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items))
        adj_mat = adj_mat.tolil()
        R = self.interaction_matrix.tolil()
        
        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()
        
        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        
        norm_adj = d_mat_inv.dot(adj_mat).dot(d_mat_inv)
        return self._sparse_mx_to_torch_sparse_tensor(norm_adj.tocoo())
    
    def _sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col))).long()
        values = torch.from_numpy(sparse_mx.data).float()
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)
        
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
            
    def forward(self, users, pos_items=None, neg_items=None):
        # Process modal features
        modal_embeds = []
        
        if self.v_feat is not None:
            visual_embed = self.image_encoder(self.image_embedding.weight)
            modal_embeds.append(visual_embed)
            
        if self.t_feat is not None:
            text_embed = self.text_encoder(self.text_embedding.weight)
            modal_embeds.append(text_embed)
        
        # Cross-modal interaction
        if len(modal_embeds) > 1:
            modal_embed = self.modal_interaction(modal_embeds)
        else:
            modal_embed = modal_embeds[0]
        
        # Get base embeddings
        user_embed = self.user_embedding.weight
        item_embed = self.item_embedding.weight
        
        # Collaborative knowledge distillation
        distilled_embed = self.knowledge_distillation(item_embed, modal_embed)
        
        # Graph convolution
        all_embed = torch.cat([user_embed, distilled_embed], dim=0)
        embeds_list = [all_embed]
        
        for layer in self.light_gcn:
            all_embed = layer(all_embed, self.norm_adj)
            embeds_list.append(all_embed)
            
        all_embed = torch.stack(embeds_list, dim=1)
        all_embed = self.interest_aggregation(all_embed)
        
        user_embed, item_embed = torch.split(all_embed, [self.n_users, self.n_items])
        
        if pos_items is not None:
            user_embed = user_embed[users]
            pos_embed = item_embed[pos_items]
            neg_embed = item_embed[neg_items]
            return user_embed, pos_embed, neg_embed
            
        return user_embed[users], item_embed

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        
        user_embed, pos_embed, neg_embed = self.forward(users, pos_items, neg_items)
        
        # Main recommendation loss
        pos_scores = (user_embed * pos_embed).sum(dim=1)
        neg_scores = (user_embed * neg_embed).sum(dim=1)
        rec_loss = F.softplus(neg_scores - pos_scores).mean()
        
        # Contrastive loss
        pos_dist = F.pairwise_distance(user_embed, pos_embed)
        neg_dist = F.pairwise_distance(user_embed, neg_embed)
        contrast_loss = F.relu(pos_dist - neg_dist + self.temperature).mean()
        
        # Regularization
        reg_loss = self.reg_weight * (
            user_embed.norm(2).pow(2) +
            pos_embed.norm(2).pow(2) +
            neg_embed.norm(2).pow(2)
        ) / len(users)
        
        return rec_loss + 0.1 * contrast_loss + reg_loss

    def full_sort_predict(self, interaction):
        users = interaction[0]
        user_embed, item_embed = self.forward(users)
        scores = torch.matmul(user_embed, item_embed.t())
        return scores

class PositionalEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        pe = torch.zeros(num_embeddings, embedding_dim)
        position = torch.arange(0, num_embeddings).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2) * -(math.log(10000.0) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    @property
    def weight(self):
        return self.embedding.weight + self.pe

class ModalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_heads):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
    def forward(self, x):
        x = self.mlp(x)
        x = x.unsqueeze(0)
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        return self.norm2(x).squeeze(0)

class CrossModalInteraction(nn.Module):
    def __init__(self, hidden_dim, n_heads):
        super().__init__()
        
        self.cross_attention = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
    def forward(self, modalities):
        x = torch.stack(modalities, dim=1)
        attn_out, _ = self.cross_attention(x, x, x)
        
        gate = self.gate(torch.cat([x, attn_out], dim=-1))
        x = gate * attn_out + (1 - gate) * x
        
        return self.norm(x.mean(dim=1))

class CollaborativeDistillation(nn.Module):
    def __init__(self, id_dim, feat_dim):
        super().__init__()
        
        self.transfer = nn.Sequential(
            nn.Linear(feat_dim, id_dim),
            nn.LayerNorm(id_dim),
            nn.GELU()
        )
        
        self.gate = nn.Sequential(
            nn.Linear(id_dim * 2, id_dim),
            nn.Sigmoid()
        )
        
    def forward(self, id_embeds, feat_embeds):
        feat_embeds = self.transfer(feat_embeds)
        gate = self.gate(torch.cat([id_embeds, feat_embeds], dim=-1))
        return gate * id_embeds + (1 - gate) * feat_embeds

class InterestAwareAggregation(nn.Module):
    def __init__(self, id_dim, feat_dim):
        super().__init__()
        
        self.attention = nn.Sequential(
            nn.Linear(id_dim, id_dim),
            nn.Tanh(),
            nn.Linear(id_dim, 1, bias=False)
        )
        
    def forward(self, embeddings):
        attention = self.attention(embeddings)
        attention = F.softmax(attention, dim=1)
        return (attention * embeddings).sum(dim=1)