import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SAGEConv, TransformerConv
from torch_sparse import SparseTensor
from common.abstract_recommender import GeneralRecommender
import math

class MMGAT(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MMGAT, self).__init__(config, dataset)
        
        # Hyperparameters
        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.num_heads = 8
        self.num_layers = 3
        self.dropout = config['dropout']
        self.reg_weight = config['reg_weight']
        self.ssl_temp = 0.2
        self.ssl_reg = 0.1
        self.hard_neg_weight = 0.5
        
        # Base embeddings with positional encoding
        self.user_embedding = PositionalEmbedding(self.n_users, self.embedding_dim)
        self.item_embedding = PositionalEmbedding(self.n_items, self.embedding_dim)
        
        # Modality-specific processors
        if self.v_feat is not None:
            self.visual_processor = ModalityProcessor(
                input_dim=self.v_feat.shape[1],
                hidden_dim=self.feat_embed_dim,
                num_heads=self.num_heads,
                num_layers=self.num_layers,
                dropout=self.dropout
            )
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            
        if self.t_feat is not None:
            self.text_processor = ModalityProcessor(
                input_dim=self.t_feat.shape[1],
                hidden_dim=self.feat_embed_dim,
                num_heads=self.num_heads,
                num_layers=self.num_layers,
                dropout=self.dropout
            )
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
        
        # Multi-modal fusion with gating
        self.modal_fusion = MultiModalFusionTransformer(
            dim=self.feat_embed_dim,
            num_heads=self.num_heads,
            dropout=self.dropout
        )
        
        # Graph neural networks
        self.gnn_layers = nn.ModuleList([
            MultiModalGraphBlock(
                dim=self.embedding_dim,
                num_heads=self.num_heads
            ) for _ in range(self.num_layers)
        ])
        
        # Contrastive learning projector
        self.projector = MLPProjector(self.embedding_dim)
        
        # Initialize attention scores
        self.att_scores = None
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    
    def forward(self, users, pos_items=None, neg_items=None):
        # Process modalities with attention
        modal_features = []
        att_weights = []
        
        if self.v_feat is not None:
            visual_feats, v_att = self.visual_processor(self.image_embedding.weight)
            modal_features.append(visual_feats)
            att_weights.append(v_att)
            
        if self.t_feat is not None:
            text_feats, t_att = self.text_processor(self.text_embedding.weight)
            modal_features.append(text_feats)
            att_weights.append(t_att)
        
        # Fuse modalities with dynamic weighting
        if len(modal_features) > 1:
            fused_features = self.modal_fusion(modal_features, att_weights)
        else:
            fused_features = modal_features[0]
            
        # Get base embeddings with positional encoding
        user_emb = self.user_embedding(torch.arange(self.n_users).to(self.device))
        item_emb = self.item_embedding(torch.arange(self.n_items).to(self.device))
        
        # Combine with modal features
        item_emb = item_emb + fused_features
        
        # Graph message passing with residual connections
        all_embeddings = torch.cat([user_emb, item_emb], dim=0)
        embeddings_list = [all_embeddings]
        
        for gnn in self.gnn_layers:
            curr_emb = gnn(all_embeddings, self.edge_index)
            curr_emb = F.dropout(curr_emb, self.dropout, training=self.training)
            all_embeddings = curr_emb + all_embeddings
            embeddings_list.append(all_embeddings)
        
        # Multi-scale fusion
        all_embeddings = torch.stack(embeddings_list, dim=1)
        all_embeddings = self.attention_fusion(all_embeddings)
        
        user_embeddings = all_embeddings[:self.n_users]
        item_embeddings = all_embeddings[self.n_users:]
        
        users_emb = user_embeddings[users]
        
        if pos_items is not None and neg_items is not None:
            pos_emb = item_embeddings[pos_items]
            neg_emb = item_embeddings[neg_items]
            return users_emb, pos_emb, neg_emb
            
        return users_emb, item_embeddings

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        
        user_emb, pos_emb, neg_emb = self.forward(users, pos_items, neg_items)
        
        # BPR loss
        pos_scores = (user_emb * pos_emb).sum(dim=1)
        neg_scores = (user_emb * neg_emb).sum(dim=1)
        
        bpr_loss = F.softplus(neg_scores - pos_scores).mean()
        
        # Contrastive learning loss
        user_proj = self.projector(user_emb)
        pos_proj = self.projector(pos_emb)
        neg_proj = self.projector(neg_emb)
        
        ssl_loss = self.ssl_loss(user_proj, pos_proj, neg_proj)
        
        # Hard negative mining
        with torch.no_grad():
            hard_neg_idx = self.mine_hard_negatives(user_emb, neg_emb)
            
        hard_neg_emb = neg_emb[hard_neg_idx]
        hard_neg_loss = F.softplus((user_emb * hard_neg_emb).sum(dim=1)).mean()
        
        # L2 regularization
        reg_loss = self.reg_weight * (
            user_emb.norm(2).pow(2) +
            pos_emb.norm(2).pow(2) +
            neg_emb.norm(2).pow(2)
        ) / float(len(users))
        
        return bpr_loss + self.ssl_reg * ssl_loss + self.hard_neg_weight * hard_neg_loss + reg_loss

    def ssl_loss(self, anchor, positive, negative):
        pos_sim = F.cosine_similarity(anchor, positive)
        neg_sim = F.cosine_similarity(anchor, negative)
        
        pos_exp = torch.exp(pos_sim / self.ssl_temp)
        neg_exp = torch.exp(neg_sim / self.ssl_temp)
        
        ssl_loss = -torch.log(pos_exp / (pos_exp + neg_exp))
        return ssl_loss.mean()

    def mine_hard_negatives(self, user_emb, neg_emb):
        with torch.no_grad():
            neg_scores = torch.matmul(user_emb, neg_emb.t())
            hard_neg_idx = neg_scores.topk(k=5, dim=1).indices[:, 0]
        return hard_neg_idx

    def attention_fusion(self, embeddings):
        # Multi-head attention for embedding fusion
        attention = torch.matmul(embeddings, embeddings.transpose(-2, -1))
        attention = F.softmax(attention / math.sqrt(embeddings.size(-1)), dim=-1)
        return torch.matmul(attention, embeddings).mean(dim=1)

    def full_sort_predict(self, interaction):
        users = interaction[0]
        
        user_emb, item_emb = self.forward(users)
        scores = torch.matmul(user_emb, item_emb.t())
        
        return scores

class PositionalEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.position_enc = self.create_sinusoidal_positions(num_embeddings, embedding_dim)
        
    def create_sinusoidal_positions(self, num_pos, dim):
        position = torch.arange(num_pos, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pos_enc = torch.zeros(num_pos, dim)
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        return nn.Parameter(pos_enc, requires_grad=False)
    
    def forward(self, x):
        return self.embedding(x) + self.position_enc[x]

class ModalityProcessor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, dropout):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation='gelu'
            ) for _ in range(num_layers)
        ])
        
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout)
        
    def forward(self, x):
        x = self.input_proj(x).unsqueeze(0)
        
        for layer in self.transformer_layers:
            x = layer(x)
            
        att_output, att_weights = self.attention(x, x, x)
        
        return att_output.squeeze(0), att_weights

class MultiModalFusionTransformer(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(dim, num_heads, dropout=dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, features, attention_weights=None):
        features = torch.stack(features, dim=0)
        
        if attention_weights is not None:
            attention_weights = torch.stack(attention_weights, dim=0)
            att_output, _ = self.attention(
                features, features, features,
                key_padding_mask=None,
                need_weights=True,
                attn_mask=attention_weights
            )
        else:
            att_output, _ = self.attention(features, features, features)
            
        x = self.norm1(features + self.dropout(att_output))
        x = self.norm2(x + self.dropout(self.feed_forward(x)))
        
        return x.mean(dim=0)

class MultiModalGraphBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        
        self.gat = GATConv(dim, dim // num_heads, heads=num_heads)
        self.sage = SAGEConv(dim, dim, normalize=True)
        self.transformer = TransformerConv(dim, dim, heads=num_heads, dropout=0.1)
        
        self.fusion = nn.Linear(dim * 3, dim)
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x, edge_index):
        gat_out = self.gat(x, edge_index)
        sage_out = self.sage(x, edge_index)
        transformer_out = self.transformer(x, edge_index)
        
        concat = torch.cat([gat_out, sage_out, transformer_out], dim=-1)
        out = self.fusion(concat)
        
        return self.norm(out)

class MLPProjector(nn.Module):
    def __init__(self, dim):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.BatchNorm1d(dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(dim * 2, dim)
        )
        
    def forward(self, x):
        return self.net(x)