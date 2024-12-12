import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SAGEConv
from common.abstract_recommender import GeneralRecommender

class MMGAT(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MMGAT, self).__init__(config, dataset)
        
        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.num_heads = 8
        self.dropout = config['dropout']
        self.reg_weight = config['reg_weight']
        self.n_layers = 2
        
        # Base embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        
        # Modal processors
        if self.v_feat is not None:
            self.image_proj = ModalProjector(
                self.v_feat.shape[1], 
                self.feat_embed_dim,
                self.dropout
            )
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            
        if self.t_feat is not None:
            self.text_proj = ModalProjector(
                self.t_feat.shape[1],
                self.feat_embed_dim,
                self.dropout
            )
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
        
        # Fusion module
        self.modal_fusion = ModalFusion(
            self.feat_embed_dim,
            self.num_heads,
            self.dropout
        )
        
        # Graph layers
        self.gnn_layers = nn.ModuleList([
            GraphLayer(
                self.embedding_dim,
                self.num_heads
            ) for _ in range(self.n_layers)
        ])
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
            
    def forward(self, users, pos_items=None, neg_items=None):
        # Process modalities
        modal_feats = []
        
        if self.v_feat is not None:
            img_feat = self.image_proj(self.image_embedding.weight)
            modal_feats.append(img_feat)
            
        if self.t_feat is not None:
            txt_feat = self.text_proj(self.text_embedding.weight)
            modal_feats.append(txt_feat)
        
        # Fuse modalities
        if len(modal_feats) > 1:
            item_feat = self.modal_fusion(modal_feats)
        else:
            item_feat = modal_feats[0]
            
        # Get base embeddings
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight + item_feat
        
        # Graph convolution
        all_emb = torch.cat([user_emb, item_emb], dim=0)
        emb_list = [all_emb]
        
        for layer in self.gnn_layers:
            all_emb = layer(all_emb)
            emb_list.append(all_emb)
            
        all_emb = torch.stack(emb_list, dim=1)
        all_emb = torch.mean(all_emb, dim=1)
        
        user_emb, item_emb = torch.split(all_emb, [self.n_users, self.n_items])
        
        if pos_items is not None:
            user_emb = user_emb[users]
            pos_emb = item_emb[pos_items]
            neg_emb = item_emb[neg_items]
            return user_emb, pos_emb, neg_emb
            
        return user_emb[users], item_emb
        
    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        
        user_emb, pos_emb, neg_emb = self.forward(users, pos_items, neg_items)
        
        # BPR loss
        pos_scores = (user_emb * pos_emb).sum(dim=1)
        neg_scores = (user_emb * neg_emb).sum(dim=1)
        loss = -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()
        
        # Regularization
        reg_loss = self.reg_weight * (
            user_emb.norm(2).pow(2) +
            pos_emb.norm(2).pow(2) +
            neg_emb.norm(2).pow(2)
        ) / len(users)
        
        return loss + reg_loss

    def full_sort_predict(self, interaction):
        users = interaction[0]
        user_emb, item_emb = self.forward(users)
        scores = torch.matmul(user_emb, item_emb.t())
        return scores

class ModalProjector(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
    def forward(self, x):
        return self.net(x)

class ModalFusion(nn.Module):
    def __init__(self, dim, num_heads, dropout):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim)
        )
        
    def forward(self, features):
        # Stack features: [num_modalities, num_items, dim]
        x = torch.stack(features, dim=0)
        batch_size = x.size(1)
        
        # Reshape: [num_items, num_modalities, dim]
        x = x.transpose(0, 1)
        
        # Self-attention
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_out)
        
        # FFN
        x = self.norm2(x + self.ffn(x))
        
        # Mean pooling across modalities
        return x.mean(dim=1)

class GraphLayer(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        
        self.gat = GATConv(dim, dim // num_heads, heads=num_heads)
        self.sage = SAGEConv(dim, dim)
        self.proj = nn.Linear(dim * 2, dim)
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        gat_out = self.gat(x, edge_index)
        sage_out = self.sage(x, edge_index)
        
        concat = torch.cat([gat_out, sage_out], dim=-1)
        out = self.proj(concat)
        
        return self.norm(out + x)