import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np

from common.abstract_recommender import GeneralRecommender
from common.loss import EmbLoss

class LightGCNConv(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, adj):
        return torch.sparse.mm(adj, x)

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return self.norm(x)

class ModalFusionTransformer(nn.Module):
    def __init__(self, dim, num_heads=4, mlp_ratio=4.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class MMGAT(GeneralRecommender):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        
        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.n_layers = config['n_layers']
        self.dropout = config['dropout']
        self.reg_weight = config['reg_weight']
        
        self.n_nodes = self.n_users + self.n_items
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.norm_adj = self.get_norm_adj_mat().to(self.device)
        
        # Embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)
        
        # Modal Encoders
        self.modal_encoders = nn.ModuleDict()
        if self.v_feat is not None:
            self.v_feat = self.v_feat.to(self.device)
            self.modal_encoders['image'] = nn.Sequential(
                nn.Linear(self.v_feat.shape[1], self.feat_embed_dim),
                nn.LayerNorm(self.feat_embed_dim),
                nn.GELU(),
                nn.Linear(self.feat_embed_dim, self.feat_embed_dim)
            )
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            
        if self.t_feat is not None:
            self.t_feat = self.t_feat.to(self.device)
            self.modal_encoders['text'] = nn.Sequential(
                nn.Linear(self.t_feat.shape[1], self.feat_embed_dim),
                nn.LayerNorm(self.feat_embed_dim),
                nn.GELU(),
                nn.Linear(self.feat_embed_dim, self.feat_embed_dim)
            )
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
        
        # Graph Convolution
        self.gcn_layers = nn.ModuleList([LightGCNConv() for _ in range(self.n_layers)])
        
        # Modal Fusion
        self.modal_fusion = ModalFusionTransformer(self.feat_embed_dim)
        
        # Projection heads
        self.projector = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
            nn.GELU(),
            nn.Linear(self.embedding_dim, self.embedding_dim)
        )
        
        self.to(self.device)
        
    def encode_modalities(self):
        modal_embeddings = []
        
        if 'image' in self.modal_encoders:
            img_emb = self.modal_encoders['image'](self.image_embedding.weight)
            modal_embeddings.append(img_emb)
            
        if 'text' in self.modal_encoders:
            txt_emb = self.modal_encoders['text'](self.text_embedding.weight)
            modal_embeddings.append(txt_emb)
            
        if not modal_embeddings:
            return None
            
        # Stack modalities and apply fusion transformer
        modal_stack = torch.stack(modal_embeddings, dim=1)
        fused = self.modal_fusion(modal_stack)
        return torch.mean(fused, dim=1)
    
    def forward(self):
        # Graph Convolution
        h = torch.cat([self.user_embedding.weight, self.item_embedding.weight])
        all_embeddings = [h]
        
        for layer in self.gcn_layers:
            h = layer(h, self.norm_adj)
            all_embeddings.append(h)
            
        h = torch.mean(torch.stack(all_embeddings, dim=1), dim=1)
        user_embeddings, item_embeddings = torch.split(h, [self.n_users, self.n_items])
        
        # Modal Enhancement
        modal_embeddings = self.encode_modalities()
        if modal_embeddings is not None:
            item_embeddings = item_embeddings + modal_embeddings
            
        return user_embeddings, item_embeddings
        
    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        
        user_embeddings, item_embeddings = self.forward()
        
        # Project embeddings
        user_embeddings = self.projector(user_embeddings)
        item_embeddings = self.projector(item_embeddings)
        
        u_embeddings = user_embeddings[users]
        pos_embeddings = item_embeddings[pos_items]
        neg_embeddings = item_embeddings[neg_items]
        
        # InfoNCE loss
        pos_scores = (u_embeddings * pos_embeddings).sum(dim=1)
        neg_scores = (u_embeddings * neg_embeddings).sum(dim=1)
        
        loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        
        # Modal Alignment Loss
        if hasattr(self, 'image_embedding') and hasattr(self, 'text_embedding'):
            img_feats = self.modal_encoders['image'](self.image_embedding.weight)
            txt_feats = self.modal_encoders['text'](self.text_embedding.weight)
            
            align_matrix = torch.matmul(F.normalize(img_feats, dim=-1),
                                      F.normalize(txt_feats, dim=-1).t())
            align_loss = -torch.mean(torch.diag(F.log_softmax(align_matrix, dim=1)))
            loss = loss + 0.1 * align_loss
        
        # L2 Regularization
        l2_loss = self.reg_weight * (
            torch.norm(u_embeddings) +
            torch.norm(pos_embeddings) +
            torch.norm(neg_embeddings)
        )
        
        return loss + l2_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_embeddings, item_embeddings = self.forward()
        
        user_embeddings = self.projector(user_embeddings)
        item_embeddings = self.projector(item_embeddings)
        
        scores = torch.matmul(user_embeddings[user], item_embeddings.transpose(0, 1))
        return scores

    def get_norm_adj_mat(self):
        A = sp.dok_matrix((self.n_nodes, self.n_nodes), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        for key, value in data_dict.items():
            A[key] = value
        
        sumArr = (A > 0).sum(axis=1)
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        L = sp.coo_matrix(L)
        indices = torch.LongTensor(np.array([L.row, L.col]))
        values = torch.FloatTensor(L.data)
        return torch.sparse_coo_tensor(indices, values, torch.Size((self.n_nodes, self.n_nodes)))