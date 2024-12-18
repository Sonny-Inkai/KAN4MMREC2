import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np

from common.abstract_recommender import GeneralRecommender
from common.loss import EmbLoss

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1, alpha=0.2):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.alpha = alpha
        
        self.W = nn.Parameter(torch.empty(size=(in_dim, out_dim)))
        nn.init.xavier_uniform_(self.W.data)
        
        self.a = nn.Parameter(torch.empty(size=(2*out_dim, 1)))
        nn.init.xavier_uniform_(self.a.data)
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)
    
    def forward(self, h, adj):
        Wh = torch.mm(h, self.W)
        e = self._prepare_attentional_mechanism_input(Wh)
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)
        return F.elu(h_prime)
    
    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_dim, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_dim:, :])
        e = Wh1 + Wh2.transpose(0,1)
        return self.leakyrelu(e)

class CrossModalAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.scale = dim ** -0.5
        self.gate = nn.Sequential(
            nn.Linear(dim*2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
        
    def forward(self, x1, x2):
        q = self.query(x1)
        k = self.key(x2)
        v = self.value(x2)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        
        gate = self.gate(torch.cat([x1, out], dim=-1))
        return x1 + gate * out

class MMGAT(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MMGAT, self).__init__(config, dataset)
        
        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.n_layers = config['n_layers']
        self.n_heads = 4
        self.dropout = config['dropout']
        self.reg_weight = config['reg_weight']
        self.temperature = 0.2
        self.n_nodes = self.n_users + self.n_items
        
        # Load interaction matrix
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.norm_adj = self.get_norm_adj_mat().to(self.device)
        
        # Core embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        
        # Modal encoders with attention
        if self.v_feat is not None:
            self.v_feat = self.v_feat.to(self.device)
            self.image_encoder = nn.Sequential(
                nn.Linear(self.v_feat.shape[1], self.feat_embed_dim),
                nn.LayerNorm(self.feat_embed_dim),
                nn.GELU(),
                nn.Linear(self.feat_embed_dim, self.feat_embed_dim)
            )
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            
        if self.t_feat is not None:
            self.t_feat = self.t_feat.to(self.device)
            self.text_encoder = nn.Sequential(
                nn.Linear(self.t_feat.shape[1], self.feat_embed_dim),
                nn.LayerNorm(self.feat_embed_dim),
                nn.GELU(),
                nn.Linear(self.feat_embed_dim, self.feat_embed_dim)
            )
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
        
        # Cross-modal attention
        self.cross_attention = CrossModalAttention(self.feat_embed_dim)
        
        # Graph attention layers
        self.gat_layers = nn.ModuleList([
            GraphAttentionLayer(self.embedding_dim, self.embedding_dim, self.dropout)
            for _ in range(self.n_layers)
        ])
        
        # Projection head for contrastive learning
        self.projector = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
            nn.GELU(),
            nn.Linear(self.embedding_dim, self.embedding_dim)
        )
        
        self.to(self.device)

    def forward(self):
        # Process modalities
        modal_feats = []
        if self.v_feat is not None:
            img_feat = self.image_encoder(self.image_embedding.weight)
            modal_feats.append(img_feat)
            
        if self.t_feat is not None:
            txt_feat = self.text_encoder(self.text_embedding.weight)
            modal_feats.append(txt_feat)
        
        # Cross-modal fusion with attention
        if len(modal_feats) > 1:
            fused_features = self.cross_attention(modal_feats[0], modal_feats[1])
        elif len(modal_feats) == 1:
            fused_features = modal_feats[0]
        else:
            fused_features = self.item_embedding.weight
        
        # Graph attention with residual connections
        ego_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight])
        all_embeddings = [ego_embeddings]
        
        adj = self.norm_adj.to_dense()
        for gat_layer in self.gat_layers:
            ego_embeddings = gat_layer(ego_embeddings, adj) + ego_embeddings
            all_embeddings.append(ego_embeddings)
        
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items])
        
        # Enhance item embeddings with modal features
        i_g_embeddings = i_g_embeddings + fused_features
        
        return u_g_embeddings, i_g_embeddings

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        
        u_embeddings, i_embeddings = self.forward()
        
        # Project embeddings
        u_embeddings = self.projector(u_embeddings)
        i_embeddings = self.projector(i_embeddings)
        
        u_e = u_embeddings[users]
        pos_e = i_embeddings[pos_items]
        neg_e = i_embeddings[neg_items]
        
        # InfoNCE loss
        pos_scores = torch.sum(u_e * pos_e, dim=1) / self.temperature
        neg_scores = torch.sum(u_e * neg_e, dim=1) / self.temperature
        
        loss_ce = -torch.mean(pos_scores - torch.logsumexp(torch.stack([pos_scores, neg_scores], dim=1), dim=1))
        
        # Hard negative mining
        with torch.no_grad():
            neg_candidates = torch.matmul(u_e, i_embeddings.t())
            neg_candidates[users.unsqueeze(1) == torch.arange(self.n_items).to(self.device)] = -1e10
            _, hard_neg_items = neg_candidates.topk(10, dim=1)
            hard_neg_e = i_embeddings[hard_neg_items]
        
        hard_neg_scores = torch.sum(u_e.unsqueeze(1) * hard_neg_e, dim=2) / self.temperature
        loss_hard = -torch.mean(pos_scores.unsqueeze(1) - torch.logsumexp(hard_neg_scores, dim=1))
        
        # Regularization
        reg_loss = self.reg_weight * (
            torch.norm(u_e) / len(users) +
            torch.norm(pos_e) / len(pos_items) +
            torch.norm(neg_e) / len(neg_items)
        )
        
        return loss_ce + 0.1 * loss_hard + reg_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        
        u_embeddings, i_embeddings = self.forward()
        u_embeddings = self.projector(u_embeddings)
        i_embeddings = self.projector(i_embeddings)
        
        scores = torch.matmul(u_embeddings[user], i_embeddings.transpose(0, 1))
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