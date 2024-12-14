import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss

class MMGAT(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MMGAT, self).__init__(config, dataset)
        
        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.n_layers = config['n_layers']
        self.dropout = config['dropout']
        self.reg_weight = config['reg_weight']
        self.temp = config['temperature']
        
        self.n_nodes = self.n_users + self.n_items
        
        # Get normalized interaction matrix
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.norm_adj = self.get_norm_adj_mat().to(self.device)
        
        # Initialize embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)
        
        # Feature transformation layers
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_projector = ProjectionHead(self.v_feat.shape[1], self.feat_embed_dim, self.dropout)
            self.image_momentum_projector = ProjectionHead(self.v_feat.shape[1], self.feat_embed_dim, self.dropout)
            self._init_momentum_parameters(self.image_projector, self.image_momentum_projector)
            
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_projector = ProjectionHead(self.t_feat.shape[1], self.feat_embed_dim, self.dropout)
            self.text_momentum_projector = ProjectionHead(self.t_feat.shape[1], self.feat_embed_dim, self.dropout)
            self._init_momentum_parameters(self.text_projector, self.text_momentum_projector)
        
        # Fusion layer
        if self.v_feat is not None and self.t_feat is not None:
            self.fusion_layer = nn.Sequential(
                nn.Linear(self.feat_embed_dim * 2, self.embedding_dim),
                nn.LayerNorm(self.embedding_dim),
                nn.Dropout(self.dropout),
                nn.ReLU()
            )
        
        self.momentum = 0.999
        self.adaptive_weight = nn.Parameter(torch.FloatTensor([0.5, 0.5]))
        
    def _init_momentum_parameters(self, online_net, momentum_net):
        """Initialize momentum network as a copy of online network"""
        for param_q, param_k in zip(online_net.parameters(), momentum_net.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
            
    def get_norm_adj_mat(self):
        adj_mat = sp.dok_matrix((self.n_nodes, self.n_nodes), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.interaction_matrix.tolil()
        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()
        
        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum + 1e-9, -0.5).flatten()
        d_mat_inv = sp.diags(d_inv)
        norm_adj = d_mat_inv.dot(adj_mat).dot(d_mat_inv)
        
        norm_adj = norm_adj.tocoo()
        values = norm_adj.data
        indices = np.vstack((norm_adj.row, norm_adj.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = torch.Size(norm_adj.shape)
        return torch.sparse_coo_tensor(i, v, shape)
    
    def forward(self):
        # Process multimodal features
        if self.v_feat is not None:
            image_feats = self.image_projector(self.image_embedding.weight)
        if self.t_feat is not None:
            text_feats = self.text_projector(self.text_embedding.weight)
            
        # Adaptive feature fusion
        if self.v_feat is not None and self.t_feat is not None:
            weights = F.softmax(self.adaptive_weight, dim=0)
            fused_feats = self.fusion_layer(
                torch.cat([weights[0] * image_feats, weights[1] * text_feats], dim=1)
            )
        elif self.v_feat is not None:
            fused_feats = image_feats
        else:
            fused_feats = text_feats
            
        # Graph convolution
        ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
        all_embeddings = [F.normalize(ego_embeddings, p=2, dim=1)]
        
        for layer in range(self.n_layers):
            side_embeddings = torch.sparse.mm(self.norm_adj, ego_embeddings)
            ego_embeddings = F.normalize(side_embeddings + ego_embeddings, p=2, dim=1)
            if layer < self.n_layers - 1:
                ego_embeddings = F.dropout(ego_embeddings, p=self.dropout, training=self.training)
            all_embeddings.append(ego_embeddings)
            
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        i_g_embeddings = i_g_embeddings + fused_feats
        
        return u_g_embeddings, i_g_embeddings
        
    @torch.no_grad()    
    def _momentum_update(self):
        if self.v_feat is not None:
            for param_q, param_k in zip(self.image_projector.parameters(), 
                                      self.image_momentum_projector.parameters()):
                param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)
        if self.t_feat is not None:
            for param_q, param_k in zip(self.text_projector.parameters(),
                                      self.text_momentum_projector.parameters()):
                param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)
    
    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        
        u_embeddings, i_embeddings = self.forward()
        
        u_embeddings = F.normalize(u_embeddings[users], p=2, dim=1)
        pos_embeddings = F.normalize(i_embeddings[pos_items], p=2, dim=1)
        neg_embeddings = F.normalize(i_embeddings[neg_items], p=2, dim=1)
        
        pos_scores = torch.sum(u_embeddings * pos_embeddings, dim=1)
        neg_scores = torch.sum(u_embeddings * neg_embeddings, dim=1)
        
        bpr_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        
        # Contrastive learning with momentum encoders
        contrastive_loss = 0.0
        if self.v_feat is not None:
            self._momentum_update()
            q_v = self.image_projector(self.image_embedding.weight[pos_items])
            k_v = self.image_momentum_projector(self.image_embedding.weight[pos_items])
            contrastive_loss += self._contrastive_loss(q_v, k_v)
            
        if self.t_feat is not None:
            q_t = self.text_projector(self.text_embedding.weight[pos_items])
            k_t = self.text_momentum_projector(self.text_embedding.weight[pos_items])
            contrastive_loss += self._contrastive_loss(q_t, k_t)
            
        reg_loss = self.reg_weight * (
            torch.norm(u_embeddings) +
            torch.norm(pos_embeddings) +
            torch.norm(neg_embeddings)
        )
        
        return bpr_loss + 0.1 * contrastive_loss + reg_loss
    
    def _contrastive_loss(self, q, k):
        q = F.normalize(q, dim=1)
        k = F.normalize(k, dim=1)
        
        pos_logit = torch.sum(q * k, dim=1, keepdim=True)
        neg_logits = torch.mm(q, k.t())
        
        logits = torch.cat([pos_logit, neg_logits], dim=1) / self.temp
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        
        return F.cross_entropy(logits, labels)
    
    def full_sort_predict(self, interaction):
        user = interaction[0]
        u_embeddings, i_embeddings = self.forward()
        
        u_embeddings = F.normalize(u_embeddings[user], p=2, dim=1)
        i_embeddings = F.normalize(i_embeddings, p=2, dim=1)
        
        scores = torch.matmul(u_embeddings, i_embeddings.t())
        return scores

class ProjectionHead(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout),
            nn.ReLU()
        )
        
    def forward(self, x):
        return self.net(x)