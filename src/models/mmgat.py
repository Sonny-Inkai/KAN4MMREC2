import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np

from common.abstract_recommender import GeneralRecommender

class MMGAT(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MMGAT, self).__init__(config, dataset)
        
        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.n_layers = config['n_layers']
        self.dropout = config['dropout']
        self.reg_weight = config['reg_weight']
        self.n_nodes = self.n_users + self.n_items
        
        # Load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.norm_adj = self.get_norm_adj_mat().to(self.device)
        
        # User and item embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)
        
        # Multi-modal feature processors
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_encoder = nn.Sequential(
                nn.Linear(self.v_feat.shape[1], self.feat_embed_dim),
                nn.LayerNorm(self.feat_embed_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout)
            )
            self.image_projector = nn.Linear(self.feat_embed_dim, self.embedding_dim)
            
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_encoder = nn.Sequential(
                nn.Linear(self.t_feat.shape[1], self.feat_embed_dim),
                nn.LayerNorm(self.feat_embed_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout)
            )
            self.text_projector = nn.Linear(self.feat_embed_dim, self.embedding_dim)
        
        # Graph attention layers
        self.gnn_layers = nn.ModuleList([
            LightGCNLayer(self.embedding_dim) 
            for _ in range(self.n_layers)
        ])
        
        # Modal fusion
        self.modal_fusion = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, self.embedding_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.embedding_dim, self.embedding_dim)
        ) if self.v_feat is not None and self.t_feat is not None else None
        
        # Adaptive weights
        self.adaptive_weight = nn.Parameter(torch.ones(3) / 3)
        self.softmax = nn.Softmax(dim=0)
        
    def get_norm_adj_mat(self):
        adj_mat = sp.dok_matrix((self.n_nodes, self.n_nodes), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.interaction_matrix.tolil()
        
        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()
        
        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        
        norm_adj = d_mat_inv.dot(adj_mat.dot(d_mat_inv))
        return self._sparse_mx_to_torch_sparse_tensor(norm_adj.tocoo())
    
    def _sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col))).long()
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)
    
    def forward(self):
        # Process multi-modal features
        modal_embeds = []
        if self.v_feat is not None:
            image_feats = self.image_encoder(self.image_embedding.weight)
            image_embeds = self.image_projector(image_feats)
            modal_embeds.append(image_embeds)
            
        if self.t_feat is not None:
            text_feats = self.text_encoder(self.text_embedding.weight)
            text_embeds = self.text_projector(text_feats)
            modal_embeds.append(text_embeds)
        
        # Combine modal embeddings if both exist
        if len(modal_embeds) == 2:
            item_modal_embeds = self.modal_fusion(
                torch.cat([modal_embeds[0], modal_embeds[1]], dim=-1)
            )
        elif len(modal_embeds) == 1:
            item_modal_embeds = modal_embeds[0]
        else:
            item_modal_embeds = torch.zeros_like(self.item_id_embedding.weight)
        
        # Graph neural network
        ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
        all_embeddings = [ego_embeddings]
        
        for gnn in self.gnn_layers:
            ego_embeddings = gnn(ego_embeddings, self.norm_adj)
            all_embeddings.append(ego_embeddings)
            
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        
        # Adaptive fusion
        weights = self.softmax(self.adaptive_weight)
        i_embeddings = weights[0] * i_g_embeddings + \
                      weights[1] * item_modal_embeds + \
                      weights[2] * self.item_id_embedding.weight
        
        return u_g_embeddings, i_embeddings
    
    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        
        u_embeddings, i_embeddings = self.forward()
        
        u_embeddings = u_embeddings[users]
        pos_embeddings = i_embeddings[pos_items]
        neg_embeddings = i_embeddings[neg_items]
        
        # BPR loss
        pos_scores = torch.sum(u_embeddings * pos_embeddings, dim=1)
        neg_scores = torch.sum(u_embeddings * neg_embeddings, dim=1)
        bpr_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        
        # L2 regularization
        reg_loss = self.reg_weight * (
            torch.norm(u_embeddings) +
            torch.norm(pos_embeddings) +
            torch.norm(neg_embeddings)
        )
        
        return bpr_loss + reg_loss
    
    def full_sort_predict(self, interaction):
        user = interaction[0]
        
        u_embeddings, i_embeddings = self.forward()
        u_embeddings = u_embeddings[user]
        
        scores = torch.matmul(u_embeddings, i_embeddings.transpose(0, 1))
        return scores

class LightGCNLayer(nn.Module):
    def __init__(self, embedding_dim):
        super(LightGCNLayer, self).__init__()
        self.embedding_dim = embedding_dim
        
    def forward(self, x, adj):
        return torch.sparse.mm(adj, x)