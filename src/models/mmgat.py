from common.abstract_recommender import GeneralRecommender
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from torch.nn.functional import cosine_similarity

class MMGAT(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MMGAT, self).__init__(config, dataset)
        
        self.embedding_dim = 64 
        self.feat_embed_dim = 64
        self.num_heads = 4
        self.dropout = 0.2
        self.n_layers = 2
        self.reg_weight = 0.001
        self.mm_fusion_mode = 'weighted_sum'
        self.temperature = 0.2
        self.knn_k = 10
        
        self.n_nodes = self.n_users + self.n_items
    
        
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        
        # Move embeddings to device during initialization
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim).to(self.device)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim).to(self.device)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        # Feature transformations with explicit device placement
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False).to(self.device)
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim).to(self.device)
            self.image_attn = nn.MultiheadAttention(self.feat_embed_dim, self.num_heads, dropout=self.dropout, batch_first=True).to(self.device)
            
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False).to(self.device)
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim).to(self.device)
            self.text_attn = nn.MultiheadAttention(self.feat_embed_dim, self.num_heads, dropout=self.dropout, batch_first=True).to(self.device)
        
        self.modal_weights = nn.Parameter(torch.FloatTensor([0.5, 0.5]).to(self.device))
        self.softmax = nn.Softmax(dim=0)
        
        self.predictor = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, self.embedding_dim)
        ).to(self.device)
        
        self.mm_adj = None
        self.norm_adj = self.get_norm_adj_mat().to(self.device)
        
        if self.v_feat is not None or self.t_feat is not None:
            self._init_mm_adj()

    def get_norm_adj_mat(self):
        adj_mat = sp.dok_matrix((self.n_nodes, self.n_nodes), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.interaction_matrix.tolil()
        
        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()
        
        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum + 1e-12, -0.5).flatten() # Added epsilon to prevent divide by zero
        d_mat_inv = sp.diags(d_inv)
        norm_adj = d_mat_inv.dot(adj_mat).dot(d_mat_inv)
        
        norm_adj = norm_adj.tocoo()
        values = norm_adj.data
        indices = np.vstack((norm_adj.row, norm_adj.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = torch.Size(norm_adj.shape)
        
        return torch.sparse_coo_tensor(i, v, shape).to(self.device)

    def _build_knn_graph(self, features):
        # Ensure features are on correct device
        features = features.to(self.device)
        sim = torch.mm(features, features.t())
        _, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
        adj_size = sim.size()
        
        rows = torch.arange(knn_ind.size(0), device=self.device).view(-1, 1).repeat(1, self.knn_k)
        adj = torch.sparse_coo_tensor(
            torch.stack([rows.flatten(), knn_ind.flatten()]),
            torch.ones(rows.numel(), device=self.device),
            adj_size
        ).to(self.device)
        
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        values = r_inv_sqrt[rows.flatten()] * r_inv_sqrt[knn_ind.flatten()]
        
        return torch.sparse_coo_tensor(adj._indices(), values, adj_size).to(self.device)

    def _init_mm_adj(self):
        if self.v_feat is not None:
            image_feats = self.image_trs(self.image_embedding.weight.to(self.device))
            image_adj = self._build_knn_graph(image_feats)
            self.mm_adj = image_adj
            
        if self.t_feat is not None:
            text_feats = self.text_trs(self.text_embedding.weight.to(self.device))
            text_adj = self._build_knn_graph(text_feats)
            if self.mm_adj is None:
                self.mm_adj = text_adj
            else:
                weights = self.softmax(self.modal_weights)
                self.mm_adj = weights[0] * image_adj + weights[1] * text_adj

    def forward(self):
        if self.v_feat is not None:
            image_feats = self.image_trs(self.image_embedding.weight)
            image_feats = image_feats.unsqueeze(0)
            image_feats, _ = self.image_attn(image_feats, image_feats, image_feats)
            image_feats = image_feats.squeeze(0)
            
        if self.t_feat is not None:
            text_feats = self.text_trs(self.text_embedding.weight)
            text_feats = text_feats.unsqueeze(0)
            text_feats, _ = self.text_attn(text_feats, text_feats, text_feats)
            text_feats = text_feats.squeeze(0)

        if self.v_feat is not None and self.t_feat is not None:
            weights = self.softmax(self.modal_weights)
            item_feats = weights[0] * image_feats + weights[1] * text_feats
        elif self.v_feat is not None:
            item_feats = image_feats
        elif self.t_feat is not None:
            item_feats = text_feats
        else:
            item_feats = self.item_id_embedding.weight

        ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
        all_embeddings = [ego_embeddings]
        
        for _ in range(self.n_layers):
            side_embeddings = torch.sparse.mm(self.norm_adj, ego_embeddings)
            item_embeddings = torch.sparse.mm(self.mm_adj, item_feats)
            combined_embeddings = torch.cat((side_embeddings[:self.n_users], 
                                          side_embeddings[self.n_users:] + item_embeddings), dim=0)
            
            ego_embeddings = combined_embeddings
            all_embeddings.append(ego_embeddings)
            
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        return u_g_embeddings, i_g_embeddings

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        u_embeddings, i_embeddings = self.forward()
        
        u_online = self.predictor(u_embeddings)
        i_online = self.predictor(i_embeddings)
        
        u_target = u_embeddings.detach()
        i_target = i_embeddings.detach()

        u_e = u_online[users]
        pos_e = i_online[pos_items]
        neg_e = i_online[neg_items]
        
        pos_scores = torch.sum(torch.mul(u_e, pos_e), dim=1)
        neg_scores = torch.sum(torch.mul(u_e, neg_e), dim=1)
        
        bpr_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))

        pos_score = cosine_similarity(u_e, i_target[pos_items], dim=-1) / self.temperature
        neg_score = cosine_similarity(u_e, i_target[neg_items], dim=-1) / self.temperature
        contrastive_loss = -torch.mean(F.logsigmoid(pos_score - neg_score))

        modal_loss = 0.0
        if self.v_feat is not None and self.t_feat is not None:
            image_feats = self.image_trs(self.image_embedding.weight)
            text_feats = self.text_trs(self.text_embedding.weight)
            modal_loss = 1 - torch.mean(cosine_similarity(image_feats[pos_items], text_feats[pos_items]))

        reg_loss = self.reg_weight * (
            torch.norm(u_embeddings[users]) +
            torch.norm(i_embeddings[pos_items]) + 
            torch.norm(i_embeddings[neg_items])
        )

        return bpr_loss + contrastive_loss + modal_loss + reg_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        u_embeddings, i_embeddings = self.forward()
        
        u_embeddings = self.predictor(u_embeddings)
        i_embeddings = self.predictor(i_embeddings)
        
        score_mat = torch.matmul(u_embeddings[user], i_embeddings.transpose(0, 1))
        return score_mat