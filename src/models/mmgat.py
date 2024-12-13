import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from common.abstract_recommender import GeneralRecommender
import numpy as np
import scipy.sparse as sp

class MMGAT(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MMGAT, self).__init__(config, dataset)
        
        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.n_layers = config['n_layers']
        self.dropout = config['dropout']
        self.reg_weight = config['reg_weight']
        self.n_nodes = self.n_users + self.n_items
        self.knn_k = config['knn_k']
        self.temp = 0.2
        
        # Load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.norm_adj = self.get_norm_adj_mat().to(self.device)
        
        # Embeddings
        self.user_embedding = nn.Parameter(torch.randn(self.n_users, self.embedding_dim))
        self.item_embedding = nn.Parameter(torch.randn(self.n_items, self.embedding_dim))
        nn.init.xavier_uniform_(self.user_embedding)
        nn.init.xavier_uniform_(self.item_embedding)

        # GNN layers
        self.gnn_layers = nn.ModuleList([
            GATConv(self.embedding_dim, self.embedding_dim, heads=4, dropout=self.dropout)
            for _ in range(self.n_layers)
        ])
        
        # Projection heads
        self.user_proj = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.BatchNorm1d(self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, self.embedding_dim)
        )
        
        self.item_proj = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.BatchNorm1d(self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, self.embedding_dim)
        )
        
        # Modal-specific components
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_encoder = nn.Sequential(
                nn.Linear(self.v_feat.shape[1], self.feat_embed_dim),
                nn.LayerNorm(self.feat_embed_dim),
                nn.ReLU(),
                nn.Linear(self.feat_embed_dim, self.embedding_dim)
            )
            self.image_proj = nn.Sequential(
                nn.Linear(self.embedding_dim, self.embedding_dim),
                nn.BatchNorm1d(self.embedding_dim),
                nn.ReLU(),
                nn.Linear(self.embedding_dim, self.embedding_dim)
            )
            
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_encoder = nn.Sequential(
                nn.Linear(self.t_feat.shape[1], self.feat_embed_dim),
                nn.LayerNorm(self.feat_embed_dim),
                nn.ReLU(),
                nn.Linear(self.feat_embed_dim, self.embedding_dim)
            )
            self.text_proj = nn.Sequential(
                nn.Linear(self.embedding_dim, self.embedding_dim),
                nn.BatchNorm1d(self.embedding_dim),
                nn.ReLU(),
                nn.Linear(self.embedding_dim, self.embedding_dim)
            )

        # Initialize modality fusion weights
        if self.v_feat is not None and self.t_feat is not None:
            self.modal_weights = nn.Parameter(torch.ones(2))

    def get_norm_adj_mat(self):
        adj_mat = sp.dok_matrix((self.n_nodes, self.n_nodes), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.interaction_matrix.tolil()
        
        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()
        
        row_sum = np.array(adj_mat.sum(1))
        d_inv = np.power(row_sum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)
        
        coo = adj_matrix.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def forward(self, adj=None):
        if adj is None:
            adj = self.norm_adj
            
        # Base embeddings
        all_embeddings = torch.cat([self.user_embedding, self.item_embedding])
        embeddings_list = [all_embeddings]
        
        # Graph convolution
        edge_index = adj._indices()
        edge_weight = adj._values()
        
        x = all_embeddings
        for layer in self.gnn_layers:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = layer(x, edge_index)
            x = F.leaky_relu(x)
            norm = torch.norm(x, dim=1, keepdim=True)
            x = x / (norm + 1e-8)
            embeddings_list.append(x)
            
        embeddings = torch.stack(embeddings_list, dim=1)
        embeddings = torch.mean(embeddings, dim=1)
        
        users_emb, items_emb = torch.split(embeddings, [self.n_users, self.n_items])
        
        # Modal features
        modal_emb = 0
        if self.v_feat is not None:
            image_emb = self.image_encoder(self.image_embedding.weight)
            image_emb = self.image_proj(image_emb)
            modal_emb = image_emb
            
        if self.t_feat is not None:
            text_emb = self.text_encoder(self.text_embedding.weight)
            text_emb = self.text_proj(text_emb)
            if self.v_feat is not None:
                weights = F.softmax(self.modal_weights, dim=0)
                modal_emb = weights[0] * image_emb + weights[1] * text_emb
            else:
                modal_emb = text_emb
                
        items_emb = items_emb + F.normalize(modal_emb, p=2, dim=1)
        
        return users_emb, items_emb

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        
        user_embeddings, item_embeddings = self.forward()
        
        # Contrastive Learning Loss
        user_proj = self.user_proj(user_embeddings)
        item_proj = self.item_proj(item_embeddings)
        
        u_norm = F.normalize(user_proj, dim=1)
        i_norm = F.normalize(item_proj, dim=1)
        
        pos_score = torch.sum(torch.mul(u_norm[users], i_norm[pos_items]), dim=1)
        neg_score = torch.sum(torch.mul(u_norm[users], i_norm[neg_items]), dim=1)
        
        loss_con = torch.mean(-torch.log(torch.sigmoid(pos_score/self.temp)) - 
                            torch.log(torch.sigmoid(-neg_score/self.temp)))
        
        # BPR Loss
        batch_users_emb = user_embeddings[users]
        batch_pos_items_emb = item_embeddings[pos_items]
        batch_neg_items_emb = item_embeddings[neg_items]
        
        pos_scores = torch.sum(batch_users_emb * batch_pos_items_emb, dim=1)
        neg_scores = torch.sum(batch_users_emb * batch_neg_items_emb, dim=1)
        
        loss_bpr = torch.mean(-torch.log(torch.sigmoid(pos_scores - neg_scores)))
        
        # L2 regularization
        reg_loss = self.reg_weight * (
            torch.norm(batch_users_emb) +
            torch.norm(batch_pos_items_emb) +
            torch.norm(batch_neg_items_emb)
        )
        
        return loss_bpr + 0.1 * loss_con + reg_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_embeddings, item_embeddings = self.forward()
        
        u_embeddings = user_embeddings[user]
        scores = torch.matmul(u_embeddings, item_embeddings.t())
        
        return scores