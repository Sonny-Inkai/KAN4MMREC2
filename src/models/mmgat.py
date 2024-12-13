import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from common.abstract_recommender import GeneralRecommender

class MMGAT(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MMGAT, self).__init__(config, dataset)
        
        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.n_layers = config['n_layers']
        self.reg_weight = config['reg_weight']
        self.n_heads = 4
        self.dropout = 0.2
        self.slope = 0.2
        
        self.n_nodes = self.n_users + self.n_items
        self.norm_adj = self.get_norm_adj_mat(dataset.inter_matrix(form='coo').astype('float32')).to(self.device)
        
        # User and item embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        # Multimodal feature processing
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_trs = nn.Sequential(
                nn.Linear(self.v_feat.shape[1], self.feat_embed_dim),
                nn.LayerNorm(self.feat_embed_dim),
                nn.Dropout(self.dropout)
            )
            
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_trs = nn.Sequential(
                nn.Linear(self.t_feat.shape[1], self.feat_embed_dim),
                nn.LayerNorm(self.feat_embed_dim),
                nn.Dropout(self.dropout)
            )

        # Graph attention layers
        self.gat_layers = nn.ModuleList([
            GATConv(self.embedding_dim, self.embedding_dim // self.n_heads, heads=self.n_heads, dropout=self.dropout)
            for _ in range(self.n_layers)
        ])
        
        # Cross-modal attention
        self.cross_attn = nn.MultiheadAttention(self.feat_embed_dim, num_heads=4, dropout=self.dropout)
        
        # Output transformation
        self.output_layer = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
            nn.LeakyReLU(self.slope),
            nn.Dropout(self.dropout)
        )

    def get_norm_adj_mat(self, interaction_matrix):
        import scipy.sparse as sp
        import numpy as np
        
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = interaction_matrix.tolil()
        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()

        row_sum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(row_sum + 1e-7, -0.5).flatten()
        d_mat_inv = sp.diags(d_inv)
        norm_adj = d_mat_inv.dot(adj_mat.dot(d_mat_inv))
        
        # Convert to PyTorch sparse tensor
        coo = norm_adj.tocoo()
        indices = torch.from_numpy(np.vstack((coo.row, coo.col))).long()
        values = torch.from_numpy(coo.data).float()
        shape = torch.Size(coo.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def process_graph(self, edge_index, x):
        for gat_layer in self.gat_layers:
            x = gat_layer(x, edge_index)
            x = F.leaky_relu(x, self.slope)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def forward(self):
        # Convert sparse adjacency matrix to edge index format
        edge_index = self.norm_adj._indices()
        
        # Initial embeddings
        user_emb = self.user_embedding.weight
        item_emb = self.item_id_embedding.weight
        x = torch.cat([user_emb, item_emb], dim=0)
        
        # Process with GAT layers
        x = self.process_graph(edge_index, x)
        
        # Split user and item embeddings
        user_embeddings, item_embeddings = torch.split(x, [self.n_users, self.n_items])
        
        # Process multimodal features
        modal_emb = None
        if self.v_feat is not None and self.t_feat is not None:
            img_feats = self.image_trs(self.image_embedding.weight)
            txt_feats = self.text_trs(self.text_embedding.weight)
            
            # Cross-modal attention
            modal_emb, _ = self.cross_attn(
                img_feats.unsqueeze(0),
                txt_feats.unsqueeze(0),
                txt_feats.unsqueeze(0)
            )
            modal_emb = modal_emb.squeeze(0)
            
        elif self.v_feat is not None:
            modal_emb = self.image_trs(self.image_embedding.weight)
        elif self.t_feat is not None:
            modal_emb = self.text_trs(self.text_embedding.weight)
            
        # Combine embeddings with modal features if available
        if modal_emb is not None:
            item_embeddings = self.output_layer(
                torch.cat([item_embeddings, modal_emb], dim=-1)
            )
            
        return user_embeddings, item_embeddings

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        
        user_embeddings, item_embeddings = self.forward()
        
        # Get embeddings for specific users and items
        u_embeddings = user_embeddings[users]
        pos_embeddings = item_embeddings[pos_items]
        neg_embeddings = item_embeddings[neg_items]
        
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
        user_embeddings, item_embeddings = self.forward()
        
        u_embeddings = user_embeddings[user]
        scores = torch.matmul(u_embeddings, item_embeddings.t())
        
        return scores