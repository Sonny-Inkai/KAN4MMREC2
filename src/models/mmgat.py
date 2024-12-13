import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_sparse import SparseTensor
from common.abstract_recommender import GeneralRecommender
from torch.nn.functional import cosine_similarity

class MMGAT(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MMGAT, self).__init__(config, dataset)
        
        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.n_layers = config['n_layers']
        self.reg_weight = config['reg_weight']
        self.num_heads = 4
        self.dropout = 0.2
        self.n_nodes = self.n_users + self.n_items
        
        # User and item embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)
        
        # Multimodal feature processors
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_encoder = nn.Sequential(
                nn.Linear(self.v_feat.shape[1], self.feat_embed_dim * 2),
                nn.LayerNorm(self.feat_embed_dim * 2),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.feat_embed_dim * 2, self.feat_embed_dim)
            )
        
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_encoder = nn.Sequential(
                nn.Linear(self.t_feat.shape[1], self.feat_embed_dim * 2),
                nn.LayerNorm(self.feat_embed_dim * 2),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.feat_embed_dim * 2, self.feat_embed_dim)
            )
        
        # Graph attention layers
        self.gat_layers = nn.ModuleList([
            GATConv(self.embedding_dim, self.embedding_dim, heads=self.num_heads, dropout=self.dropout)
            for _ in range(self.n_layers)
        ])
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(self.embedding_dim * self.num_heads, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
        # Modal attention
        self.modal_attention = nn.Parameter(torch.ones(2) / 2)
        self.softmax = nn.Softmax(dim=0)
        
        # Initialize adjacency matrix
        self.norm_adj = self.get_norm_adj_mat().to(self.device)
        
    def get_norm_adj_mat(self):
        """Compute normalized adjacency matrix"""
        inter_M = torch.zeros((self.n_users, self.n_items), device=self.device)
        inter_M_indices = torch.tensor(self.dataset.inter_matrix(form='coo').nonzero(), device=self.device)
        inter_M_values = torch.ones(inter_M_indices.shape[1], device=self.device)
        inter_M = torch.sparse_coo_tensor(inter_M_indices, inter_M_values, (self.n_users, self.n_items))
        
        # Build adjacency matrix
        adj_mat = torch.sparse.FloatTensor(
            torch.cat([
                torch.cat([torch.zeros_like(inter_M), inter_M], dim=1),
                torch.cat([inter_M.t(), torch.zeros_like(inter_M.t())], dim=1)
            ], dim=0)
        )
        
        # Normalize
        rowsum = torch.sparse.sum(adj_mat, dim=1).to_dense()
        d_inv = torch.pow(rowsum + 1e-7, -0.5)
        d_mat = torch.diag(d_inv)
        
        norm_adj = torch.sparse.mm(torch.sparse.mm(d_mat, adj_mat), d_mat)
        return norm_adj
        
    def process_graph(self, x, edge_index):
        """Process graph with GAT layers"""
        for gat_layer in self.gat_layers:
            x = gat_layer(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x
        
    def forward(self):
        # Initialize embeddings
        user_emb = self.user_embedding.weight
        item_emb = self.item_id_embedding.weight
        
        # Process multimodal features
        if self.v_feat is not None:
            image_emb = self.image_encoder(self.image_embedding.weight)
            item_emb = item_emb + image_emb
            
        if self.t_feat is not None:
            text_emb = self.text_encoder(self.text_embedding.weight)
            item_emb = item_emb + text_emb
            
        # Combine embeddings
        all_emb = torch.cat([user_emb, item_emb], dim=0)
        
        # Convert sparse adjacency to edge indices for GAT
        edge_index = self.norm_adj._indices()
        
        # Apply graph attention
        enhanced_emb = self.process_graph(all_emb, edge_index)
        enhanced_emb = self.fusion(enhanced_emb)
        
        # Split back user and item embeddings
        user_enhanced, item_enhanced = torch.split(enhanced_emb, [self.n_users, self.n_items])
        
        return user_enhanced, item_enhanced
        
    def calculate_loss(self, interaction):
        user, pos_item, neg_item = interaction[0], interaction[1], interaction[2]
        
        user_all_emb, item_all_emb = self.forward()
        
        user_emb = user_all_emb[user]
        pos_emb = item_all_emb[pos_item]
        neg_emb = item_all_emb[neg_item]
        
        # BPR Loss
        pos_scores = torch.sum(user_emb * pos_emb, dim=1)
        neg_scores = torch.sum(user_emb * neg_emb, dim=1)
        
        bpr_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        
        # Contrastive Loss for multimodal features
        modal_loss = 0.0
        if self.v_feat is not None and self.t_feat is not None:
            image_emb = self.image_encoder(self.image_embedding.weight)
            text_emb = self.text_encoder(self.text_embedding.weight)
            
            modal_sim = cosine_similarity(image_emb[pos_item], text_emb[pos_item])
            modal_loss = -torch.mean(torch.log(torch.sigmoid(modal_sim)))
        
        # L2 regularization
        reg_loss = self.reg_weight * (
            torch.norm(user_emb) +
            torch.norm(pos_emb) +
            torch.norm(neg_emb)
        )
        
        return bpr_loss + 0.1 * modal_loss + reg_loss
    
    def full_sort_predict(self, interaction):
        user = interaction[0]
        
        user_emb, item_emb = self.forward()
        
        scores = torch.matmul(user_emb[user], item_emb.t())
        return scores