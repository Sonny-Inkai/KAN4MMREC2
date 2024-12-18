import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np

from common.abstract_recommender import GeneralRecommender

class MMGAT(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MMGAT, self).__init__(config, dataset)
        
        # Basic parameters
        self.embedding_dim = 64  
        self.feat_embed_dim = 64
        self.num_layers = 2
        self.num_heads = 4
        self.dropout = 0.1
        self.reg_weight = 0.001
        self.lambda_coeff = 0.5
        self.mm_fusion_mode = 'weighted'
        self.knn_k = 10
        self.device = config['device']
        
        # Number of users and items from parent class
        self.n_nodes = self.n_users + self.n_items
        
        # Load interaction matrix
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.norm_adj = self.get_normalized_adj().to(self.device)
        
        # Initialize embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)
        
        # Multimodal feature processing
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_transform = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)
            self.image_attention = nn.MultiheadAttention(self.feat_embed_dim, self.num_heads, dropout=self.dropout)
            
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_transform = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)
            self.text_attention = nn.MultiheadAttention(self.feat_embed_dim, self.num_heads, dropout=self.dropout)
        
        # Modality fusion
        self.modal_fusion = nn.Parameter(torch.FloatTensor([0.5, 0.5]))
        self.fusion_layer = nn.Linear(self.feat_embed_dim * 2, self.feat_embed_dim)
        
        # Graph attention layers
        self.gat_layers = nn.ModuleList([
            GraphAttentionLayer(self.embedding_dim, self.embedding_dim, dropout=self.dropout, alpha=0.2) 
            for _ in range(self.num_layers)
        ])

    def get_normalized_adj(self):
        adj_mat = sp.dok_matrix((self.n_nodes, self.n_nodes), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.interaction_matrix.tolil()
        
        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()
        
        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj = d_mat_inv.dot(adj).dot(d_mat_inv)
            return norm_adj.tocoo()
        
        norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        return self._convert_sp_mat_to_tensor(norm_adj_mat)

    def _convert_sp_mat_to_tensor(self, X):
        coo = X.tocoo()
        indices = torch.from_numpy(np.vstack((coo.row, coo.col))).long()
        values = torch.from_numpy(coo.data).float()
        shape = torch.Size(coo.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def forward(self):
        # Process multimodal features
        if self.v_feat is not None:
            image_feats = self.image_transform(self.image_embedding.weight)
            image_feats = self.image_attention(image_feats, image_feats, image_feats)[0]
            
        if self.t_feat is not None:
            text_feats = self.text_transform(self.text_embedding.weight)
            text_feats = self.text_attention(text_feats, text_feats, text_feats)[0]
        
        # Multimodal fusion
        if self.v_feat is not None and self.t_feat is not None:
            weights = F.softmax(self.modal_fusion, dim=0)
            item_mm_feats = self.fusion_layer(
                torch.cat([weights[0] * image_feats, weights[1] * text_feats], dim=1)
            )
        elif self.v_feat is not None:
            item_mm_feats = image_feats
        elif self.t_feat is not None:
            item_mm_feats = text_feats
        
        # Graph attention processing
        user_emb = self.user_embedding.weight
        item_emb = self.item_id_embedding.weight + item_mm_feats
        
        all_emb = torch.cat([user_emb, item_emb], dim=0)
        embs = [all_emb]
        
        # Mirror gradient update
        if self.training:
            all_emb_grad = all_emb.clone()
            for gat_layer in self.gat_layers:
                all_emb = gat_layer(all_emb, self.norm_adj)
                all_emb_grad = gat_layer(all_emb_grad, self.norm_adj)
                all_emb = all_emb + 0.1 * (all_emb - all_emb_grad.detach())
                embs.append(all_emb)
        else:
            for gat_layer in self.gat_layers:
                all_emb = gat_layer(all_emb, self.norm_adj)
                embs.append(all_emb)
                
        embs = torch.stack(embs, dim=1)
        embs = torch.mean(embs, dim=1)
        
        user_emb, item_emb = torch.split(embs, [self.n_users, self.n_items])
        return user_emb, item_emb

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        
        user_emb, item_emb = self.forward()
        
        u_emb = user_emb[users]
        pos_emb = item_emb[pos_items]
        neg_emb = item_emb[neg_items]
        
        # BPR loss with margin
        pos_scores = torch.sum(u_emb * pos_emb, dim=1)
        neg_scores = torch.sum(u_emb * neg_emb, dim=1)
        margin = 1.0
        base_loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores + margin)))
        
        # Adaptive margin loss
        pos_norm = torch.norm(pos_emb, p=2, dim=1)
        neg_norm = torch.norm(neg_emb, p=2, dim=1)
        margin_loss = torch.mean(F.relu(neg_norm - pos_norm + 0.5))
        
        # Contrastive learning loss
        if self.v_feat is not None and self.t_feat is not None:
            image_pos = self.image_transform(self.image_embedding.weight[pos_items])
            text_pos = self.text_transform(self.text_embedding.weight[pos_items])
            modal_loss = 1 - F.cosine_similarity(image_pos, text_pos, dim=1).mean()
        else:
            modal_loss = 0.0
            
        # L2 regularization
        reg_loss = self.reg_weight * (
            torch.norm(u_emb) +
            torch.norm(pos_emb) +
            torch.norm(neg_emb)
        )
        
        return base_loss + 0.1 * margin_loss + 0.1 * modal_loss + reg_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_emb, item_emb = self.forward()
        user_emb = user_emb[user]
        scores = torch.matmul(user_emb, item_emb.t())
        return scores

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Linear(2 * out_features, 1, bias=False)
        
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a.weight)
        
    def forward(self, x, adj):
        Wh = self.W(x)
        e = self._prepare_attentional_mechanism_input(Wh)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)
        return F.elu(h_prime)
        
    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a.weight[0:self.out_features].T)
        Wh2 = torch.matmul(Wh, self.a.weight[self.out_features:].T)
        e = Wh1 + Wh2.T
        return self.alpha * e