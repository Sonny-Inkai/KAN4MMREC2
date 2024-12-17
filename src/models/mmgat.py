import torch
import torch.nn as nn
import torch.nn.functional as F
from common.abstract_recommender import GeneralRecommender
import numpy as np
import scipy.sparse as sp

class MMGAT(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MMGAT, self).__init__(config, dataset)
        
        # Hyperparameters
        self.embedding_dim = config['embedding_size'] 
        self.feat_embed_dim = config['feat_embed_dim']
        self.num_heads = 4
        self.dropout = 0.1
        self.n_layers = config['n_layers']
        self.reg_weight = config['reg_weight']
        self.knn_k = config['knn_k']
        self.temperature = 0.2
        self.lambda_coeff = config['lambda_coeff']
        self.mm_fusion_mode = 'gate'
        self.n_nodes = self.n_users + self.n_items
        
        # Initialize embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        # Load multimodal features
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)
            self.image_attention = nn.MultiheadAttention(self.feat_embed_dim, self.num_heads, dropout=self.dropout, batch_first=True)
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)
            self.text_attention = nn.MultiheadAttention(self.feat_embed_dim, self.num_heads, dropout=self.dropout, batch_first=True)

        # Graph attention layers
        self.gat_layers = nn.ModuleList([
            GraphAttentionLayer(self.embedding_dim, self.embedding_dim, dropout=self.dropout, alpha=0.2)
            for _ in range(self.n_layers)
        ])

        # Multimodal fusion
        if self.mm_fusion_mode == 'gate':
            self.fusion_gate = nn.Linear(self.feat_embed_dim * 2, 2)
        
        # Load adjacency matrix
        self.norm_adj = self.get_norm_adj_mat().to(self.device)
        self.edge_indices, self.edge_values = self.get_edge_info()
        self.edge_indices = self.edge_indices.to(self.device)
        self.edge_values = self.edge_values.to(self.device)
        
        # Mirror gradient parameters
        self.alpha = 0.2  # Mirror coefficient
        self.beta = 0.1   # Gradient scaling factor
        
    def get_norm_adj_mat(self):
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.interaction_matrix.tolil()
        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()

        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj = d_mat_inv.dot(adj_mat).dot(d_mat_inv)
        
        # Convert to sparse tensor
        coo = norm_adj.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def get_edge_info(self):
        rows = torch.from_numpy(self.interaction_matrix.row)
        cols = torch.from_numpy(self.interaction_matrix.col)
        edges = torch.stack([rows, cols])
        
        # Compute normalized edge values
        adj = sp.coo_matrix((np.ones_like(rows), (rows, cols)), 
                          shape=(self.n_users, self.n_items), dtype=np.float32)
        degree_u = np.array(adj.sum(1)).flatten()
        degree_i = np.array(adj.sum(0)).flatten()
        degree_u = np.power(degree_u, -0.5)
        degree_i = np.power(degree_i, -0.5)
        
        values = torch.FloatTensor(degree_u[rows] * degree_i[cols])
        return edges, values

    def compute_normalized_laplacian(self, adj):
        rowsum = adj.sum(1)
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        return d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt

    def forward(self, adj_matrix=None):
        if adj_matrix is None:
            adj_matrix = self.norm_adj
            
        # Process multimodal features
        if self.v_feat is not None:
            image_feats = self.image_trs(self.image_embedding.weight)
            image_feats = F.dropout(image_feats, self.dropout, training=self.training)
            image_feats, _ = self.image_attention(image_feats, image_feats, image_feats)
            
        if self.t_feat is not None:
            text_feats = self.text_trs(self.text_embedding.weight)
            text_feats = F.dropout(text_feats, self.dropout, training=self.training)
            text_feats, _ = self.text_attention(text_feats, text_feats, text_feats)

        # Multimodal fusion
        if self.v_feat is not None and self.t_feat is not None:
            if self.mm_fusion_mode == 'gate':
                concat_feats = torch.cat([image_feats, text_feats], dim=-1)
                gates = torch.softmax(self.fusion_gate(concat_feats), dim=-1)
                item_feats = gates[:, 0].unsqueeze(-1) * image_feats + gates[:, 1].unsqueeze(-1) * text_feats
            else:
                item_feats = (image_feats + text_feats) / 2
        else:
            item_feats = image_feats if self.v_feat is not None else text_feats

        # Graph attention layers
        ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
        all_embeddings = [ego_embeddings]
        
        for gat_layer in self.gat_layers:
            ego_embeddings = gat_layer(ego_embeddings, adj_matrix)
            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
            all_embeddings.append(norm_embeddings)
            
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        i_g_embeddings = i_g_embeddings + item_feats
        
        return u_g_embeddings, i_g_embeddings

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        # Forward pass
        u_embeddings, i_embeddings = self.forward()
        
        # Basic BPR loss
        u_embeddings = u_embeddings[users]
        pos_embeddings = i_embeddings[pos_items]
        neg_embeddings = i_embeddings[neg_items]

        # Positive scores
        pos_scores = torch.sum(u_embeddings * pos_embeddings, dim=1)
        neg_scores = torch.sum(u_embeddings * neg_embeddings, dim=1)
        
        # Mirror gradient loss
        with torch.no_grad():
            mirror_u = u_embeddings.clone()
            mirror_pos = pos_embeddings.clone()
            mirror_neg = neg_embeddings.clone()
            
        mirror_pos_scores = torch.sum(mirror_u * mirror_pos, dim=1)
        mirror_neg_scores = torch.sum(mirror_u * mirror_neg, dim=1)
        
        # BPR loss with temperature scaling
        bpr_loss = -torch.mean(F.logsigmoid((pos_scores - neg_scores) / self.temperature))
        mirror_loss = -self.alpha * torch.mean(F.logsigmoid((mirror_neg_scores - mirror_pos_scores) / self.temperature))
        
        # Contrastive loss for multimodal features
        modal_loss = 0.0
        if self.v_feat is not None and self.t_feat is not None:
            image_feats = self.image_trs(self.image_embedding.weight)
            text_feats = self.text_trs(self.text_embedding.weight)
            
            # Normalize features
            image_feats = F.normalize(image_feats, p=2, dim=1)
            text_feats = F.normalize(text_feats, p=2, dim=1)
            
            # Compute similarity
            pos_sim = torch.sum(image_feats[pos_items] * text_feats[pos_items], dim=1)
            neg_sim = torch.sum(image_feats[neg_items] * text_feats[neg_items], dim=1)
            
            modal_loss = -torch.mean(F.logsigmoid((pos_sim - neg_sim) / self.temperature))
        
        # L2 regularization
        reg_loss = self.reg_weight * (
            torch.norm(u_embeddings) +
            torch.norm(pos_embeddings) +
            torch.norm(neg_embeddings)
        )
        
        return bpr_loss + mirror_loss + self.lambda_coeff * modal_loss + reg_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        
        restore_user_e, restore_item_e = self.forward(self.norm_adj)
        u_embeddings = restore_user_e[user]
        
        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
        return scores

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha

        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Linear(2 * out_features, 1, bias=False)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        
    def forward(self, h, adj):
        Wh = self.W(h)
        e = self._prepare_attentional_mechanism_input(Wh)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)
        return F.elu(h_prime)
        
    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size(0)
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        return self.leakyrelu(self.a(all_combinations_matrix)).view(N, N)