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
        self.n_mm_layers = config['n_mm_layers'] 
        self.dropout = config['dropout']
        self.reg_weight = config['reg_weight']
        self.knn_k = config['knn_k']
        self.lambda_coeff = config['lambda_coeff']
        self.cl_weight = config['cl_weight']
        self.mm_image_weight = config['mm_image_weight']
        self.degree_ratio = config['degree_ratio']
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
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)
            nn.init.xavier_normal_(self.image_trs.weight)
            
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)
            nn.init.xavier_normal_(self.text_trs.weight)

        # Initialize multimodal adjacency matrix
        self.mm_adj = None
        if self.v_feat is not None:
            v_feat_map = self.image_trs(self.image_embedding.weight)
            indices, image_adj = self.get_knn_adj_mat(v_feat_map)
            self.mm_adj = image_adj
            
        if self.t_feat is not None:
            t_feat_map = self.text_trs(self.text_embedding.weight)
            indices, text_adj = self.get_knn_adj_mat(t_feat_map)
            self.mm_adj = text_adj if self.mm_adj is None else self.mm_adj
            
        if self.v_feat is not None and self.t_feat is not None:
            self.mm_adj = self.mm_image_weight * image_adj + (1.0 - self.mm_image_weight) * text_adj

        # Predictor for mirror gradient
        self.predictor = nn.Linear(self.embedding_dim, self.embedding_dim)
        nn.init.xavier_normal_(self.predictor.weight)

    def get_norm_adj_mat(self):
        A = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
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
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        return torch.sparse.FloatTensor(i, data, torch.Size((self.n_nodes, self.n_nodes)))

    def get_knn_adj_mat(self, mm_embeddings):
        context_norm = mm_embeddings.div(torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True))
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        _, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
        adj_size = sim.size()
        del sim
        
        indices0 = torch.arange(knn_ind.shape[0]).to(self.device)
        indices0 = torch.unsqueeze(indices0, 1)
        indices0 = indices0.expand(-1, self.knn_k)
        indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)
        
        return indices, self.compute_normalized_laplacian(indices, adj_size)

    def compute_normalized_laplacian(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return torch.sparse.FloatTensor(indices, values, adj_size)

    def forward(self):
        ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
        all_embeddings = [ego_embeddings]
        
        for _ in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.norm_adj, ego_embeddings)
            all_embeddings.append(ego_embeddings)
            
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        
        # Process multimodal features
        if self.mm_adj is not None:
            mm_embeddings = i_g_embeddings
            for _ in range(self.n_mm_layers):
                mm_embeddings = torch.sparse.mm(self.mm_adj, mm_embeddings)
            i_g_embeddings = i_g_embeddings + mm_embeddings
            
        return u_g_embeddings, i_g_embeddings

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        u_embeddings, i_embeddings = self.forward()
        
        # Mirror gradient
        with torch.no_grad():
            u_target = u_embeddings.clone()
            i_target = i_embeddings.clone()
            u_target = F.dropout(u_target, self.dropout)
            i_target = F.dropout(i_target, self.dropout)

        u_online = self.predictor(u_embeddings)
        i_online = self.predictor(i_embeddings)

        # BPR Loss
        u_e = u_online[users]
        pos_e = i_online[pos_items]
        neg_e = i_online[neg_items]
        
        pos_scores = torch.sum(u_e * pos_e, dim=1)
        neg_scores = torch.sum(u_e * neg_e, dim=1)
        bpr_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))

        # Contrastive Loss  
        u_target = u_target[users]
        i_target = i_target[pos_items]
        loss_ui = 1 - F.cosine_similarity(u_e, i_target.detach(), dim=-1).mean()
        loss_iu = 1 - F.cosine_similarity(pos_e, u_target.detach(), dim=-1).mean()
        cl_loss = (loss_ui + loss_iu) * self.cl_weight

        # L2 regularization
        l2_loss = self.reg_weight * (torch.norm(u_e) + torch.norm(pos_e) + torch.norm(neg_e))

        return bpr_loss + cl_loss + l2_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        
        u_embeddings, i_embeddings = self.forward()
        u_embeddings = self.predictor(u_embeddings)
        i_embeddings = self.predictor(i_embeddings)
        
        scores = torch.matmul(u_embeddings[user], i_embeddings.transpose(0, 1))
        return scores