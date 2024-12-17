import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity
from common.abstract_recommender import GeneralRecommender
import scipy.sparse as sp
import numpy as np

class MMGAT(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MMGAT, self).__init__(config, dataset)
        
        # Model hyperparameters
        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.num_heads = 4
        self.dropout = 0.1
        self.n_layers = config['n_layers']
        self.temperature = 0.2
        self.lambda_coeff = config['lambda_coeff']
        self.knn_k = config['knn_k']
        self.reg_weight = config['reg_weight']
        self.mm_fusion_mode = 'weight'
        self.n_nodes = self.n_users + self.n_items
        self.alpha = 0.1
        
        # Load dataset info - following FREEDOM's implementation
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        
        # Initialize embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)
        
        # Initialize feature transformations
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)
            self.image_gat = GraphAttentionLayer(self.feat_embed_dim, self.feat_embed_dim, self.dropout, self.alpha)
        
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)
            self.text_gat = GraphAttentionLayer(self.feat_embed_dim, self.feat_embed_dim, self.dropout, self.alpha)

        # Multimodal fusion weights
        if self.v_feat is not None and self.t_feat is not None:
            self.modal_weight = nn.Parameter(torch.Tensor([0.5, 0.5]))
            self.softmax = nn.Softmax(dim=0)
            
        # Get normalized adjacency matrix
        self.norm_adj = self.get_norm_adj_mat().to(self.device)
        self.mm_adj = None
        self.build_item_graph = True

    def get_norm_adj_mat(self):
        # Following FREEDOM's implementation
        A = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        for key, value in data_dict.items():
            A[key] = value
        
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid Divide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        
        # convert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor([row, col])
        data = torch.FloatTensor(L.data)
        
        return torch.sparse.FloatTensor(i, data, torch.Size((self.n_nodes, self.n_nodes)))

    def get_knn_adj_mat(self, mm_embeddings):
        # Following BM3's implementation
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

    def forward(self, build_item_graph=True):
        if build_item_graph:
            # Build multimodal item graph
            if self.v_feat is not None:
                image_feats = self.image_trs(self.image_embedding.weight)
                image_feats = self.image_gat(image_feats, self.norm_adj)
                self.image_adj = self.get_knn_adj_mat(image_feats)[1]
                learned_adj = self.image_adj
            
            if self.t_feat is not None:
                text_feats = self.text_trs(self.text_embedding.weight)
                text_feats = self.text_gat(text_feats, self.norm_adj)
                self.text_adj = self.get_knn_adj_mat(text_feats)[1]
                learned_adj = self.text_adj
            
            if self.v_feat is not None and self.t_feat is not None:
                weight = self.softmax(self.modal_weight)
                learned_adj = weight[0] * self.image_adj + weight[1] * self.text_adj
            
            self.mm_adj = learned_adj

        # Multi-layer Graph Attention
        ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
        all_embeddings = [ego_embeddings]
        
        for _ in range(self.n_layers):
            side_embeddings = torch.sparse.mm(self.norm_adj, ego_embeddings)
            ego_embeddings = F.dropout(side_embeddings, self.dropout, training=self.training)
            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
            all_embeddings.append(norm_embeddings)
        
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)

        return u_g_embeddings, i_g_embeddings

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        u_embeddings, i_embeddings = self.forward(build_item_graph=self.build_item_graph)
        self.build_item_graph = False

        # BPR Loss
        u_embeddings = u_embeddings[users]
        pos_embeddings = i_embeddings[pos_items]
        neg_embeddings = i_embeddings[neg_items]

        pos_scores = torch.sum(u_embeddings * pos_embeddings, dim=1)
        neg_scores = torch.sum(u_embeddings * neg_embeddings, dim=1)
        bpr_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))

        # Mirror Gradient Loss
        with torch.enable_grad():
            mirror_reg = torch.mean(torch.abs(torch.sign(u_embeddings) + torch.sign(pos_embeddings)))
        
        # Contrastive Loss for multimodal features
        contrastive_loss = torch.tensor(0.0, device=self.device)
        if self.v_feat is not None and self.t_feat is not None:
            image_feats = self.image_trs(self.image_embedding.weight[pos_items])
            text_feats = self.text_trs(self.text_embedding.weight[pos_items])
            
            sim_matrix = torch.matmul(image_feats, text_feats.t()) / self.temperature
            labels = torch.arange(sim_matrix.size(0), device=self.device)
            contrastive_loss = F.cross_entropy(sim_matrix, labels) + F.cross_entropy(sim_matrix.t(), labels)

        # L2 regularization
        reg_loss = self.reg_weight * (
            torch.norm(u_embeddings) +
            torch.norm(pos_embeddings) +
            torch.norm(neg_embeddings)
        )

        return bpr_loss + 0.1 * mirror_reg + 0.1 * contrastive_loss + reg_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        u_embeddings, i_embeddings = self.forward(build_item_graph=True)
        
        u_embeddings = u_embeddings[user]
        scores = torch.matmul(u_embeddings, i_embeddings.transpose(0, 1))
        return scores

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, x, adj):
        Wh = torch.mm(x, self.W)
        e = self._prepare_attentional_mechanism_input(Wh)
        
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        h_prime = torch.matmul(attention, Wh)
        
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)