import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, degree
import torch_geometric

class MMRECMODEL(nn.Module):
    def __init__(self, config, dataset):
        super(MMRECMODEL, self).__init__()
        self.config = config
        self.dataset = dataset
        self.n_users = dataset.n_users
        self.n_items = dataset.n_items
        self.embedding_dim = config["embedding_size"]
        self.feat_embed_dim = config["feat_embed_dim"]
        self.knn_k = config["knn_k"]
        self.lambda_coeff = config["lambda_coeff"]
        self.cf_model = config["cf_model"]
        self.n_layers = config["n_mm_layers"]
        self.n_ui_layers = config["n_ui_layers"]
        self.reg_weight = config["reg_weight"]
        self.build_item_graph = True
        self.mm_image_weight = config["mm_image_weight"]
        self.dropout = config["dropout"]
        self.degree_ratio = config["degree_ratio"]

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        self.image_embedding = nn.Embedding.from_pretrained(dataset.v_feat, freeze=False)
        self.image_trs = nn.Linear(dataset.v_feat.shape[1], self.feat_embed_dim)
        self.text_embedding = nn.Embedding.from_pretrained(dataset.t_feat, freeze=False)
        self.text_trs = nn.Linear(dataset.t_feat.shape[1], self.feat_embed_dim)

        self.mm_adj = self.get_knn_adj_mat(self.image_embedding.weight.detach(), self.text_embedding.weight.detach())
        self.norm_adj = self.get_norm_adj_mat(dataset.inter_matrix(form="coo").astype(np.float32)).to(self.device)

        self.predictor = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.reg_loss = EmbLoss()

        self.conv_embed_1 = Base_gcn(self.embedding_dim, self.embedding_dim, aggr="add")
        self.conv_embed_2 = Base_gcn(self.embedding_dim, self.embedding_dim, aggr="add")

    def get_knn_adj_mat(self, image_embeddings, text_embeddings):
        context_norm = image_embeddings.div(torch.norm(image_embeddings, p=2, dim=-1, keepdim=True))
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        _, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
        adj_size = sim.size()
        del sim
        indices0 = torch.arange(knn_ind.shape[0]).to(self.device)
        indices0 = torch.unsqueeze(indices0, 1)
        indices0 = indices0.expand(-1, self.knn_k)
        indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return torch.sparse.FloatTensor(indices, values, adj_size)

    def get_norm_adj_mat(self, interaction_matrix):
        A = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        inter_M = interaction_matrix
        inter_M_t = interaction_matrix.transpose()
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
        return torch.sparse.FloatTensor(i, data, torch.Size((self.n_users + self.n_items, self.n_users + self.n_items)))

    def forward(self, adj):
        h = self.item_id_embedding.weight
        for i in range(self.n_layers):
            h = torch.sparse.mm(self.mm_adj, h)
        ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
        all_embeddings = [ego_embeddings]
        for i in range(self.n_ui_layers):
            ego_embeddings = torch.sparse.mm(adj, ego_embeddings)
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        return u_g_embeddings, i_g_embeddings + h

    def calculate_loss(self, interactions):
        users = interactions[0]
        pos_items = interactions[1]
        neg_items = interactions[2]

        u_online_ori, i_online_ori = self.forward(self.norm_adj)
        t_feat_online, v_feat_online = self.text_trs(self.text_embedding.weight), self.image_trs(self.image_embedding.weight)

        with torch.no_grad():
            u_target, i_target = u_online_ori.clone(), i_online_ori.clone()
            u_target.detach()
            i_target.detach()
            u_target = F.dropout(u_target, self.dropout)
            i_target = F.dropout(i_target, self.dropout)

            t_feat_target = t_feat_online.clone()
            t_feat_target = F.dropout(t_feat_target, self.dropout)

            v_feat_target = v_feat_online.clone()
            v_feat_target = F.dropout(v_feat_target, self.dropout)

        u_online, i_online = self.predictor(u_online_ori), self.predictor(i_online_ori)
        t_feat_online = self.predictor(t_feat_online)
        v_feat_online = self.predictor(v_feat_online)

        u_online = u_online[users, :]
        i_online = i_online[pos_items, :]
        t_feat_online = t_feat_online[pos_items, :]
        v_feat_online = v_feat_online[pos_items, :]
        u_target = u_target[users, :]
        i_target = i_target[pos_items, :]
        t_feat_target = t_feat_target[pos_items, :]
        v_feat_target = v_feat_target[pos_items, :]

        loss_t, loss_v, loss_tv, loss_vt = 0.0, 0.0, 0.0, 0.0
        loss_t = self.reg_loss(t_feat_online, t_feat_target)
        loss_v = self.reg_loss(v_feat_online, v_feat_target)
        loss_tv = self.reg_loss(t_feat_online, v_feat_target)
        loss_vt = self.reg_loss(v_feat_online, t_feat_target)

        mf_loss = self.bpr_loss(u_online, i_online, u_target, i_target)
        mf_loss += self.reg_weight * (loss_t + loss_v + loss_tv + loss_vt)
        return mf_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        restore_user_e, restore_item_e = self.forward(self.norm_adj)
        u_embeddings = restore_user_e[user]
        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
        return scores

    def bpr_loss(self, users, pos_items, u_target, i_target):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, i_target), dim=1)
        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)
        return mf_loss

class Base_gcn(MessagePassing):
    def __init__(self, in_channels, out_channels, normalize=True, bias=True, aggr="add", **kwargs):
        super(Base_gcn, self).__init__(aggr=aggr, **kwargs)
        self.aggr = aggr
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x, edge_index, size=None):
        if size is None:
            edge_index, _ = remove_self_loops(edge_index)
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j, edge_index, size):
        if self.aggr == "add":
            row, col = edge_index
            deg = degree(row, size[0], dtype=x_j.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
            return norm.view(-1, 1) * x_j
        return x_j

    def update(self, aggr_out):
        return aggr_out

    def __repr(self):
        return "{}({},{})".format(self.__class__.__name__, self.in_channels, self.out_channels)
