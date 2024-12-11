import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, degree
import torch_geometric

from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha

        self.W = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(in_features, out_features).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=1.414))
        self.a = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(2*out_features, 1).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=1.414))

    def forward(self, input, adj):
        h = torch.matmul(input, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        a_input = self._prepare_attention_input(h)
        e = F.leaky_relu(torch.matmul(a_input, self.a).squeeze(2)) # e.shape: (N, N)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        return F.elu(h_prime)

    def _prepare_attention_input(self, Wh):
        N = Wh.size()[0] # number of nodes

        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)

        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)

        return all_combinations_matrix

class MultiModalFusion(nn.Module):
    def __init__(self, embedding_dim, feat_embed_dim):
        super(MultiModalFusion, self).__init__()
        self.embedding_dim = embedding_dim
        self.feat_embed_dim = feat_embed_dim
        self.fc = nn.Linear(embedding_dim + feat_embed_dim, embedding_dim)

    def forward(self, x, feat):
        x = torch.cat((x, feat), dim=1)
        x = self.fc(x)
        return x

class MMGAT(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MMGAT, self).__init__(config, dataset)

        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.n_layers = config['n_layers']
        self.n_ui_layers = config['n_ui_layers']
        self.reg_weight = config['reg_weight']
        self.mm_image_weight = config['mm_image_weight']
        self.dropout = config['dropout']

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_normal_(self.user_embedding.weight)
        nn.init.xavier_normal_(self.item_id_embedding.weight)

        self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
        self.image_trs = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)
        nn.init.xavier_normal_(self.image_trs.weight)

        self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
        self.text_trs = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)
        nn.init.xavier_normal_(self.text_trs.weight)

        self.mm_fusion = MultiModalFusion(self.embedding_dim, self.feat_embed_dim)

        self.graph_attention = GraphAttentionLayer(self.embedding_dim, self.embedding_dim, self.dropout, 0.2)

        self.fc_layers = nn.ModuleList([nn.Linear(self.embedding_dim, self.embedding_dim) for _ in range(self.n_ui_layers)])

        # Initialize the inter_matrix attribute
        self.inter_matrix = dataset.inter_matrix

        # Initialize the norm_adj matrix
        self.norm_adj = self._init_norm_adj()

    def _init_norm_adj(self):
        adj = self._build_adj()
        norm_adj = self._normalize_adj(adj)
        return norm_adj

    def _build_adj(self):
        adj = torch.zeros(self.n_users + self.n_items, self.n_users + self.n_items)
        for user in range(self.n_users):
            for item in range(self.n_items):
                if self.inter_matrix[user, item] > 0:
                    adj[user, item + self.n_users] = 1
                    adj[item + self.n_users, user] = 1
        return adj

    def _normalize_adj(self, adj):
        D = torch.sum(adj, dim=1)
        D_inv = 1 / D
        D_inv[D_inv == float('inf')] = 0
        norm_adj = torch.diag(D_inv) @ adj
        return norm_adj

    def forward(self):
        u_embeddings = self.user_embedding.weight
        i_embeddings = self.item_id_embedding.weight

        image_feats = self.image_trs(self.image_embedding.weight)
        text_feats = self.text_trs(self.text_embedding.weight)

        i_embeddings = self.mm_fusion(i_embeddings, image_feats + text_feats)

        ego_embeddings = torch.cat((u_embeddings, i_embeddings), dim=0)
        all_embeddings = [ego_embeddings]
        for i in range(self.n_ui_layers):
            ego_embeddings = self.graph_attention(ego_embeddings, self.norm_adj)
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)

        return u_g_embeddings, i_g_embeddings

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        u_g_embeddings, i_g_embeddings = self.forward()

        u_embeddings = u_g_embeddings[users]
        pos_i_embeddings = i_g_embeddings[pos_items]
        neg_i_embeddings = i_g_embeddings[neg_items]

        batch_mf_loss = BPRLoss()(u_embeddings, pos_i_embeddings, neg_i_embeddings)

        reg_embedding_loss = EmbLoss()(u_g_embeddings, i_g_embeddings)

        return batch_mf_loss + self.reg_weight * reg_embedding_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]

        restore_user_e, restore_item_e = self.forward()
        u_embeddings = restore_user_e[user]
        scores = torch.matmul(u_embeddings, restore_item_e.T)

        return scores
