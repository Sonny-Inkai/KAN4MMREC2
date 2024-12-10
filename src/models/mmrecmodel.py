import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_
from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss

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

class MMRECMODEL(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MMRECMODEL, self).__init__(config, dataset)

        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.n_layers = config['n_layers']
        self.n_ui_layers = config['n_ui_layers']
        self.reg_weight = config['reg_weight']

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        xavier_normal_(self.user_embedding.weight)
        xavier_normal_(self.item_id_embedding.weight)

        self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
        self.image_trs = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)
        xavier_normal_(self.image_trs.weight)

        self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
        self.text_trs = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)
        xavier_normal_(self.text_trs.weight)

        self.mm_fusion = MultiModalFusion(self.embedding_dim, self.feat_embed_dim)

        self.fc_layers = nn.ModuleList([nn.Linear(self.embedding_dim, self.embedding_dim) for _ in range(self.n_ui_layers)])

    def forward(self):
        u_embeddings = self.user_embedding.weight
        i_embeddings = self.item_id_embedding.weight

        image_feats = self.image_trs(self.image_embedding.weight)
        text_feats = self.text_trs(self.text_embedding.weight)

        i_embeddings = self.mm_fusion(i_embeddings, image_feats + text_feats)

        ego_embeddings = torch.cat((u_embeddings, i_embeddings), dim=0)
        all_embeddings = [ego_embeddings]
        for i in range(self.n_ui_layers):
            ego_embeddings = F.relu(self.fc_layers[i](ego_embeddings))
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

        u_g_embeddings = u_g_embeddings[users]
        pos_i_g_embeddings = i_g_embeddings[pos_items]
        neg_i_g_embeddings = i_g_embeddings[neg_items]

        batch_mf_loss = BPRLoss()(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings)

        return batch_mf_loss + self.reg_weight * (u_g_embeddings.norm(2).pow(2) + pos_i_g_embeddings.norm(2).pow(2) + neg_i_g_embeddings.norm(2).pow(2))

    def full_sort_predict(self, interaction):
        user = interaction[0]

        restore_user_e, restore_item_e = self.forward()
        u_embeddings = restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
        return scores
