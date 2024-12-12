# coding: utf-8
import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from common.abstract_recommender import GeneralRecommender

class ModalProjection(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim)
        )
        
    def forward(self, x):
        return self.net(x)

class ModalFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, 4, dropout=0.1, batch_first=True)
        self.fusion = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.LayerNorm(dim)
        )
        
    def forward(self, x1, x2):
        h1, _ = self.attention(x1.unsqueeze(1), x2.unsqueeze(1), x2.unsqueeze(1))
        h2, _ = self.attention(x2.unsqueeze(1), x1.unsqueeze(1), x1.unsqueeze(1))
        h = torch.cat([h1.squeeze(1), h2.squeeze(1)], dim=1)
        return self.fusion(h)

class MMGAT(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MMGAT, self).__init__(config, dataset)
        
        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.n_layers = config['n_mm_layers']
        self.reg_weight = config['reg_weight']
        
        # User-Item embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)
        
        # Modal towers
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_proj = ModalProjection(self.v_feat.shape[1], self.feat_embed_dim)
            
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_proj = ModalProjection(self.t_feat.shape[1], self.feat_embed_dim)
        
        # Modal fusion
        self.modal_fusion = ModalFusion(self.feat_embed_dim)
        
        # Load interaction data
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.norm_adj = self.build_graph()
        self.to(self.device)

    def build_graph(self):
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.interaction_matrix.tolil()
        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()
        
        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum + 1e-7, -0.5).flatten()
        d_mat = sp.diags(d_inv)
        norm_adj = d_mat.dot(adj_mat).dot(d_mat)
        
        norm_adj = norm_adj.tocoo()
        indices = np.vstack((norm_adj.row, norm_adj.col))
        i = torch.LongTensor(indices).to(self.device)
        v = torch.FloatTensor(norm_adj.data).to(self.device)
        return torch.sparse_coo_tensor(i, v, norm_adj.shape)

    def message_passing(self, emb):
        embeds = [F.normalize(emb)]
        for _ in range(self.n_layers):
            emb = F.normalize(torch.sparse.mm(self.norm_adj, emb))
            embeds.append(emb)
        return torch.stack(embeds, dim=1).mean(dim=1)

    def forward(self):
        # Process modalities
        image_feat, text_feat = None, None
        if self.v_feat is not None:
            image_feat = self.image_proj(self.image_embedding.weight)
        if self.t_feat is not None:
            text_feat = self.text_proj(self.text_embedding.weight)
            
        # Fuse modalities
        if image_feat is not None and text_feat is not None:
            modal_feat = self.modal_fusion(image_feat, text_feat)
        else:
            modal_feat = image_feat if image_feat is not None else text_feat
            
        # Graph message passing
        ego_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        all_embeddings = self.message_passing(ego_embeddings)
        
        user_embeddings, item_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items])
        item_embeddings = item_embeddings + modal_feat
        
        return user_embeddings, item_embeddings, image_feat, text_feat
    
    def masked_softmax(self, logits):
        max_val = torch.max(logits, dim=1, keepdim=True)[0]
        exp = torch.exp(logits - max_val)
        return exp / torch.sum(exp, dim=1, keepdim=True)

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        
        user_emb, item_emb, img_feat, txt_feat = self.forward()
        
        u_e = user_emb[users]
        pos_e = item_emb[pos_items]
        neg_e = item_emb[neg_items]
        
        # BPR loss
        pos_scores = torch.sum(u_e * pos_e, dim=1)
        neg_scores = torch.sum(u_e * neg_e, dim=1)
        bpr_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        
        # Contrastive loss for modality alignment
        contrastive_loss = 0.0
        if img_feat is not None:
            contrastive_loss += torch.mean(
                1 - F.cosine_similarity(u_e, img_feat[pos_items]) +
                F.cosine_similarity(u_e, img_feat[neg_items])
            )
            
        if txt_feat is not None:
            contrastive_loss += torch.mean(
                1 - F.cosine_similarity(u_e, txt_feat[pos_items]) +
                F.cosine_similarity(u_e, txt_feat[neg_items])
            )
        
        # Regularization
        reg_loss = self.reg_weight * (
            torch.norm(u_e) + 
            torch.norm(pos_e) + 
            torch.norm(neg_e)
        )
        
        # Total loss with adaptive weighting
        total_loss = bpr_loss + 0.5 * contrastive_loss + reg_loss
        return total_loss



    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_emb, item_emb, _, _ = self.forward()
        scores = torch.matmul(user_emb[user], item_emb.transpose(0, 1))
        return scores