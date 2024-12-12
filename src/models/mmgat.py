import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np

from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss

class MMGAT(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MMGAT, self).__init__(config, dataset)
        
        # Basic configuration
        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.n_layers = config['n_layers']
        self.knn_k = config['knn_k']
        self.dropout = config['dropout']
        self.n_heads = config['n_heads']
        self.temperature = config['temperature']
        self.reg_weight = config['reg_weight']
        self.n_nodes = self.n_users + self.n_items
        
        # Load interaction matrix
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.norm_adj = self.get_normalized_adj().to(self.device)
        
        # Initialize embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        # Initialize modal-specific components
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_transform = nn.Sequential(
                nn.Linear(self.v_feat.shape[1], self.feat_embed_dim),
                nn.LayerNorm(self.feat_embed_dim),
                nn.ReLU()
            )
            self.image_attention = MultiHeadAttention(self.feat_embed_dim, self.n_heads)
            
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_transform = nn.Sequential(
                nn.Linear(self.t_feat.shape[1], self.feat_embed_dim),
                nn.LayerNorm(self.feat_embed_dim),
                nn.ReLU()
            )
            self.text_attention = MultiHeadAttention(self.feat_embed_dim, self.n_heads)

        # Cross-modal fusion
        self.modal_fusion = CrossModalFusion(self.feat_embed_dim)
        
        # Adaptive modal weighting
        self.modal_weight_net = nn.Sequential(
            nn.Linear(self.feat_embed_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=-1)
        )
        
        # Graph structure learning
        self.structure_learning = GraphStructureLearning(self.feat_embed_dim, self.knn_k)
        
        # Contrastive learning components
        self.momentum_encoder = MomentumEncoder(self.feat_embed_dim)
        self.queue = ModalityQueue(queue_size=config['queue_size'], feat_dim=self.feat_embed_dim)

    def get_normalized_adj(self):
        # Convert interaction matrix to adjacency matrix
        adj_mat = sp.dok_matrix((self.n_nodes, self.n_nodes), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.interaction_matrix.tolil()
        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()
        
        # Normalize adjacency matrix
        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj = d_mat_inv.dot(adj_mat).dot(d_mat_inv)
        
        # Convert to sparse tensor
        norm_adj = norm_adj.tocoo()
        indices = torch.LongTensor([norm_adj.row, norm_adj.col])
        values = torch.FloatTensor(norm_adj.data)
        shape = torch.Size(norm_adj.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def forward(self, users, items=None):
        # Process modalities
        modal_embeddings = []
        
        if self.v_feat is not None:
            image_feats = self.image_transform(self.image_embedding.weight)
            image_feats = self.image_attention(image_feats)
            modal_embeddings.append(image_feats)
            
        if self.t_feat is not None:
            text_feats = self.text_transform(self.text_embedding.weight)
            text_feats = self.text_attention(text_feats)
            modal_embeddings.append(text_feats)
        
        # Cross-modal fusion
        if len(modal_embeddings) > 1:
            modal_weights = self.modal_weight_net(torch.cat(modal_embeddings, dim=-1))
            fused_embeddings = self.modal_fusion(modal_embeddings, modal_weights)
        else:
            fused_embeddings = modal_embeddings[0]
        
        # Learn graph structure
        learned_adj = self.structure_learning(fused_embeddings)
        
        # Message passing on learned graph
        item_embeddings = self.item_id_embedding.weight
        for _ in range(self.n_layers):
            item_embeddings = torch.sparse.mm(learned_adj, item_embeddings)
        
        # User-item graph convolution
        ego_embeddings = torch.cat((self.user_embedding.weight, item_embeddings), dim=0)
        all_embeddings = [ego_embeddings]
        
        for _ in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.norm_adj, ego_embeddings)
            all_embeddings.append(ego_embeddings)
            
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        
        user_embeddings, item_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items])
        
        # Momentum update
        with torch.no_grad():
            self.momentum_encoder.update(fused_embeddings)
        
        if items is not None:
            user_embeddings = user_embeddings[users]
            item_embeddings = item_embeddings[items]
            
        return user_embeddings, item_embeddings, fused_embeddings

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        
        user_embeddings, item_embeddings, fused_embeddings = self.forward(users, torch.cat([pos_items, neg_items]))
        pos_embeddings, neg_embeddings = torch.split(item_embeddings, [pos_items.shape[0], neg_items.shape[0]])
        
        # BPR loss
        pos_scores = torch.sum(user_embeddings * pos_embeddings, dim=1)
        neg_scores = torch.sum(user_embeddings * neg_embeddings, dim=1)
        bpr_loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores)))
        
        # Contrastive loss
        momentum_embeddings = self.momentum_encoder(fused_embeddings)
        queue_embeddings = self.queue.get()
        
        pos_sim = torch.einsum('nc,nc->n', [fused_embeddings, momentum_embeddings])
        neg_sim = torch.einsum('nc,kc->nk', [fused_embeddings, queue_embeddings])
        
        logits = torch.cat([pos_sim.unsqueeze(-1), neg_sim], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=self.device)
        
        contrastive_loss = F.cross_entropy(logits / self.temperature, labels)
        
        # Update queue
        self.queue.update(momentum_embeddings.detach())
        
        # Regularization
        reg_loss = self.reg_weight * (
            torch.norm(user_embeddings) +
            torch.norm(pos_embeddings) +
            torch.norm(neg_embeddings)
        )
        
        return bpr_loss + contrastive_loss + reg_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_embeddings, item_embeddings, _ = self.forward(user)
        scores = torch.matmul(user_embeddings, item_embeddings.transpose(0, 1))
        return scores

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2), qkv)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        return self.proj(out)

class CrossModalFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, 1)
        )
        
    def forward(self, modal_embeddings, weights):
        fused = torch.zeros_like(modal_embeddings[0])
        for i, embedding in enumerate(modal_embeddings):
            fused += weights[:, i].unsqueeze(-1) * embedding
        return fused

class GraphStructureLearning(nn.Module):
    def __init__(self, dim, k):
        super().__init__()
        self.k = k
        self.temperature = nn.Parameter(torch.tensor(0.07))
        
    def forward(self, x):
        sim = torch.matmul(x, x.transpose(-2, -1))
        sim = sim / self.temperature
        
        # KNN graph
        _, idx = torch.topk(sim, self.k, dim=-1)
        mask = torch.zeros_like(sim).scatter_(-1, idx, 1)
        
        # Symmetrize
        mask = (mask + mask.transpose(-2, -1)) / 2
        
        # Normalize
        deg = mask.sum(dim=-1, keepdim=True).sqrt()
        adj = mask / torch.matmul(deg, deg.transpose(-2, -1))
        
        return adj

class MomentumEncoder(nn.Module):
    def __init__(self, dim, momentum=0.999):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        self.momentum = momentum
        
    def forward(self, x):
        return self.encoder(x)
        
    def update(self, x):
        for param, target_param in zip(x.parameters(), self.encoder.parameters()):
            target_param.data = target_param.data * self.momentum + param.data * (1 - self.momentum)

class ModalityQueue:
    def __init__(self, queue_size, feat_dim):
        self.queue_size = queue_size
        self.feat_dim = feat_dim
        self.queue = torch.randn(queue_size, feat_dim)
        self.queue = F.normalize(self.queue, dim=1)
        self.ptr = 0
        
    def get(self):
        return self.queue
        
    def update(self, embeddings):
        batch_size = embeddings.shape[0]
        assert self.feat_dim == embeddings.shape[1]
        
        ptr = int(self.ptr)
        if ptr + batch_size > self.queue_size:
            self.queue[ptr:] = embeddings[:self.queue_size - ptr].detach()
            self.queue[:batch_size - (self.queue_size - ptr)] = embeddings[-(batch_size - (self.queue_size - ptr)):].detach()
            self.ptr = batch_size - (self.queue_size - ptr)
        else:
            self.queue[ptr:ptr + batch_size] = embeddings.detach()
            self.ptr = (ptr + batch_size) % self.queue_size