# coding: utf-8
import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from common.abstract_recommender import GeneralRecommender
from torch_geometric.utils import remove_self_loops, add_self_loops, degree, softmax
from torch_scatter import scatter_add
# 0.0260
class ModalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
    def forward(self, x):
        return F.normalize(self.encoder(x), p=2, dim=1)

class GraphAttentionLayer(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.gat = GATConv(dim, dim // 4, heads=4, dropout=dropout)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index):
        return self.norm(x + self.dropout(self.gat(x, edge_index)))

class EnhancedModalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        # Add residual connection
        self.residual = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        
    def forward(self, x):
        h = self.encoder(x)
        x = self.residual(x)
        return F.normalize(h + x, p=2, dim=1)

class EnhancedGATLayer(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.2):
        super().__init__()
        self.gat = GATConv(dim, dim // heads, heads=heads, dropout=dropout, add_self_loops=True)
        self.efficient_attn = MemoryEfficientAttention(dim, num_heads=heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim)
        )
        
    def forward(self, x, edge_index, edge_weight=None):
        # Convert sparse inputs to dense if needed
        if isinstance(x, torch.sparse.Tensor):
            x = x.to_dense()
        
        # Graph attention
        h = self.gat(x, edge_index, edge_weight)
        h = self.dropout(h)
        h = self.norm1(h)
        
        # Memory-efficient self attention
        h = h + self.efficient_attn(h.unsqueeze(0)).squeeze(0)
        h = self.norm1(h)
        
        # Feed-forward network
        h2 = self.ffn(h)
        h = h + h2  # Residual connection
        h = self.norm2(h)  # Final layer norm
        return h

class ModalFusionLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, 2),
            nn.Softmax(dim=1)
        )
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
        self.fusion_layer = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.LayerNorm(dim),
            nn.Dropout(0.2)
        )
        
    def forward(self, img_emb, txt_emb):
        # Compute attention weights
        combined = torch.cat([img_emb, txt_emb], dim=1)
        weights = self.attention(combined)
        
        # Weighted combination
        fused = weights[:, 0:1] * img_emb + weights[:, 1:2] * txt_emb
        
        # Additional fusion transformation
        concat_features = torch.cat([img_emb, txt_emb], dim=1)
        fused = fused + self.fusion_layer(concat_features)
        return fused

class MemoryEfficientAttention(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.2):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        B, N, C = x.shape
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Chunked attention to save memory
        chunk_size = 128
        attn_chunks = []
        for i in range(0, N, chunk_size):
            end_idx = min(i + chunk_size, N)
            Q = q[:, :, i:end_idx]
            attn_weights = (Q @ k.transpose(-2, -1)) * self.scale
            
            if mask is not None:
                attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
            
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = self.dropout(attn_weights)
            attn_chunk = attn_weights @ v
            attn_chunks.append(attn_chunk)
            
        attn_output = torch.cat(attn_chunks, dim=2)
        attn_output = attn_output.transpose(1, 2).reshape(B, N, C)
        return self.out_proj(attn_output)

class MMGAT(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MMGAT, self).__init__(config, dataset)
        
        self.embedding_dim = config["embedding_size"]
        self.feat_embed_dim = config["feat_embed_dim"]
        self.hidden_dim = self.feat_embed_dim * 2
        self.n_layers = config["n_mm_layers"]
        self.dropout = config["dropout"]
        self.reg_weight = config["reg_weight"]
        self.knn_k = config["knn_k"]
        
        # Add new parameters
        self.n_ui_layers = config.get("n_ui_layers", 2)  # Number of user-item graph layers
        self.lambda_coeff = config.get("lambda_coeff", 0.5) # Weight for modal loss
        
        # User-Item embeddings with better initialization
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight, gain=1.0)
        nn.init.xavier_uniform_(self.item_embedding.weight, gain=1.0)

        # Improved modal encoders with residual connections
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_encoder = EnhancedModalEncoder(self.v_feat.shape[1], 
                                                    self.hidden_dim,
                                                    self.feat_embed_dim, 
                                                    self.dropout)
            
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_encoder = EnhancedModalEncoder(self.t_feat.shape[1],
                                                   self.hidden_dim,
                                                   self.feat_embed_dim,
                                                   self.dropout)

        # Enhanced GAT layers with multi-head attention
        self.mm_layers = nn.ModuleList([
            EnhancedGATLayer(self.feat_embed_dim, heads=4, dropout=self.dropout)
            for _ in range(self.n_layers)
        ])
        
        self.ui_layers = nn.ModuleList([
            EnhancedGATLayer(self.embedding_dim, heads=4, dropout=self.dropout)
            for _ in range(self.n_ui_layers)
        ])

        # Advanced modal fusion with gating mechanism
        if self.v_feat is not None and self.t_feat is not None:
            self.fusion = ModalFusionLayer(self.feat_embed_dim)
        
        # Load and process interaction data
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.edge_index, self.edge_weight = self.build_edges()
        self.mm_edge_index = None
        self.build_modal_graph()
        
        # Enable gradient checkpointing for memory efficiency
        for layer in self.mm_layers:
            layer.gat.gradient_checkpointing = True
        for layer in self.ui_layers:
            layer.gat.gradient_checkpointing = True
        
        self.to(self.device)

    def get_norm_adj_mat(self):
        """Get normalized adjacency matrix in a way that's compatible with CUDA sparse operations"""
        # Create adjacency matrix
        A = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        A._update(data_dict)

        # Normalize adjacency matrix
        sumArr = (A > 0).sum(axis=1)
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        
        # Convert to COO format
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        
        # Convert to torch tensor
        indices = torch.LongTensor([row, col])
        values = torch.FloatTensor(L.data)
        
        # Use sparse_coo_tensor instead of SparseTensor
        return torch.sparse_coo_tensor(
            indices, 
            values,
            torch.Size((self.n_nodes, self.n_nodes)),
            device=self.device
        )

    def build_edges(self):
        """Build edge indices and weights in a CUDA-compatible way"""
        rows = self.interaction_matrix.row
        cols = self.interaction_matrix.col + self.n_users
        
        edge_index = torch.tensor(np.vstack([
            np.concatenate([rows, cols]),
            np.concatenate([cols, rows])
        ]), dtype=torch.long).to(self.device)
        
        # Add self-loops and normalize
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=self.n_users + self.n_items)
        
        # Calculate edge weights using dense operations
        row, col = edge_index
        deg = degree(row, self.n_users + self.n_items, dtype=torch.float)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        return edge_index, edge_weight

    def build_modal_graph(self):
        """Build modal graph with dense operations"""
        if self.v_feat is None and self.t_feat is None:
            return
            
        # Build modal graph with dense similarity computation
        if self.v_feat is not None:
            v_feats = F.normalize(self.v_feat.to_dense() if self.v_feat.is_sparse else self.v_feat, p=2, dim=1)
            v_sim = torch.mm(v_feats, v_feats.t())
            v_values, v_indices = v_sim.topk(k=self.knn_k, dim=1)
            del v_sim
            
        if self.t_feat is not None:
            t_feats = F.normalize(self.t_feat.to_dense() if self.t_feat.is_sparse else self.t_feat, p=2, dim=1)
            t_sim = torch.mm(t_feats, t_feats.t())
            t_values, t_indices = t_sim.topk(k=self.knn_k, dim=1)
            del t_sim
        
        # Combine modalities with adaptive weighting
        if self.v_feat is not None and self.t_feat is not None:
            # Adaptive weighting based on feature quality
            v_weight = torch.sigmoid(torch.mean(v_values))
            t_weight = torch.sigmoid(torch.mean(t_values))
            total = v_weight + t_weight
            v_weight = v_weight / total
            t_weight = t_weight / total
            
            indices = torch.round(v_weight * v_indices.float() + t_weight * t_indices.float()).long()
        else:
            indices = v_indices if self.v_feat is not None else t_indices
            
        rows = torch.arange(indices.size(0), device=self.device).view(-1, 1).expand_as(indices)
        
        edge_index = torch.stack([
            torch.cat([rows.reshape(-1), indices.reshape(-1)]),
            torch.cat([indices.reshape(-1), rows.reshape(-1)])
        ]).to(self.device)
        
        # Calculate edge weights for modal graph
        row, col = edge_index
        deg = degree(row, indices.size(0), dtype=torch.float)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        self.mm_edge_index = edge_index
        self.mm_edge_weight = edge_weight

    def forward(self):
        # Enhanced forward pass with residual connections and layer normalization
        img_emb = txt_emb = None
        
        if self.v_feat is not None:
            img_emb = self.image_encoder(self.image_embedding.weight)
            
        if self.t_feat is not None:
            txt_emb = self.text_encoder(self.text_embedding.weight)
        
        # Modal fusion with attention
        if img_emb is not None and txt_emb is not None:
            modal_emb = self.fusion(img_emb, txt_emb)
        else:
            modal_emb = img_emb if img_emb is not None else txt_emb
            
        # Process modalities through GAT layers
        if modal_emb is not None:
            h = modal_emb
            for layer in self.mm_layers:
                h_new = layer(h, self.mm_edge_index, self.mm_edge_weight)
                h = h + h_new  # Residual connection
                h = F.layer_norm(h, h.size()[1:])
            modal_emb = h
            
        # Process user-item graph
        x = torch.cat([self.user_embedding.weight, self.item_embedding.weight])
        all_embs = [x]
        
        for layer in self.ui_layers:
            if self.training:
                x = F.dropout(x, p=self.dropout)
            x_new = layer(x, self.edge_index, self.edge_weight)
            x = x + x_new  # Residual connection
            x = F.layer_norm(x, x.size()[1:])
            all_embs.append(x)
            
        x = torch.stack(all_embs, dim=1).mean(dim=1)
        user_emb, item_emb = torch.split(x, [self.n_users, self.n_items])
        
        # Combine with modal embeddings using gating
        if modal_emb is not None:
            gate = torch.sigmoid(self.fusion.gate(item_emb, modal_emb))
            item_emb = gate * item_emb + (1 - gate) * modal_emb
            
        return user_emb, item_emb, img_emb, txt_emb

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        
        user_emb, item_emb, img_emb, txt_emb = self.forward()
        
        u_emb = user_emb[users]
        pos_emb = item_emb[pos_items]
        neg_emb = item_emb[neg_items]
        
        # BPR loss with temperature scaling
        temperature = 0.2
        pos_scores = torch.sum(u_emb * pos_emb, dim=1) / temperature
        neg_scores = torch.sum(u_emb * neg_emb, dim=1) / temperature
        mf_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        
        # Enhanced modal contrastive loss
        modal_loss = 0.0
        if img_emb is not None and txt_emb is not None:
            # Intra-modal contrastive loss
            pos_sim_img = F.cosine_similarity(img_emb[pos_items], img_emb[pos_items])
            neg_sim_img = F.cosine_similarity(img_emb[pos_items], img_emb[neg_items])
            img_loss = -torch.mean(F.logsigmoid(pos_sim_img - neg_sim_img))
            
            pos_sim_txt = F.cosine_similarity(txt_emb[pos_items], txt_emb[pos_items])
            neg_sim_txt = F.cosine_similarity(txt_emb[pos_items], txt_emb[neg_items])
            txt_loss = -torch.mean(F.logsigmoid(pos_sim_txt - neg_sim_txt))
            
            # Cross-modal contrastive loss
            pos_sim_cross = F.cosine_similarity(img_emb[pos_items], txt_emb[pos_items])
            neg_sim_cross = F.cosine_similarity(img_emb[pos_items], txt_emb[neg_items])
            cross_loss = -torch.mean(F.logsigmoid(pos_sim_cross - neg_sim_cross))
            
            modal_loss = img_loss + txt_loss + cross_loss
        
        # L2 regularization with weight decay
        reg_loss = self.reg_weight * (
            torch.norm(u_emb) +
            torch.norm(pos_emb) +
            torch.norm(neg_emb) +
            (torch.norm(img_emb) + torch.norm(txt_emb) if img_emb is not None and txt_emb is not None else 0)
        )
        
        # Adaptive loss weighting
        modal_weight = self.lambda_coeff * torch.sigmoid(modal_loss.detach())
        
        # Apply warm-up scaling to losses during early training
        if hasattr(self, 'epoch') and self.epoch < 5:
            warmup_factor = min(1.0, self.epoch / 5)
            modal_weight = modal_weight * warmup_factor
            reg_loss = reg_loss * warmup_factor
        
        total_loss = mf_loss + modal_weight * modal_loss + reg_loss
        return total_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_emb, item_emb, _, _ = self.forward()
        scores = torch.matmul(user_emb[user], item_emb.transpose(0, 1))
        return scores