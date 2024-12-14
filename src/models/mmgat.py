import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from common.abstract_recommender import GeneralRecommender

class MMGAT(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MMGAT, self).__init__(config, dataset)
        
        # Basic configuration
        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.dropout = config['dropout']
        self.reg_weight = config['reg_weight']
        self.temp = config['temperature']
        
        # Number of nodes
        self.n_nodes = self.n_users + self.n_items
        
        # Base embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)
        
        # Modal-specific components
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_encoder = ModalEncoder(
                self.v_feat.shape[1], 
                self.feat_embed_dim,
                self.n_heads
            )
            
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_encoder = ModalEncoder(
                self.t_feat.shape[1], 
                self.feat_embed_dim,
                self.n_heads
            )
        
        # Dynamic modal fusion
        self.modal_attention = ModalAttention(self.feat_embed_dim, self.n_heads)
        
        # Graph attention layers for each modality
        self.gat_layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.gat_layers.append(
                GATConv(self.embedding_dim, self.embedding_dim, self.n_heads)
            )
            
        # Modal-specific user preference
        self.user_modal_preference = nn.Parameter(
            torch.zeros(self.n_users, 2)  # For two modalities
        )
        
        # Contrastive learning projection
        self.projector = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, self.embedding_dim)
        )

    def encode_modalities(self, users):
        modal_embeddings = []
        
        # Encode visual features if available
        if self.v_feat is not None:
            visual_emb = self.image_encoder(self.image_embedding.weight)
            modal_embeddings.append(visual_emb)
            
        # Encode textual features if available
        if self.t_feat is not None:
            text_emb = self.text_encoder(self.text_embedding.weight)
            modal_embeddings.append(text_emb)
            
        # Dynamic fusion of modalities
        if len(modal_embeddings) > 1:
            # Get user preferences for modalities
            user_prefs = F.softmax(self.user_modal_preference[users], dim=-1)
            fused_emb = self.modal_attention(
                modal_embeddings[0],
                modal_embeddings[1],
                user_prefs
            )
        else:
            fused_emb = modal_embeddings[0]
            
        return fused_emb, modal_embeddings

    def forward(self, users, items):
        # Get base embeddings
        user_emb = self.user_embedding(users)
        item_emb = self.item_id_embedding(items)
        
        # Encode and fuse modalities
        fused_modal_emb, modal_embs = self.encode_modalities(users)
        
        # Apply graph attention layers
        x = torch.cat([user_emb, item_emb], dim=0)
        for gat_layer in self.gat_layers:
            x = gat_layer(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Split back user and item embeddings
        user_final_emb, item_final_emb = torch.split(x, [user_emb.size(0), item_emb.size(0)])
        
        # Combine with modal information
        item_final_emb = item_final_emb + fused_modal_emb
        
        return user_final_emb, item_final_emb, modal_embs

    def calc_loss(self, user_emb, item_emb, modal_embs):
        # Basic recommendation loss
        pred_ratings = torch.sum(user_emb * item_emb, dim=1)
        rec_loss = F.binary_cross_entropy_with_logits(pred_ratings, labels)
        
        # Contrastive loss for modalities
        if len(modal_embs) > 1:
            proj_modal1 = self.projector(modal_embs[0])
            proj_modal2 = self.projector(modal_embs[1])
            
            sim_matrix = torch.mm(proj_modal1, proj_modal2.t()) / self.temp
            labels = torch.arange(sim_matrix.size(0)).to(self.device)
            contra_loss = F.cross_entropy(sim_matrix, labels)
        else:
            contra_loss = 0
            
        # Regularization
        reg_loss = self.reg_weight * (
            torch.norm(user_emb) +
            torch.norm(item_emb) +
            torch.norm(self.user_modal_preference)
        )
        
        return rec_loss + contra_loss + reg_loss

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        
        user_emb, pos_emb, modal_embs = self.forward(users, pos_items)
        _, neg_emb, _ = self.forward(users, neg_items)
        
        pos_scores = torch.sum(user_emb * pos_emb, dim=1)
        neg_scores = torch.sum(user_emb * neg_emb, dim=1)
        
        loss = self.calc_loss(user_emb, pos_emb, modal_embs)
        loss += -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores)))
        
        return loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_emb = self.user_embedding(user)
        
        all_items = torch.arange(self.n_items).to(self.device)
        item_emb = self.item_id_embedding(all_items)
        
        fused_modal_emb, _ = self.encode_modalities(user)
        item_emb = item_emb + fused_modal_emb
        
        scores = torch.matmul(user_emb, item_emb.t())
        return scores

class ModalEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, n_heads):
        super(ModalEncoder, self).__init__()
        self.transform = nn.Sequential(
            nn.Linear(input_dim, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.ReLU(),
            nn.Linear(output_dim * 2, output_dim)
        )
        self.attention = nn.MultiheadAttention(output_dim, n_heads)
        
    def forward(self, x):
        # Transform features
        h = self.transform(x)
        
        # Self-attention
        h = h.unsqueeze(0)  # Add sequence dimension
        h, _ = self.attention(h, h, h)
        h = h.squeeze(0)  # Remove sequence dimension
        
        return h

class ModalAttention(nn.Module):
    def __init__(self, embed_dim, n_heads):
        super(ModalAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, n_heads)
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU()
        )
        
    def forward(self, modal1, modal2, user_prefs):
        # Cross-modal attention
        modal1 = modal1.unsqueeze(0)
        modal2 = modal2.unsqueeze(0)
        
        attn_1_2, _ = self.attention(modal1, modal2, modal2)
        attn_2_1, _ = self.attention(modal2, modal1, modal1)
        
        # Remove sequence dimension
        attn_1_2 = attn_1_2.squeeze(0)
        attn_2_1 = attn_2_1.squeeze(0)
        
        # Weighted combination based on user preferences
        combined = torch.stack([attn_1_2, attn_2_1], dim=1)
        user_prefs = user_prefs.unsqueeze(-1)
        fused = torch.sum(combined * user_prefs, dim=1)
        
        return self.fusion(fused)