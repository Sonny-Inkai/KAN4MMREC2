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

    def encode_modalities(self, items):
        """
        Encode and fuse modal features for given items
        Args:
            items: Item indices tensor of shape (batch_size,)
        Returns:
            Tuple of (fused_embeddings, list of modal_embeddings)
        """
        modal_embeddings = []
        batch_size = items.size(0)
        
        # Encode visual features if available
        if self.v_feat is not None:
            visual_emb = self.image_encoder(self.image_embedding.weight[items])
            modal_embeddings.append(visual_emb)
            
        # Encode textual features if available
        if self.t_feat is not None:
            text_emb = self.text_encoder(self.text_embedding.weight[items])
            modal_embeddings.append(text_emb)
            
        # Dynamic fusion of modalities
        if len(modal_embeddings) > 1:
            fused_emb = self.modal_attention(
                modal_embeddings[0],
                modal_embeddings[1],
                batch_size
            )
        else:
            fused_emb = modal_embeddings[0]
            
        return fused_emb, modal_embeddings

    def forward(self, users, items):
        """
        Forward pass of the model
        Args:
            users: User indices tensor of shape (batch_size,)
            items: Item indices tensor of shape (batch_size,)
        """
        # Get base embeddings
        user_emb = self.user_embedding(users)
        item_emb = self.item_id_embedding(items)
        
        # Encode and fuse modalities for items
        fused_modal_emb, modal_embs = self.encode_modalities(items)
        
        # Combine item embeddings with modal information
        item_emb = item_emb + fused_modal_emb
        
        return user_emb, item_emb, modal_embs

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        
        user_emb, pos_emb, modal_embs = self.forward(users, pos_items)
        _, neg_emb, _ = self.forward(users, neg_items)
        
        # BPR loss
        pos_scores = torch.sum(user_emb * pos_emb, dim=1)
        neg_scores = torch.sum(user_emb * neg_emb, dim=1)
        bpr_loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores)))
        
        # Contrastive loss for modalities if multiple modalities exist
        contra_loss = 0
        if len(modal_embs) > 1:
            proj_modal1 = self.projector(modal_embs[0])
            proj_modal2 = self.projector(modal_embs[1])
            
            sim = torch.matmul(proj_modal1, proj_modal2.transpose(0, 1)) / self.temp
            labels = torch.arange(sim.size(0)).to(self.device)
            contra_loss = F.cross_entropy(sim, labels)
        
        # Regularization
        reg_loss = self.reg_weight * (
            torch.norm(user_emb) +
            torch.norm(pos_emb) +
            torch.norm(neg_emb)
        )
        
        return bpr_loss + contra_loss + reg_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_emb = self.user_embedding(user)
        
        # Get embeddings for all items
        all_items = torch.arange(self.n_items).to(self.device)
        item_emb = self.item_id_embedding(all_items)
        
        # Get modal embeddings for all items
        fused_modal_emb, _ = self.encode_modalities(all_items)
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
        
    def forward(self, x):
        return self.transform(x)

class ModalAttention(nn.Module):
    def __init__(self, embed_dim, n_heads):
        super(ModalAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, n_heads)
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU()
        )
        
    def forward(self, modal1, modal2, batch_size):
        """
        Args:
            modal1: First modality tensor of shape (batch_size, embed_dim)
            modal2: Second modality tensor of shape (batch_size, embed_dim)
            batch_size: Size of the batch
        """
        # Reshape for attention
        modal1 = modal1.unsqueeze(0)  # (1, batch_size, embed_dim)
        modal2 = modal2.unsqueeze(0)  # (1, batch_size, embed_dim)
        
        # Cross-modal attention
        attn_1_2, _ = self.attention(modal1, modal2, modal2)
        attn_2_1, _ = self.attention(modal2, modal1, modal1)
        
        # Remove sequence dimension
        attn_1_2 = attn_1_2.squeeze(0)  # (batch_size, embed_dim)
        attn_2_1 = attn_2_1.squeeze(0)  # (batch_size, embed_dim)
        
        # Concatenate and fuse
        concat_features = torch.cat([attn_1_2, attn_2_1], dim=-1)
        fused = self.fusion(concat_features)
        
        return fused