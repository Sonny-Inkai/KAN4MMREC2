import torch
import torch.nn as nn
import torch.nn.functional as F
from common.abstract_recommender import GeneralRecommender
from utils.fasterkan import FasterKAN  # Advanced rational transformer-based architecture
from timm.layers import use_fused_attn
from torch_geometric.nn import GATConv

### Apply Fasterkan for all matrix [user, item] -> it is not work because it is too large for gpu in kaggle 

class KAN4MMREC(GeneralRecommender):
    def __init__(self, config, dataset):
        super(KAN4MMREC, self).__init__(config, dataset)

        # Load configuration and dataset parameters
        self.embedding_size = config['embedding_size']
        self.n_layers = config['n_layers']
        self.cl_weight = config['cl_weight']
        self.reg_weight = config['reg_weight']
        self.dropout = config['dropout']

        # User, item (ID-based), image, and text embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        nn.init.xavier_uniform_(self.user_embedding.weight)

        # If available, use pretrained image and text embeddings
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.embedding_size)
            nn.init.xavier_normal_(self.image_trs.weight)
        else:
            self.image_embedding = nn.Embedding(self.n_items, self.embedding_size)
            nn.init.xavier_uniform_(self.image_embedding.weight)

        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.embedding_size)
            nn.init.xavier_normal_(self.text_trs.weight)
        else:
            self.text_embedding = nn.Embedding(self.n_items, self.embedding_size)
            nn.init.xavier_uniform_(self.text_embedding.weight)

        self.layer_norm = nn.LayerNorm(self.embedding_size)

        # KANTransformer for image and text features
        self.kan_user = KANTransformer(self.embedding_size, self.n_layers, dropout=self.dropout) # For users 
        self.kan_image = KANTransformer(self.embedding_size, self.n_layers, dropout=self.dropout)  # For image interactions
        self.kan_text = KANTransformer(self.embedding_size, self.n_layers, dropout=self.dropout)   # For text interactions

        # Added GAT for Graph-based interaction modeling
        self.gat_conv_user = GATConv(self.embedding_size, self.embedding_size, heads=4, dropout=self.dropout)
        self.gat_conv_item = GATConv(self.embedding_size, self.embedding_size, heads=4, dropout=self.dropout)

        self.SplineLinear1 = SplineLinear(self.embedding_size, self.embedding_size)
        self.SplineLinear2 = SplineLinear(self.embedding_size, self.embedding_size)
        self.SplineLinear3 = SplineLinear(self.embedding_size, self.embedding_size)
        self.mlp_fusion = nn.Sequential(
            nn.Linear(self.embedding_size * 2, self.embedding_size * 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.embedding_size * 2, self.embedding_size)
        )

    def forward(self, edge_index_user=None, edge_index_item=None):
        # Transform embeddings
        image_embedding_transformed = self.image_trs(self.image_embedding.weight)
        text_embedding_transformed = self.text_trs(self.text_embedding.weight)

        # Pass through the rational KAN-based transformer layers
        u_transformed = self.kan_user(self.user_embedding.weight) # [num_users, emb_size]
        i_transformed = self.kan_image(image_embedding_transformed)  # [num_items, emb_size]
        t_transformed = self.kan_text(text_embedding_transformed)  # [num_items, emb_size]

        # Apply GAT for graph-based interaction (if edge_index is provided)
        if edge_index_user is not None:
            u_transformed = self.gat_conv_user(u_transformed, edge_index_user)
        if edge_index_item is not None:
            i_transformed = self.gat_conv_item(i_transformed, edge_index_item)
            t_transformed = self.gat_conv_item(t_transformed, edge_index_item)

        u_transformed = self.SplineLinear1(u_transformed)
        i_transformed = self.SplineLinear2(i_transformed)
        t_transformed = self.SplineLinear3(t_transformed)

        return u_transformed, i_transformed, t_transformed

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(users * pos_items, dim=1)
        neg_scores = torch.sum(users * neg_items, dim=1)

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        return mf_loss

    def calculate_loss(self, interaction):
        """
        Calculate the loss using SIGLIP-like approach for user-user , and loss including interaction labels for positive samples.

        Args:
            interaction: Tuple containing users and items (ground-truth interaction matrix [num_users, num_items]),
                         where users have interacted with the corresponding items.
        Returns:
            Total loss for training.
        """
        # Predict interaction scores for u_i and u_t
        u_transformed, i_transformed, t_transformed = self.forward(
            edge_index_user=interaction[3] if len(interaction) > 3 else None,
            edge_index_item=interaction[4] if len(interaction) > 4 else None
        )

        # Interaction-based loss component
        users = interaction[0]  # Corresponding items that users interacted with (positive items)
        pos_items = interaction[1] # Positive items
        neg_items = interaction[2]  # Negative sampled items

        mf_v_loss = self.bpr_loss(u_transformed[users], i_transformed[pos_items], i_transformed[neg_items])
        mf_t_loss = self.bpr_loss(u_transformed[users], t_transformed[pos_items], t_transformed[neg_items])

        # Predicting scores using interaction matrices
        u_i = torch.matmul(u_transformed, i_transformed.transpose(0, 1))
        u_t = torch.matmul(u_transformed, t_transformed.transpose(0, 1))
        
        # Fuse the scores using an MLP instead of simple averaging
        fusion_input = torch.cat([u_i, u_t], dim=-1)
        fusion_input = fusion_input.view(-1, self.embedding_size * 2)  # Flatten the input for MLP
        u_i_mat = self.mlp_fusion(fusion_input).view(u_i.shape)  # Reshape back to original dimensions

        u_i_pos = u_i_mat[torch.arange(len(users)), pos_items]
        u_i_neg = u_i_mat[torch.arange(len(users)), neg_items]

        maxi = F.logsigmoid(u_i_pos - u_i_neg)

        batch_loss = -torch.mean(maxi)
        reg_loss = self.reg_weight * (mf_t_loss + mf_v_loss)
        total_loss = batch_loss + reg_loss
        print(f"Batch loss: {batch_loss}")
        print(f"Reg loss: {reg_loss}")
        print(f"Total Loss: {total_loss}") 
        return total_loss

    def full_sort_predict(self, interaction):
        """
        Predict scores for all items for a given user by fusing image and text matrices.

        Args:
            interaction: User interaction data.

        Returns:
            score_mat_ui: Predicted scores for all items.
        """
        users = interaction[0]
        u_transformed, i_transformed, t_transformed = self.forward()

        u_i = torch.matmul(u_transformed, i_transformed.transpose(0, 1))
        u_t = torch.matmul(u_transformed, t_transformed.transpose(0, 1))
        
        # Fuse the scores using an MLP instead of simple averaging
        fusion_input = torch.cat([u_i, u_t], dim=-1)
        fusion_input = fusion_input.view(-1, self.embedding_size * 2)  # Flatten the input for MLP
        score_mat_ui = self.mlp_fusion(fusion_input).view(u_i.shape)  # Reshape back to original dimensions
        score = score_mat_ui[users]

        return score
    
class SplineLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, init_scale: float = 0.1, **kw) -> None:
        self.init_scale = init_scale
        super().__init__(in_features, out_features, bias=False, **kw)

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)  # Using Xavier Uniform initialization

class KANTransformer(nn.Module):
    """
    KANsiglip class that functions as a transformer-like module with KAN rational activation,
    similar to the image encoder in the SIGLIP model, but enhanced with advanced rational units.
    """
    def __init__(self, embedding_size, n_layers, dropout=0.5):
        super(KANTransformer, self).__init__()
        self.embedding_size = embedding_size
        self.n_layers = n_layers

        # Transformer-like attention mechanism combined with rational KAN
        self.layers = nn.ModuleList([
            KANLayer(embedding_size, dropout=dropout)
            for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor):
        """
        Forward pass for KANsiglip.

        Args:
            x: Input tensor of shape [seq_len, embedding_size].

        Returns:
            Transformed tensor of shape [seq_len, embedding_size].
        """
        for layer in self.layers:
            x = layer(x)
        return x

class KANLayerScale(nn.Module):
    def __init__(
            self,
            dim: int,
            init_values: float = 1e-5,
            inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

class KANLayer(nn.Module):
    def __init__(self, embedding_size, dropout=0.2):
        super(KANLayer, self).__init__()
        # Use the advanced Attention from katransformer.py
        self.norm1 = nn.LayerNorm(embedding_size)
        self.norm2 = nn.LayerNorm(embedding_size)

        # Use the FasterKAN rational MLP
        self.FasterKAN = FasterKAN(layers_hidden=[embedding_size, embedding_size])

        # Stochastic depth and dropout
        self.drop_path = nn.Identity() if dropout == 0 else nn.Dropout(dropout)
        self.layer_scale = KANLayerScale(dim=embedding_size, init_values=1e-5)  # Layer scaling to stabilize training

    def forward(self, hidden_states: torch.Tensor):
        # Apply attention
        residual = hidden_states
        hidden_states = self.drop_path(self.layer_scale(self.norm1(hidden_states)))

        # Apply FasterKAN
        hidden_states = self.FasterKAN(hidden_states)
        hidden_states = residual + self.layer_scale(hidden_states)
        hidden_states = self.norm2(hidden_states)
        return hidden_states
