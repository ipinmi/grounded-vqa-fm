import torch
import torch.nn as nn


class CrossModalFusion(nn.Module):
    def __init__(
        self,
        image_dim: int,
        text_dim: int,
        hidden_dim: int,
        num_heads: int,
        drop_out: float = 0.5,
    ):
        super(CrossModalFusion, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.drop_out = drop_out
        self.num_heads = num_heads

        # Input Projection layers for image and text to hidden dimension for cross-modal fusion

        self.image_projection = nn.Sequential(
            nn.Linear(image_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.drop_out),
        ).to(self.device)

        self.text_projection = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.drop_out),
        ).to(self.device)

        # Multi-head attention layer
        self.multi_head_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=self.num_heads,
            dropout=self.drop_out,
            batch_first=True,
        ).to(self.device)

        # Output projection layer
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.drop_out),
        ).to(self.device)

    def forward(self, image_features, text_features):
        """
        Perform cross-modal fusion using multi-head attention mechanism,
        projecting the text features into the visual feature space.

        Args:
            image_features: CLIP Image features (batch_size, image_dim)
            text_features: CLIP Text features (batch_size, text_dim)

        Returns:
            fused_features: Cross-modal fused features (batch_size, hidden_dim)
        """
        # Project image and text features to hidden dimension
        projected_image = self.image_projection(image_features)  # from 512 to 256
        projected_text = self.text_projection(text_features)  # from 512 to 256

        # Reshape the projected image and text features for multi-head attention
        image_seq = projected_image.unsqueeze(1)  # (batch_size, 1, hidden_dim)
        text_seq = projected_text.unsqueeze(1)  # (batch_size, 1, hidden_dim)

        # Apply multi-head cross attention
        crossed_features, _ = self.multi_head_attention(
            query=image_seq,
            key=text_seq,
            value=text_seq,
        )  # (batch_size, 1, hidden_dim)

        # Reshape the fused features to remove the extra dimension
        crossed_features = crossed_features.squeeze(1)  # (batch_size, hidden_dim)

        # Combine the original visual features (projected_image) with the text-aware visual features (crossed_features)
        fused_features = torch.cat(
            [projected_image, crossed_features], dim=1
        )  # Shape: (batch_size, hidden_dim * 2)

        # Project the fused features to hidden dimension
        fused_features = self.output_projection(fused_features)

        return fused_features
