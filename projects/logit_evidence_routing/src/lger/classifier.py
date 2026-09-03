"""A shared lightweight classifier for every patch selector."""

from __future__ import annotations

import torch
from torch import nn


class EvidenceClassifier(nn.Module):
    """Project selected states, add spatial/layer identity, and classify via CLS."""

    def __init__(
        self,
        hidden_dim: int,
        num_classes: int,
        *,
        model_dim: int = 512,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_layer_id: int = 128,
    ) -> None:
        super().__init__()
        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        self.projection = nn.Linear(hidden_dim, model_dim)
        self.coordinate_encoder = nn.Sequential(
            nn.Linear(2, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )
        self.layer_embedding = nn.Embedding(max_layer_id + 1, model_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, model_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=4 * model_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(model_dim)
        self.head = nn.Linear(model_dim, num_classes)
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(
        self,
        hidden_states: torch.Tensor,
        patch_coords: torch.Tensor,
        layer_ids: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        unbatched = hidden_states.ndim == 2
        if unbatched:
            hidden_states = hidden_states.unsqueeze(0)
            patch_coords = patch_coords.unsqueeze(0)
            layer_ids = layer_ids.unsqueeze(0)
            if key_padding_mask is not None:
                key_padding_mask = key_padding_mask.unsqueeze(0)
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must be [batch, patches, hidden]")
        if patch_coords.shape != (*hidden_states.shape[:2], 2):
            raise ValueError("patch_coords must align with batch and patches")
        if layer_ids.shape != hidden_states.shape[:2]:
            raise ValueError("layer_ids must align with batch and patches")
        if layer_ids.min() < 0 or layer_ids.max() >= self.layer_embedding.num_embeddings:
            raise ValueError("layer_ids exceed the configured embedding range")

        model_dtype = self.projection.weight.dtype
        model_device = self.projection.weight.device
        hidden_states = hidden_states.to(device=model_device, dtype=model_dtype)
        patch_coords = patch_coords.to(device=model_device, dtype=model_dtype)
        layer_ids = layer_ids.to(model_device)
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.to(model_device)
        normalized_coords = self._normalize_coords(patch_coords)
        tokens = (
            self.projection(hidden_states)
            + self.coordinate_encoder(normalized_coords)
            + self.layer_embedding(layer_ids.long())
        )
        cls = self.cls_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat((cls, tokens), dim=1)
        if key_padding_mask is not None:
            cls_mask = torch.zeros(
                key_padding_mask.shape[0],
                1,
                dtype=torch.bool,
                device=key_padding_mask.device,
            )
            key_padding_mask = torch.cat((cls_mask, key_padding_mask.bool()), dim=1)
        encoded = self.encoder(tokens, src_key_padding_mask=key_padding_mask)
        output = self.head(self.norm(encoded[:, 0]))
        return output[0] if unbatched else output

    @staticmethod
    def _normalize_coords(coords: torch.Tensor) -> torch.Tensor:
        minimum = coords.amin(dim=1, keepdim=True)
        span = (coords.amax(dim=1, keepdim=True) - minimum).clamp_min(1.0)
        return 2.0 * (coords - minimum) / span - 1.0
