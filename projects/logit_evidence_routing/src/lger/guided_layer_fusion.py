"""Lightweight guided multi-head fusion over frozen multi-layer patch states."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn


@dataclass(frozen=True)
class GuidedFusionOutput:
    """Prediction and auditable attention weights from the fusion head."""

    logits: torch.Tensor
    patch_attention: tuple[torch.Tensor, ...]
    layer_attention: torch.Tensor
    fused_state: torch.Tensor


class GuidedLayerAttentionFusion(nn.Module):
    """Attend to patches within layers, then attend across layer summaries.

    The base VLM is intentionally absent from this module.  Callers pass frozen
    hidden states as one tensor per measurement stage, allowing architectures
    with different hidden widths and dynamic visual-token counts to share the
    same measurement concept without claiming identical internal layers.
    """

    def __init__(
        self,
        input_dims: Sequence[int],
        *,
        query_dim: int,
        embed_dim: int = 256,
        num_heads: int = 4,
        output_dim: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not input_dims or any(int(value) <= 0 for value in input_dims):
            raise ValueError("input_dims must be a non-empty sequence of positive widths")
        if query_dim <= 0 or embed_dim <= 0 or output_dim <= 0:
            raise ValueError("query, embedding, and output dimensions must be positive")
        if num_heads <= 0 or embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.input_dims = tuple(int(value) for value in input_dims)
        self.embed_dim = int(embed_dim)
        self.layer_projections = nn.ModuleList(
            nn.Linear(width, embed_dim) for width in self.input_dims
        )
        self.query_projection = nn.Linear(query_dim, embed_dim)
        self.patch_attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.layer_attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.normalization = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, output_dim)

    def forward(
        self,
        layer_states: Sequence[torch.Tensor],
        attribute_query: torch.Tensor,
        *,
        valid_patch_masks: Sequence[torch.Tensor] | None = None,
    ) -> GuidedFusionOutput:
        if len(layer_states) != len(self.input_dims):
            raise ValueError("one layer-state tensor is required for every configured input")
        if attribute_query.ndim != 2:
            raise ValueError("attribute_query must have shape [batch, query_dim]")
        batch = int(attribute_query.shape[0])
        if valid_patch_masks is None:
            valid_patch_masks = [
                torch.ones((batch, int(state.shape[1])), dtype=torch.bool, device=state.device)
                for state in layer_states
            ]
        if len(valid_patch_masks) != len(layer_states):
            raise ValueError("valid_patch_masks must align with layer_states")

        query = self.query_projection(attribute_query).unsqueeze(1)
        pooled_layers: list[torch.Tensor] = []
        patch_weights: list[torch.Tensor] = []
        for index, (state, mask, projection, width) in enumerate(zip(
            layer_states, valid_patch_masks, self.layer_projections, self.input_dims
        )):
            if state.ndim != 3 or int(state.shape[0]) != batch or int(state.shape[2]) != width:
                raise ValueError(
                    f"layer {index} must have shape [batch, patches, {width}]"
                )
            if mask.shape != state.shape[:2] or mask.dtype != torch.bool:
                raise ValueError(f"layer {index} mask must be boolean [batch, patches]")
            if not bool(mask.any(dim=1).all()):
                raise ValueError(f"layer {index} must retain at least one valid patch per example")
            projected = projection(state)
            pooled, weights = self.patch_attention(
                query, projected, projected,
                key_padding_mask=~mask.to(projected.device),
                need_weights=True,
                average_attn_weights=False,
            )
            pooled_layers.append(pooled.squeeze(1))
            patch_weights.append(weights.squeeze(2))
        layer_tokens = torch.stack(pooled_layers, dim=1)
        fused, layer_weights = self.layer_attention(
            query, layer_tokens, layer_tokens,
            need_weights=True,
            average_attn_weights=False,
        )
        fused_state = self.normalization(fused.squeeze(1))
        logits = self.classifier(fused_state)
        return GuidedFusionOutput(
            logits=logits,
            patch_attention=tuple(patch_weights),
            layer_attention=layer_weights.squeeze(2),
            fused_state=fused_state,
        )


class MeanLayerPoolingBaseline(nn.Module):
    """Simple projected mean-pooling baseline for the guided fusion head."""

    def __init__(self, input_dims: Sequence[int], *, embed_dim: int = 256, output_dim: int = 1):
        super().__init__()
        if not input_dims:
            raise ValueError("input_dims cannot be empty")
        self.input_dims = tuple(int(value) for value in input_dims)
        self.projections = nn.ModuleList(nn.Linear(width, embed_dim) for width in self.input_dims)
        self.classifier = nn.Linear(embed_dim, output_dim)

    def forward(
        self,
        layer_states: Sequence[torch.Tensor],
        *,
        valid_patch_masks: Sequence[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if len(layer_states) != len(self.input_dims):
            raise ValueError("one layer-state tensor is required for every configured input")
        if valid_patch_masks is None:
            valid_patch_masks = [
                torch.ones(state.shape[:2], dtype=torch.bool, device=state.device)
                for state in layer_states
            ]
        pooled = []
        for index, (state, mask, projection, width) in enumerate(zip(
            layer_states, valid_patch_masks, self.projections, self.input_dims
        )):
            if state.ndim != 3 or int(state.shape[2]) != width or mask.shape != state.shape[:2]:
                raise ValueError(f"invalid state or mask shape for layer {index}")
            weights = mask.to(state.device, dtype=state.dtype).unsqueeze(-1)
            denominator = weights.sum(dim=1).clamp_min(1.0)
            pooled.append((projection(state) * weights).sum(dim=1) / denominator)
        return self.classifier(torch.stack(pooled, dim=1).mean(dim=1))
