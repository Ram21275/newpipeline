"""Model-facing utilities for visual-token extraction and frozen logit projection."""

from collections.abc import Sequence

import torch
from torch import nn


def gather_visual_hidden_states(
    hidden_states: Sequence[torch.Tensor],
    visual_token_indices: torch.Tensor,
    layer_ids: Sequence[int],
) -> torch.Tensor:
    """Gather only image-token positions as ``[layers, patches, hidden]``.

    Hugging Face models often include the embedding output at index zero. Callers
    must pass the exact model-output indices recorded in config. A batch dimension
    is accepted only when its size is one, keeping pilot alignment explicit.
    """

    if visual_token_indices.ndim != 1 or visual_token_indices.numel() == 0:
        raise ValueError("visual_token_indices must be a non-empty 1D tensor")
    if visual_token_indices.dtype not in (torch.int32, torch.int64):
        raise TypeError("visual_token_indices must be integer-valued")
    if int(visual_token_indices.min()) < 0:
        raise IndexError("visual token indices cannot be negative")
    if visual_token_indices.unique().numel() != visual_token_indices.numel():
        raise ValueError("visual_token_indices must be unique")
    if len(layer_ids) == 0:
        raise ValueError("layer_ids cannot be empty")

    gathered: list[torch.Tensor] = []
    for layer_id in layer_ids:
        state = hidden_states[layer_id]
        if state.ndim == 3:
            if state.shape[0] != 1:
                raise ValueError("pilot extraction currently expects batch size one")
            state = state[0]
        if state.ndim != 2:
            raise ValueError("each hidden state must have shape [tokens, hidden]")
        if int(visual_token_indices.max()) >= state.shape[0]:
            raise IndexError("a visual token index is outside the model sequence")
        gathered.append(
            state.index_select(0, visual_token_indices.to(state.device, dtype=torch.long))
        )
    return torch.stack(gathered, dim=0)


class FrozenLogitProjector(nn.Module):
    """Apply the frozen final normalization and LM head to patch states."""

    def __init__(self, final_norm: nn.Module, lm_head: nn.Module) -> None:
        super().__init__()
        self.final_norm = final_norm
        self.lm_head = lm_head
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True) -> "FrozenLogitProjector":
        """Keep frozen projection modules in evaluation mode."""

        super().train(False)
        return self

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape [layers, patches, hidden]")
        with torch.no_grad():
            return self.lm_head(self.final_norm(hidden_states))
