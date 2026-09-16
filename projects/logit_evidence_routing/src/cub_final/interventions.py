"""Frozen-state replacement utilities with auditable selected positions."""

from __future__ import annotations

from typing import Any, Sequence


def replace_positions(
    states: Any,
    positions: Sequence[int],
    replacements: Any,
) -> Any:
    """Return a cloned [batch,tokens,hidden] tensor with exact replacements."""

    import torch

    if states.ndim != 3:
        raise ValueError("states must have shape [batch,tokens,hidden]")
    indices = torch.tensor(sorted({int(value) for value in positions}), device=states.device)
    if indices.numel() != len(positions) or indices.numel() == 0:
        raise ValueError("positions must be unique and non-empty")
    if int(indices.min()) < 0 or int(indices.max()) >= states.shape[1]:
        raise ValueError("replacement position is out of range")
    output = states.clone()
    replacement = replacements.to(device=states.device, dtype=states.dtype)
    if replacement.ndim == 1:
        replacement = replacement.expand(indices.numel(), -1)
    if replacement.shape != (indices.numel(), states.shape[2]):
        raise ValueError("replacements must be [hidden] or [selected,hidden]")
    output[:, indices, :] = replacement.unsqueeze(0).expand(states.shape[0], -1, -1)
    return output


def training_mean_replacement(training_states: Any) -> Any:
    if training_states.ndim not in (2, 3):
        raise ValueError("training states must be [examples,hidden] or [examples,tokens,hidden]")
    dimensions = tuple(range(training_states.ndim - 1))
    return training_states.float().mean(dim=dimensions).to(training_states.dtype)


def donor_replacement(
    donor_states: Any,
    donor_indices: Sequence[int],
) -> Any:
    """Select pre-matched donor states; this is not claimed perfectly on-manifold."""

    import torch

    if donor_states.ndim != 2:
        raise ValueError("donor_states must have shape [candidate,hidden]")
    indices = torch.tensor(list(donor_indices), dtype=torch.long, device=donor_states.device)
    if indices.numel() == 0 or int(indices.min()) < 0 or int(indices.max()) >= donor_states.shape[0]:
        raise ValueError("donor index is out of range")
    return donor_states.index_select(0, indices)


def qwen_multilevel_intervention_audit(config: Any, changed_paths: Sequence[str]) -> dict[str, Any]:
    """Make active Qwen3 visual injection paths explicit in every record."""

    indexes = getattr(config, "deepstack_visual_indexes", None)
    configured = list(indexes) if indexes is not None else []
    changed = sorted(set(str(value) for value in changed_paths))
    return {
        "deepstack_visual_indexes": configured,
        "changed_paths": changed,
        "all_visual_paths_removed": bool(configured) and len(changed) >= len(configured) + 1,
        "warning": (
            None
            if not configured or len(changed) >= len(configured) + 1
            else "Qwen3 multi-level visual paths remain active; this is a scoped interface intervention"
        ),
    }

