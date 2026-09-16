"""Auditable yes/no-restricted DoLa adapter for the shared CUB experiment."""

from __future__ import annotations

from typing import Any, Sequence


def restricted_binary_dola(
    layer_pair_logits: Sequence[Sequence[float]],
    *,
    candidate_layers: Sequence[int],
    plausibility_alpha: float,
) -> dict[str, Any]:
    """Run dynamic DoLa on two answer tokens, with explicit restricted scope."""

    import torch

    from lger.multimodal_dola import dynamic_layer_contrast

    values = torch.tensor(layer_pair_logits, dtype=torch.float32)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("layer_pair_logits must have shape [layers,2]")
    mature = values[-1:]
    valid = sorted({int(index) for index in candidate_layers})
    if not valid or any(index < 0 or index >= values.shape[0] - 1 for index in valid):
        raise ValueError("candidate layer is outside the premature layer range")
    candidates = values[valid][None]
    output = dynamic_layer_contrast(
        mature,
        candidates,
        plausibility_alpha=plausibility_alpha,
    )
    chosen_offset = int(output.premature_indices[0])
    scores = output.contrasted_logits[0]
    return {
        "scope": "yes_no_restricted_first_token_scoring_not_full_generation",
        "candidate_layers": valid,
        "selected_premature_layer": valid[chosen_offset],
        "mature_layer": int(values.shape[0] - 1),
        "positive_score": float(scores[0]),
        "negative_score": float(scores[1]),
        "raw_margin": float(scores[0] - scores[1]),
        "plausibility_alpha": plausibility_alpha,
        "js_divergences": output.js_divergences[0].tolist(),
        "plausibility_mask": output.plausibility_mask[0].tolist(),
    }

