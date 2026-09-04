"""Shared selector definitions for the Phase 01B localization benchmark."""

from __future__ import annotations

from typing import Any

import torch

from .scoring import normalize_patch_scores


DETERMINISTIC_LOCALIZERS = (
    "llm_attention",
    "vision_cls_attention",
    "vision_attention_rollout",
    "logit_maxprob",
    "logit_margin",
    "logit_negentropy",
    "logit_concept",
    "attention_logit_fusion",
)
ALL_PROBE_SELECTORS = ("random", *DETERMINISTIC_LOCALIZERS, "global_all")


def feature_key(
    selector: str,
    k: int | None = None,
    selection_seed: int | None = None,
) -> str:
    if selector == "global_all":
        if k is not None or selection_seed is not None:
            raise ValueError("global_all does not use K or a selection seed")
        return selector
    if k is None or k <= 0:
        raise ValueError("Top-K selectors require a positive K")
    if selector == "random":
        if selection_seed is None:
            raise ValueError("random selection requires a selection seed")
        return f"random_k{k}_selection{selection_seed}"
    if selection_seed is not None:
        raise ValueError("deterministic selectors do not use a selection seed")
    return f"{selector}_k{k}"


def localizer_score_maps(evidence: Any) -> dict[str, torch.Tensor]:
    """Collect and fuse aligned patch scores from one extractor result."""

    required_evidence = {"maxprob", "margin", "negentropy", "concept_logprob"}
    missing_evidence = required_evidence - evidence.evidence_scores.keys()
    if missing_evidence:
        raise RuntimeError(f"Missing vocabulary evidence scores: {missing_evidence}")
    required_vision = {"vision_cls_attention", "vision_attention_rollout"}
    missing_vision = required_vision - evidence.localization_scores.keys()
    if missing_vision:
        raise RuntimeError(f"Missing vision localization scores: {missing_vision}")

    score_maps = {
        "llm_attention": evidence.attention_scores,
        "vision_cls_attention": evidence.localization_scores[
            "vision_cls_attention"
        ],
        "vision_attention_rollout": evidence.localization_scores[
            "vision_attention_rollout"
        ],
        "logit_maxprob": evidence.evidence_scores["maxprob"],
        "logit_margin": evidence.evidence_scores["margin"],
        "logit_negentropy": evidence.evidence_scores["negentropy"],
        "logit_concept": evidence.evidence_scores["concept_logprob"],
    }
    score_maps["attention_logit_fusion"] = (
        normalize_patch_scores(score_maps["llm_attention"])
        + normalize_patch_scores(score_maps["logit_concept"])
    ) / 2
    patch_counts = {scores.numel() for scores in score_maps.values()}
    if len(patch_counts) != 1:
        raise RuntimeError("Phase 01B localizer score maps are not patch-aligned")
    return score_maps
