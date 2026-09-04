"""Shared selector definitions for the Phase 01B localization benchmark."""

from __future__ import annotations

from typing import Any

import torch

from .scoring import normalize_patch_scores, stable_topk


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


def refresh_concept_features(
    record: dict[str, Any],
    concept_logprob: torch.Tensor,
    k_values: list[int],
) -> None:
    """Replace concept/fusion scores and their derived selections in a cache row."""

    hidden_states = record.get("patch_hidden_states")
    score_maps = record.get("score_maps")
    if not isinstance(hidden_states, torch.Tensor) or hidden_states.ndim != 2:
        raise ValueError("cache record lacks [patches, hidden] patch_hidden_states")
    if not isinstance(score_maps, dict) or "llm_attention" not in score_maps:
        raise ValueError("cache record lacks the llm_attention score map")
    if concept_logprob.ndim != 1 or concept_logprob.numel() != hidden_states.shape[0]:
        raise ValueError("concept scores must align with cached patch hidden states")
    llm_attention = score_maps["llm_attention"]
    if (
        not isinstance(llm_attention, torch.Tensor)
        or llm_attention.numel() != concept_logprob.numel()
    ):
        raise ValueError("LLM attention and concept scores are not patch-aligned")

    corrected_concept = concept_logprob.detach().cpu()
    corrected_fusion = (
        normalize_patch_scores(llm_attention.detach().cpu())
        + normalize_patch_scores(corrected_concept)
    ) / 2
    score_maps["logit_concept"] = corrected_concept
    score_maps["attention_logit_fusion"] = corrected_fusion

    features = record.get("features")
    selections = record.get("selections")
    if not isinstance(features, dict) or not isinstance(selections, dict):
        raise ValueError("cache record lacks feature/selection dictionaries")
    for k in k_values:
        if k <= 0 or k > hidden_states.shape[0]:
            raise ValueError(f"K={k} is invalid for {hidden_states.shape[0]} patches")
        for selector, scores in (
            ("logit_concept", corrected_concept),
            ("attention_logit_fusion", corrected_fusion),
        ):
            indices = stable_topk(scores, k)
            key = feature_key(selector, k)
            selections[key] = indices
            features[key] = hidden_states.index_select(0, indices).float().mean(0)
