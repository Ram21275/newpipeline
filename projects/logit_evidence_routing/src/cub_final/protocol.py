"""Automatic protocol lock after training-only smoke checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .core import SCHEMA_VERSION, atomic_write_json, canonical_hash, file_sha256
from .models import LLAVA_CHECKPOINT, LLAVA_REVISION, QWEN3_CHECKPOINT, QWEN3_REVISION


def default_protocol(manifest_dir: Path) -> dict[str, Any]:
    summary = json.loads((manifest_dir / "manifest_summary.json").read_text(encoding="utf-8"))
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "LOCKED",
        "scope": "CUB-200-2011 official-test evaluation restricted to 20 recovered historical species and 26 attributes",
        "cohort": {
            "manifest_hash": summary["manifest_hash"],
            "all_test_questions": "manifests/question_manifest_all_test.jsonl",
            "expensive_questions": "manifests/question_manifest_expensive.jsonl",
            "certainty": ["probably", "definitely"],
            "maximum_expensive_images": 256,
            "maximum_decisions_per_image": 4,
        },
        "models": {
            "llava": {"checkpoint": LLAVA_CHECKPOINT, "revision": LLAVA_REVISION},
            "qwen3": {"checkpoint": QWEN3_CHECKPOINT, "revision": QWEN3_REVISION},
        },
        "precision": {"preferred": "bf16", "fallback": "fp16", "quantization": "none"},
        "answers": {
            "primary": {"positive": "yes", "negative": "no"},
            "alternative": {"positive": "present", "negative": "absent", "fixed_subset_seed": 20260916},
            "score_complete_candidate_sequences": True,
            "generated_parser": "unique_leading_lexical_token_else_invalid",
        },
        "layers": {
            "selection_source": "development_only",
            "final_distribution": "model.logits",
            "intermediate_projection": "final_norm_then_lm_head",
            "no_ground_truth_layer_selection_at_inference": True,
        },
        "selectors": {
            "llava": ["vision_cls", "logit_concept", "decoder_attention", "random", "all_patch"],
            "qwen3": ["decoder_attention", "logit_concept", "random", "all_patch"],
            "vision_cls_is_llava_specific": True,
        },
        "interventions": {
            "primary_k": [8, 16, 32, 64],
            "central_k": 32,
            "primary_control": "norm_matched_random_at_every_k",
            "sensitivity_control": "spatial_and_norm_matched_common_feasible_cohort",
            "replacement": ["training_mean", "training_donor_spatial_norm_matched"],
            "random_draws": 5,
        },
        "shared_reencoding": {
            "arms": ["full", "predicted_crop", "random_crop", "oracle_part_crop", "full_dola", "predicted_crop_dola"],
            "crop_only_primary": True,
            "oracle_geometry_uses_attribute_group_not_answer": True,
            "qwen_compute_bar": "full_image_budget_including_localization_plus_second_pass",
            "dola_scope": "yes_no_restricted_scoring_plus_separate_normal_generation",
            "plausibility_alpha": 0.1,
        },
        "statistics": {
            "cluster_unit": "image_id",
            "bootstrap_resamples": 10000,
            "weighting": "image",
            "dose_layer_multiplicity": "bootstrap_max_t",
            "exploratory_per_attribute": True,
        },
        "seeds": {"split": 20260916, "cohort": 20260916, "controls": [17, 29, 43, 71, 113], "bootstrap": 20260916},
        "stopping_rules": {
            "no_outcome_driven_tuning": True,
            "runtime_adjustment_only_before_lock_from_training_timings": True,
            "implementation_amendments_regenerate_all_affected_arms": True,
        },
    }


def lock_protocol(
    *,
    manifest_dir: Path,
    smoke_dir: Path,
    destination: Path,
    force_rebuild: bool = False,
) -> dict[str, Any]:
    """Lock only after both training-only real-model smoke reports pass."""

    if destination.exists() and not force_rebuild:
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if existing.get("status") != "LOCKED":
            raise RuntimeError("existing protocol is not a valid lock")
        return existing
    reports: dict[str, Any] = {}
    for architecture in ("llava", "qwen3"):
        path = smoke_dir / f"{architecture}_smoke.json"
        if not path.is_file():
            raise RuntimeError(f"missing {architecture} training-only smoke report: {path}")
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("status") != "PASS" or value.get("official_test_images_used") != 0:
            raise RuntimeError(f"{architecture} smoke report is not a passing training-only check")
        reports[architecture] = {"path": str(path), "sha256": file_sha256(path)}
    protocol = default_protocol(manifest_dir)
    protocol["training_only_smoke_reports"] = reports
    protocol["protocol_hash"] = canonical_hash(protocol)
    atomic_write_json(destination, protocol)  # JSON is valid YAML 1.2.
    return protocol


def append_amendment(path: Path, amendment: dict[str, Any]) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "LOCKED":
        raise RuntimeError("cannot amend an unlocked protocol")
    amendments = list(value.get("amendments", []))
    amendments.append(amendment)
    value["amendments"] = amendments
    value["protocol_hash"] = canonical_hash({k: v for k, v in value.items() if k != "protocol_hash"})
    atomic_write_json(path, value)
    return value

