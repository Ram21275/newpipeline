"""Automatic protocol lock after training-only smoke checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .core import SCHEMA_VERSION, atomic_write_json, canonical_hash, file_sha256
from .models import LLAVA_CHECKPOINT, LLAVA_REVISION, QWEN3_CHECKPOINT, QWEN3_REVISION


_LOCKED_MODELS = {
    "llava": (LLAVA_CHECKPOINT, LLAVA_REVISION),
    "qwen3": (QWEN3_CHECKPOINT, QWEN3_REVISION),
}


def validate_smoke_report(value: dict[str, Any], *, architecture: str) -> None:
    """Fail closed on every invariant needed before official-test access."""

    checkpoint, revision = _LOCKED_MODELS[architecture]
    problems: list[str] = []
    if value.get("status") != "PASS":
        problems.append("status is not PASS")
    if value.get("official_test_images_used") != 0:
        problems.append("official_test_images_used is not zero")
    if value.get("checkpoint") != checkpoint:
        problems.append("checkpoint does not match the locked model")
    if value.get("required_revision") != revision:
        problems.append("required_revision does not match the locked revision")
    if value.get("requested_quantization") != "none":
        problems.append("requested_quantization is not none")
    audit = value.get("architecture_audit")
    if not isinstance(audit, dict):
        problems.append("architecture_audit is missing")
        audit = {}
    if audit.get("architecture") != architecture:
        problems.append("architecture_audit architecture is wrong")
    if audit.get("resolved_revision") != revision:
        problems.append("architecture_audit resolved_revision is not the locked revision")
    runtime = audit.get("runtime")
    if not isinstance(runtime, dict) or runtime.get("quantization") != "none":
        problems.append("architecture_audit does not confirm unquantized execution")
    decision = value.get("training_decision")
    if not isinstance(decision, dict) or decision.get("official_split") != "train":
        problems.append("training_decision is not from the official training split")
    measurement = value.get("measurement")
    if not isinstance(measurement, dict):
        problems.append("measurement is missing")
        measurement = {}
    if measurement.get("architecture") != architecture:
        problems.append("measurement architecture is wrong")
    if measurement.get("resolved_revision") != revision:
        problems.append("measurement resolved_revision is not the locked revision")
    agreement = measurement.get("final_logit_agreement")
    if not isinstance(agreement, dict) or "max_abs_error" not in agreement:
        problems.append("final-logit agreement validation is missing")
    if problems:
        raise RuntimeError(f"{architecture} smoke report failed validation: " + "; ".join(problems))


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

    reports: dict[str, Any] = {}
    for architecture in ("llava", "qwen3"):
        path = smoke_dir / f"{architecture}_smoke.json"
        if not path.is_file():
            raise RuntimeError(f"missing {architecture} training-only smoke report: {path}")
        value = json.loads(path.read_text(encoding="utf-8"))
        validate_smoke_report(value, architecture=architecture)
        reports[architecture] = {"path": str(path), "sha256": file_sha256(path)}
    protocol = default_protocol(manifest_dir)
    protocol["training_only_smoke_reports"] = reports
    protocol["protocol_hash"] = canonical_hash(protocol)
    if destination.exists() and not force_rebuild:
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if existing.get("status") != "LOCKED":
            raise RuntimeError("existing protocol is not a valid lock")
        if existing.get("protocol_hash") != protocol["protocol_hash"]:
            raise RuntimeError(
                "existing protocol does not match the current manifests/smoke reports; "
                "do not continue without an explicit audited rebuild"
            )
        return existing
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
