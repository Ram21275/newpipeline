#!/usr/bin/env python3
"""Run matched Phase 3 multi-label attribute probes on validated stage shards."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lger.attribute_probe import (  # noqa: E402
    binary_auroc,
    binary_f1,
    load_stage_attribute_dataset,
    random_project_features,
    run_masked_multilabel_probe,
    shuffle_observed_targets,
)
from lger.cub import PRIMARY_TARGET_CERTAINTY_NAMES  # noqa: E402
from lger.reproducibility import current_git_commit  # noqa: E402
from lger.stage_cache import (  # noqa: E402
    REQUIRED_STAGE_NAMES,
    atomic_json_write,
    write_or_validate_config,
)


CONTROL_NAMES = ("primary", "prevalence", "shuffled_labels", "random_projection")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty Phase 3 table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _attribute_rows(
    *,
    dataset: object,
    stage: str,
    pooling: str,
    control: str,
    seed: int,
    train_targets: torch.Tensor,
    validation_targets: torch.Tensor,
    validation_scores: torch.Tensor,
    thresholds: torch.Tensor,
    train_loss: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for column, attribute_id in enumerate(dataset.attribute_ids):
        train_mask = torch.isfinite(train_targets[:, column])
        validation_mask = torch.isfinite(validation_targets[:, column])
        observed_train = train_targets[train_mask, column]
        observed_validation = validation_targets[validation_mask, column]
        scores = validation_scores[validation_mask, column]
        threshold = float(thresholds[column])
        auroc = binary_auroc(scores, observed_validation) if scores.numel() else float("nan")
        f1 = (
            binary_f1(scores >= threshold, observed_validation)
            if scores.numel()
            else float("nan")
        )
        rows.append(
            {
                "stage": stage,
                "pooling": pooling,
                "control": control,
                "seed": seed,
                "attribute_id": attribute_id,
                "attribute_name": dataset.attribute_names[column],
                "attribute_group": dataset.attribute_groups[column],
                "train_observed": int(train_mask.sum()),
                "train_positive": int((observed_train == 1).sum()),
                "train_negative": int((observed_train == 0).sum()),
                "train_prevalence": float(observed_train.mean()),
                "validation_observed": int(validation_mask.sum()),
                "validation_positive": int((observed_validation == 1).sum()),
                "validation_negative": int((observed_validation == 0).sum()),
                "threshold_selected_on_train": threshold,
                "auroc": auroc,
                "f1": f1,
                "train_loss": train_loss,
            }
        )
    return rows


def _summary_row(rows: list[dict[str, object]], *, feature_dim: int) -> dict[str, object]:
    aurocs = [float(row["auroc"]) for row in rows if math.isfinite(float(row["auroc"]))]
    f1s = [float(row["f1"]) for row in rows if math.isfinite(float(row["f1"]))]
    if not aurocs or not f1s:
        raise RuntimeError("Phase 3 run has no evaluable validation attributes")
    first = rows[0]
    return {
        "stage": first["stage"],
        "pooling": first["pooling"],
        "control": first["control"],
        "seed": first["seed"],
        "macro_auroc": statistics.mean(aurocs),
        "macro_f1": statistics.mean(f1s),
        "evaluable_auroc_attributes": len(aurocs),
        "evaluable_f1_attributes": len(f1s),
        "selected_attributes": len(rows),
        "feature_dim": feature_dim,
        "train_examples": 160,
        "validation_examples": 80,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stages", nargs="+", default=list(REQUIRED_STAGE_NAMES))
    parser.add_argument("--pooling", nargs="+", choices=("mean", "max"), default=["mean"])
    parser.add_argument(
        "--controls", nargs="+", choices=CONTROL_NAMES, default=list(CONTROL_NAMES)
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--random-projection-dim", type=int, default=256)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--prompt-id",
        default="cub_attribute_yes_no_v1",
        help="Frozen Phase 5 prompt identity used to form cross-phase decision IDs.",
    )
    args = parser.parse_args()

    stages = tuple(dict.fromkeys(args.stages))
    pooling_methods = tuple(dict.fromkeys(args.pooling))
    controls = tuple(dict.fromkeys(args.controls))
    seeds = tuple(sorted(set(args.seeds)))
    if not stages or not set(stages).issubset(REQUIRED_STAGE_NAMES):
        raise ValueError("stages contain an unknown Phase 2 stage")
    if not controls or not seeds:
        raise ValueError("at least one control and seed are required")
    if "primary" not in controls:
        raise ValueError("Phase 3R requires the primary control for probe artifacts")
    if not args.prompt_id.strip() or "::" in args.prompt_id:
        raise ValueError("prompt-id must be non-empty and cannot contain '::'")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA probe device requested but unavailable")

    cache_validation_path = args.cache_dir / "validation_report.json"
    cache_run_config = json.loads((args.cache_dir / "run_config.json").read_text())
    representation_prompt = str(cache_run_config.get("prompt", ""))
    if not representation_prompt:
        raise RuntimeError("Phase 2 cache does not record its representation prompt")
    evaluation_config = {
        "schema_version": 1,
        "purpose": "phase3_development_attribute_recoverability",
        "git_commit": current_git_commit(REPO_ROOT),
        "cache_dir": str(args.cache_dir.expanduser().resolve()),
        "cache_validation_sha256": _sha256(cache_validation_path),
        "stages": list(stages),
        "pooling": list(pooling_methods),
        "controls": list(controls),
        "seeds": list(seeds),
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "random_projection_dim": args.random_projection_dim,
        "device": args.device,
        "normalization": "training_feature_mean_std_v1",
        "threshold_policy": "training_f1_only_v1",
        "masked_states": ["guess", "not visible", "missing"],
        "approved_certainty_names": sorted(PRIMARY_TARGET_CERTAINTY_NAMES),
        "certainty_policy_enforcement": "consumer_side_allowlist_v1",
        "official_test_images_used": 0,
        "decision_id_schema": "image_id::attribute_id::prompt_id",
        "prompt_id": args.prompt_id,
        "representation_prompt": representation_prompt,
        "prompt_context_alignment": (
            "vision_and_projector_prompt_invariant_llm_neutral_not_vqa_question_conditioned"
        ),
        "probe_artifacts": "primary_control_all_stages_and_seeds_v1",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_or_validate_config(
        args.output_dir / "evaluation_config.json", evaluation_config
    )
    dataset = load_stage_attribute_dataset(
        args.cache_dir,
        stages=stages,
        pooling=pooling_methods,
    )
    train_indices = torch.tensor([split == "train" for split in dataset.splits])
    validation_indices = torch.tensor([split == "val" for split in dataset.splits])
    train_targets = dataset.targets[train_indices]
    validation_targets = dataset.targets[validation_indices]

    per_attribute: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    decision_scores: list[dict[str, object]] = []
    probe_tensors: dict[str, torch.Tensor] = {}
    probe_records: list[dict[str, object]] = []
    maximum_reconstruction_error = 0.0
    for pooling in pooling_methods:
        for stage in stages:
            features = dataset.features[(pooling, stage)]
            train_features = features[train_indices]
            validation_features = features[validation_indices]
            for control in controls:
                control_seeds = (-1,) if control == "prevalence" else seeds
                for seed in control_seeds:
                    feature_dim = train_features.shape[1]
                    if control == "prevalence":
                        observed = torch.isfinite(train_targets)
                        positives = torch.where(observed, train_targets, 0.0).sum(dim=0)
                        counts = observed.sum(dim=0)
                        prevalence = (positives / counts).clamp(1e-6, 1 - 1e-6)
                        scores = torch.logit(prevalence).repeat(
                            validation_targets.shape[0], 1
                        )
                        thresholds = torch.zeros_like(prevalence)
                        loss = float("nan")
                    else:
                        probe_train_features = train_features
                        probe_validation_features = validation_features
                        probe_targets = train_targets
                        if control == "shuffled_labels":
                            probe_targets = shuffle_observed_targets(
                                train_targets, seed=seed
                            )
                        if control == "random_projection":
                            output_dim = min(
                                args.random_projection_dim, train_features.shape[1]
                            )
                            (
                                probe_train_features,
                                probe_validation_features,
                            ) = random_project_features(
                                train_features,
                                validation_features,
                                output_dim=output_dim,
                                seed=seed,
                            )
                            feature_dim = output_dim
                        result = run_masked_multilabel_probe(
                            probe_train_features,
                            probe_targets,
                            probe_validation_features,
                            seed=seed,
                            epochs=args.epochs,
                            learning_rate=args.learning_rate,
                            weight_decay=args.weight_decay,
                            device=device,
                        )
                        scores = result.validation_logits
                        thresholds = result.thresholds
                        loss = result.train_loss
                        if control == "primary":
                            prefix = f"primary/{pooling}/{stage}/seed_{seed}"
                            probe_tensors.update(
                                {
                                    f"{prefix}/normalization_mean": result.normalization_mean,
                                    f"{prefix}/normalization_scale": result.normalization_scale,
                                    f"{prefix}/classifier_weight": result.classifier_weight,
                                    f"{prefix}/classifier_bias": result.classifier_bias,
                                    f"{prefix}/thresholds": result.thresholds,
                                }
                            )
                            all_features = features.float()
                            reconstructed = (
                                (all_features - result.normalization_mean)
                                / result.normalization_scale
                            ) @ result.classifier_weight.T + result.classifier_bias
                            all_scores = torch.empty_like(reconstructed)
                            all_scores[train_indices] = result.train_logits
                            all_scores[validation_indices] = result.validation_logits
                            error = float((reconstructed - all_scores).abs().max())
                            maximum_reconstruction_error = max(
                                maximum_reconstruction_error, error
                            )
                            probe_records.append(
                                {
                                    "tensor_prefix": prefix,
                                    "stage": stage,
                                    "pooling": pooling,
                                    "control": control,
                                    "seed": seed,
                                    "feature_dim": int(result.classifier_weight.shape[1]),
                                    "attributes": int(result.classifier_weight.shape[0]),
                                    "reconstruction_max_abs_error": error,
                                }
                            )
                            for image_index, image_id in enumerate(
                                dataset.image_ids.tolist()
                            ):
                                for column, attribute_id in enumerate(dataset.attribute_ids):
                                    target = dataset.targets[image_index, column]
                                    observed = bool(torch.isfinite(target))
                                    score = float(all_scores[image_index, column])
                                    threshold = float(result.thresholds[column])
                                    predicted = score >= threshold
                                    decision_scores.append(
                                        {
                                            "decision_id": f"{image_id}::{attribute_id}::{args.prompt_id}",
                                            "image_id": int(image_id),
                                            "split": dataset.splits[image_index],
                                            "attribute_id": int(attribute_id),
                                            "attribute_name": dataset.attribute_names[column],
                                            "attribute_group": dataset.attribute_groups[column],
                                            "prompt_id": args.prompt_id,
                                            "representation_prompt": representation_prompt,
                                            "prompt_context_alignment": (
                                                "vision_and_projector_prompt_invariant_llm_neutral_not_vqa_question_conditioned"
                                            ),
                                            "stage": stage,
                                            "pooling": pooling,
                                            "control": control,
                                            "seed": seed,
                                            "target": int(float(target)) if observed else "",
                                            "observed": int(observed),
                                            "probe_logit": score,
                                            "threshold": threshold,
                                            "probe_margin": score - threshold,
                                            "probe_prediction": int(predicted),
                                            "probe_correct": int(predicted == bool(target)) if observed else "",
                                        }
                                    )
                    rows = _attribute_rows(
                        dataset=dataset,
                        stage=stage,
                        pooling=pooling,
                        control=control,
                        seed=seed,
                        train_targets=train_targets,
                        validation_targets=validation_targets,
                        validation_scores=scores,
                        thresholds=thresholds,
                        train_loss=loss,
                    )
                    per_attribute.extend(rows)
                    summary = _summary_row(rows, feature_dim=feature_dim)
                    summaries.append(summary)
                    print(
                        f"{stage:16s} {pooling:4s} {control:17s} seed={seed:2d} "
                        f"AUROC={summary['macro_auroc']:.4f} "
                        f"F1={summary['macro_f1']:.4f}"
                    )

    _write_csv(args.output_dir / "per_attribute_metrics.csv", per_attribute)
    _write_csv(args.output_dir / "attribute_probe_by_stage.csv", summaries)
    _write_csv(args.output_dir / "decision_probe_scores.csv", decision_scores)
    try:
        from safetensors.torch import save_file
    except ImportError as error:  # pragma: no cover - Kaggle dependency guard
        raise RuntimeError("Phase 3 probe export requires safetensors") from error
    temporary_parameters = args.output_dir / "probe_parameters.safetensors.tmp"
    save_file(
        {name: value.detach().cpu().contiguous() for name, value in probe_tensors.items()},
        str(temporary_parameters),
        metadata={
            "schema_version": "1",
            "decision_id_schema": "image_id::attribute_id::prompt_id",
            "prompt_id": args.prompt_id,
        },
    )
    temporary_parameters.replace(args.output_dir / "probe_parameters.safetensors")
    atomic_json_write(
        {
            "schema_version": 1,
            "decision_id_schema": "image_id::attribute_id::prompt_id",
            "prompt_id": args.prompt_id,
            "records": probe_records,
            "maximum_reconstruction_abs_error": maximum_reconstruction_error,
        },
        args.output_dir / "probe_parameters.json",
    )
    if maximum_reconstruction_error > 1e-4:
        raise RuntimeError(
            "Exported Phase 3 probe parameters do not reconstruct recorded logits"
        )
    report = {
        "schema_version": 2,
        "status": "PASS",
        "cache_config_digest": dataset.cache_config_digest,
        "images": dataset.image_ids.numel(),
        "train_images": int(train_indices.sum()),
        "validation_images": int(validation_indices.sum()),
        "selected_attributes": len(dataset.attribute_ids),
        "stages": list(stages),
        "pooling": list(pooling_methods),
        "controls": list(controls),
        "seeds": list(seeds),
        "result_rows": len(summaries),
        "per_attribute_rows": len(per_attribute),
        "decision_probe_rows": len(decision_scores),
        "probe_parameter_sets": len(probe_records),
        "probe_reconstruction_max_abs_error": maximum_reconstruction_error,
        "decision_id_schema": "image_id::attribute_id::prompt_id",
        "prompt_id": args.prompt_id,
        "representation_prompt": representation_prompt,
        "prompt_context_alignment": (
            "vision_and_projector_prompt_invariant_llm_neutral_not_vqa_question_conditioned"
        ),
        "certainty_policy_overrides_masked": dataset.certainty_policy_overrides_masked,
        "official_test_images_used": 0,
    }
    atomic_json_write(report, args.output_dir / "phase3_run_report.json")
    print(f"Phase 3 attribute-probe run PASS: {args.output_dir}")
    print("STOP: inspect control behavior before launching the full stage trajectory")


if __name__ == "__main__":
    main()
