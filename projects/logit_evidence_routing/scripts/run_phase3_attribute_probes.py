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
    args = parser.parse_args()

    stages = tuple(dict.fromkeys(args.stages))
    pooling_methods = tuple(dict.fromkeys(args.pooling))
    controls = tuple(dict.fromkeys(args.controls))
    seeds = tuple(sorted(set(args.seeds)))
    if not stages or not set(stages).issubset(REQUIRED_STAGE_NAMES):
        raise ValueError("stages contain an unknown Phase 2 stage")
    if not controls or not seeds:
        raise ValueError("at least one control and seed are required")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA probe device requested but unavailable")

    cache_validation_path = args.cache_dir / "validation_report.json"
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
    report = {
        "schema_version": 1,
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
        "certainty_policy_overrides_masked": dataset.certainty_policy_overrides_masked,
        "official_test_images_used": 0,
    }
    atomic_json_write(report, args.output_dir / "phase3_run_report.json")
    print(f"Phase 3 attribute-probe run PASS: {args.output_dir}")
    print("STOP: inspect control behavior before launching the full stage trajectory")


if __name__ == "__main__":
    main()
