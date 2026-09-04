#!/usr/bin/env python3
"""Train matched linear probes and generate Phase 01 machine-readable results."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lger.cub import load_pilot_manifest  # noqa: E402
from lger.probe import jaccard, run_linear_probe  # noqa: E402
from lger.reproducibility import current_git_commit  # noqa: E402


def feature_key(selector: str, k: int, seed: int | None = None) -> str:
    suffix = f"_seed{seed}" if seed is not None else ""
    return f"{selector}_k{k}{suffix}"


def load_cache(cache_dir: Path, expected_ids: set[int]) -> list[dict[str, object]]:
    by_id: dict[int, dict[str, object]] = {}
    for path in sorted((cache_dir / "records").glob("*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        image_id = int(payload["image_id"])
        if image_id in by_id:
            raise RuntimeError(f"Duplicate cache record for image {image_id}")
        by_id[image_id] = payload
    missing = sorted(expected_ids - by_id.keys())
    unexpected = sorted(by_id.keys() - expected_ids)
    if missing or unexpected:
        raise RuntimeError(
            f"Cache/manifest mismatch: {len(missing)} missing, "
            f"{len(unexpected)} unexpected. Missing examples: {missing[:10]}"
        )
    return [by_id[image_id] for image_id in sorted(expected_ids)]


def stack_split(
    records: list[dict[str, object]],
    split: str,
    key: str,
    label_to_index: dict[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    selected = [record for record in records if record["split"] == split]
    features = torch.stack([record["features"][key] for record in selected])
    targets = torch.tensor(
        [label_to_index[int(record["label"])] for record in selected]
    )
    return features, targets


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--k", type=int, nargs="+", default=[16, 32])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    k_values = sorted(set(args.k))
    seeds = sorted(set(args.seeds))
    manifest = load_pilot_manifest(args.manifest)
    expected_ids = {record.image_id for record in manifest}
    cache_records = load_cache(args.cache_dir, expected_ids)
    labels = sorted({record.label for record in manifest})
    label_to_index = {label: index for index, label in enumerate(labels)}
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA probe device requested but unavailable")

    metric_rows: list[dict[str, object]] = []
    selectors = ("random", "attention", "logit")
    for k in k_values:
        for selector in selectors:
            for seed in seeds:
                key = feature_key(selector, k, seed if selector == "random" else None)
                train_features, train_targets = stack_split(
                    cache_records, "train", key, label_to_index
                )
                val_features, val_targets = stack_split(
                    cache_records, "val", key, label_to_index
                )
                result = run_linear_probe(
                    train_features,
                    train_targets,
                    val_features,
                    val_targets,
                    num_classes=len(labels),
                    seed=seed,
                    epochs=args.epochs,
                    learning_rate=args.learning_rate,
                    weight_decay=args.weight_decay,
                    device=device,
                )
                row = {
                    "selector": selector,
                    "K": k,
                    "seed": seed,
                    "accuracy": result.accuracy,
                    "macro_f1": result.macro_f1,
                    "train_loss": result.train_loss,
                    "train_examples": train_features.shape[0],
                    "val_examples": val_features.shape[0],
                    "feature_dim": train_features.shape[1],
                }
                metric_rows.append(row)
                print(
                    f"{selector:9s} K={k:2d} seed={seed}: "
                    f"accuracy={result.accuracy:.4f}, macro_f1={result.macro_f1:.4f}"
                )

    metric_fields = [
        "selector",
        "K",
        "seed",
        "accuracy",
        "macro_f1",
        "train_loss",
        "train_examples",
        "val_examples",
        "feature_dim",
    ]
    write_csv(args.output_dir / "selector_metrics.csv", metric_rows, metric_fields)

    summary_rows: list[dict[str, object]] = []
    for k in k_values:
        for selector in selectors:
            group = [
                row
                for row in metric_rows
                if row["selector"] == selector and row["K"] == k
            ]
            accuracy = [float(row["accuracy"]) for row in group]
            macro_f1 = [float(row["macro_f1"]) for row in group]
            summary_rows.append(
                {
                    "selector": selector,
                    "K": k,
                    "accuracy_mean": statistics.mean(accuracy),
                    "accuracy_std": statistics.stdev(accuracy) if len(accuracy) > 1 else 0.0,
                    "macro_f1_mean": statistics.mean(macro_f1),
                    "macro_f1_std": statistics.stdev(macro_f1) if len(macro_f1) > 1 else 0.0,
                    "seeds": len(group),
                }
            )
    write_csv(
        args.output_dir / "selector_summary.csv",
        summary_rows,
        [
            "selector",
            "K",
            "accuracy_mean",
            "accuracy_std",
            "macro_f1_mean",
            "macro_f1_std",
            "seeds",
        ],
    )

    statistic_rows: list[dict[str, object]] = []
    for record in cache_records:
        selections = record["selections"]
        for k in k_values:
            attention = selections[feature_key("attention", k)]
            logit = selections[feature_key("logit", k)]
            for seed in seeds:
                random_indices = selections[feature_key("random", k, seed)]
                statistic_rows.append(
                    {
                        "image_id": record["image_id"],
                        "split": record["split"],
                        "K": k,
                        "seed": seed,
                        "attention_logit_jaccard": jaccard(attention, logit),
                        "random_attention_jaccard": jaccard(random_indices, attention),
                        "random_logit_jaccard": jaccard(random_indices, logit),
                        "attention_unique": attention.unique().numel(),
                        "logit_unique": logit.unique().numel(),
                        "random_unique": random_indices.unique().numel(),
                    }
                )
    write_csv(
        args.output_dir / "patch_statistics.csv",
        statistic_rows,
        [
            "image_id",
            "split",
            "K",
            "seed",
            "attention_logit_jaccard",
            "random_attention_jaccard",
            "random_logit_jaccard",
            "attention_unique",
            "logit_unique",
            "random_unique",
        ],
    )

    evaluation_config = {
        "git_commit": current_git_commit(REPO_ROOT),
        "manifest": str(args.manifest.resolve()),
        "cache_dir": str(args.cache_dir.resolve()),
        "k_values": k_values,
        "seeds": seeds,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "device": args.device,
        "num_classes": len(labels),
        "official_test_images_used": 0,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "evaluation_config.json").write_text(
        json.dumps(evaluation_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "notes.md").write_text(
        "# Phase 01 notes\n\n"
        "This is a validation-only CUB pilot. The official CUB test partition was "
        "not used. Random, attention, and logit selectors use the same patch budget, "
        "original hidden-state mean pooling, and linear-probe training protocol.\n\n"
        "Do not treat the phenomenon as established until the selector metrics are "
        "inspected together with patch overlap and qualitative examples.\n",
        encoding="utf-8",
    )
    print(f"Results written to {args.output_dir}")


if __name__ == "__main__":
    main()
