#!/usr/bin/env python3
"""Evaluate Phase 01B selectors with matched probes and CUB localization metrics."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lger.cub import load_pilot_manifest  # noqa: E402
from lger.localization import (  # noqa: E402
    patch_centers_in_box,
    selection_localization_metrics,
)
from lger.phase1b import ALL_PROBE_SELECTORS, DETERMINISTIC_LOCALIZERS, feature_key  # noqa: E402
from lger.probe import jaccard, run_linear_probe  # noqa: E402
from lger.reproducibility import current_git_commit  # noqa: E402


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_cache(cache_dir: Path, expected_ids: set[int]) -> list[dict[str, object]]:
    by_id: dict[int, dict[str, object]] = {}
    for path in sorted((cache_dir / "records").glob("*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        image_id = int(payload["image_id"])
        if image_id in by_id:
            raise RuntimeError(f"Duplicate cache record for image {image_id}")
        if int(payload.get("schema_version", 0)) != 2:
            raise RuntimeError(f"Phase 01B requires schema version 2: {path}")
        payload.pop("patch_hidden_states", None)
        payload.pop("processed_image", None)
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
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    selected = [record for record in records if record["split"] == split]
    try:
        features = torch.stack([record["features"][key] for record in selected])
    except KeyError as error:
        raise RuntimeError(f"Cache does not contain Phase 01B feature {key!r}") from error
    targets = torch.tensor(
        [label_to_index[int(record["label"])] for record in selected]
    )
    image_ids = [int(record["image_id"]) for record in selected]
    return features, targets, image_ids


def _mean_and_std(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--k", type=int, nargs="+", default=[16, 32])
    parser.add_argument("--selection-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--probe-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--selectors",
        nargs="+",
        choices=ALL_PROBE_SELECTORS,
        default=list(ALL_PROBE_SELECTORS),
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    k_values = sorted(set(args.k))
    selection_seeds = sorted(set(args.selection_seeds))
    probe_seeds = sorted(set(args.probe_seeds))
    selectors = tuple(dict.fromkeys(args.selectors))
    if not k_values or min(k_values) <= 0:
        raise ValueError("all K values must be positive")
    if "random" in selectors and not selection_seeds:
        raise ValueError("random requires at least one selection seed")
    if not probe_seeds:
        raise ValueError("at least one probe seed is required")

    manifest = load_pilot_manifest(args.manifest)
    expected_ids = {record.image_id for record in manifest}
    cache_records = load_cache(args.cache_dir, expected_ids)
    labels = sorted({record.label for record in manifest})
    label_to_index = {label: index for index, label in enumerate(labels)}
    index_to_label = {index: label for label, index in label_to_index.items()}
    class_name_by_label = {record.label: record.class_name for record in manifest}
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA probe device requested but unavailable")

    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    for selector in selectors:
        selector_k_values: list[int | None] = [None] if selector == "global_all" else k_values
        selector_seeds: list[int | None] = (
            selection_seeds if selector == "random" else [None]
        )
        for k in selector_k_values:
            for selection_seed in selector_seeds:
                key = feature_key(selector, k, selection_seed)
                train_features, train_targets, _ = stack_split(
                    cache_records, "train", key, label_to_index
                )
                val_features, val_targets, val_image_ids = stack_split(
                    cache_records, "val", key, label_to_index
                )
                for probe_seed in probe_seeds:
                    result = run_linear_probe(
                        train_features,
                        train_targets,
                        val_features,
                        val_targets,
                        num_classes=len(labels),
                        seed=probe_seed,
                        epochs=args.epochs,
                        learning_rate=args.learning_rate,
                        weight_decay=args.weight_decay,
                        device=device,
                    )
                    metric_rows.append(
                        {
                            "selector": selector,
                            "K": "" if k is None else k,
                            "selection_seed": (
                                "" if selection_seed is None else selection_seed
                            ),
                            "probe_seed": probe_seed,
                            "accuracy": result.accuracy,
                            "macro_f1": result.macro_f1,
                            "train_loss": result.train_loss,
                            "train_examples": train_features.shape[0],
                            "val_examples": val_features.shape[0],
                            "feature_dim": train_features.shape[1],
                        }
                    )
                    for image_id, target, prediction in zip(
                        val_image_ids,
                        val_targets.tolist(),
                        result.predictions.tolist(),
                    ):
                        prediction_rows.append(
                            {
                                "selector": selector,
                                "K": "" if k is None else k,
                                "selection_seed": (
                                    "" if selection_seed is None else selection_seed
                                ),
                                "probe_seed": probe_seed,
                                "image_id": image_id,
                                "target_index": target,
                                "prediction_index": prediction,
                                "target_label": index_to_label[target],
                                "prediction_label": index_to_label[prediction],
                                "target_class_name": class_name_by_label[
                                    index_to_label[target]
                                ],
                                "prediction_class_name": class_name_by_label[
                                    index_to_label[prediction]
                                ],
                                "correct": int(target == prediction),
                            }
                        )
                    rendered_k = "all" if k is None else str(k)
                    rendered_selection = (
                        "-" if selection_seed is None else str(selection_seed)
                    )
                    print(
                        f"{selector:26s} K={rendered_k:>3s} "
                        f"selection={rendered_selection} probe={probe_seed}: "
                        f"accuracy={result.accuracy:.4f}, macro_f1={result.macro_f1:.4f}"
                    )

    metric_fields = [
        "selector",
        "K",
        "selection_seed",
        "probe_seed",
        "accuracy",
        "macro_f1",
        "train_loss",
        "train_examples",
        "val_examples",
        "feature_dim",
    ]
    write_csv(args.output_dir / "selector_metrics.csv", metric_rows, metric_fields)
    write_csv(
        args.output_dir / "validation_predictions.csv",
        prediction_rows,
        [
            "selector",
            "K",
            "selection_seed",
            "probe_seed",
            "image_id",
            "target_index",
            "prediction_index",
            "target_label",
            "prediction_label",
            "target_class_name",
            "prediction_class_name",
            "correct",
        ],
    )

    grouped_metrics: dict[tuple[str, object], list[dict[str, object]]] = defaultdict(list)
    for row in metric_rows:
        grouped_metrics[(str(row["selector"]), row["K"])].append(row)
    metric_means = {
        key: {
            "accuracy": statistics.mean(float(row["accuracy"]) for row in group),
            "macro_f1": statistics.mean(float(row["macro_f1"]) for row in group),
        }
        for key, group in grouped_metrics.items()
    }
    summary_rows: list[dict[str, object]] = []
    for (selector, k), group in grouped_metrics.items():
        accuracy_mean, accuracy_std = _mean_and_std(
            [float(row["accuracy"]) for row in group]
        )
        f1_mean, f1_std = _mean_and_std(
            [float(row["macro_f1"]) for row in group]
        )
        random_mean = metric_means.get(("random", k))
        attention_mean = metric_means.get(("llm_attention", k))
        summary_rows.append(
            {
                "selector": selector,
                "K": k,
                "accuracy_mean": accuracy_mean,
                "accuracy_std": accuracy_std,
                "macro_f1_mean": f1_mean,
                "macro_f1_std": f1_std,
                "accuracy_delta_vs_random": (
                    "" if random_mean is None else accuracy_mean - random_mean["accuracy"]
                ),
                "macro_f1_delta_vs_random": (
                    "" if random_mean is None else f1_mean - random_mean["macro_f1"]
                ),
                "accuracy_delta_vs_llm_attention": (
                    ""
                    if attention_mean is None
                    else accuracy_mean - attention_mean["accuracy"]
                ),
                "macro_f1_delta_vs_llm_attention": (
                    ""
                    if attention_mean is None
                    else f1_mean - attention_mean["macro_f1"]
                ),
                "selection_seeds": max(
                    1,
                    len(
                        {
                            row["selection_seed"]
                            for row in group
                            if row["selection_seed"] != ""
                        }
                    ),
                ),
                "probe_seeds": len({row["probe_seed"] for row in group}),
                "runs": len(group),
            }
        )
    summary_rows.sort(key=lambda row: (str(row["K"]), str(row["selector"])))
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
            "accuracy_delta_vs_random",
            "macro_f1_delta_vs_random",
            "accuracy_delta_vs_llm_attention",
            "macro_f1_delta_vs_llm_attention",
            "selection_seeds",
            "probe_seeds",
            "runs",
        ],
    )

    localization_rows: list[dict[str, object]] = []
    localizer_selectors = [selector for selector in selectors if selector != "global_all"]
    for record in cache_records:
        grid_size = tuple(int(value) for value in record["grid_size"])
        bbox = tuple(float(value) for value in record["bbox_xyxy_model"])
        image_size = tuple(int(value) for value in record["processed_image_size"])
        bbox_mask = patch_centers_in_box(grid_size, image_size, bbox)
        for k in k_values:
            attention = record["selections"][feature_key("llm_attention", k)]
            concept = record["selections"][feature_key("logit_concept", k)]
            for selector in localizer_selectors:
                seeds: list[int | None] = (
                    selection_seeds if selector == "random" else [None]
                )
                for selection_seed in seeds:
                    key = feature_key(selector, k, selection_seed)
                    try:
                        selected = record["selections"][key]
                    except KeyError as error:
                        raise RuntimeError(
                            f"Cache does not contain Phase 01B selection {key!r}"
                        ) from error
                    metrics = selection_localization_metrics(selected, bbox_mask)
                    localization_rows.append(
                        {
                            "image_id": record["image_id"],
                            "split": record["split"],
                            "selector": selector,
                            "K": k,
                            "selection_seed": (
                                "" if selection_seed is None else selection_seed
                            ),
                            **metrics,
                            "llm_attention_jaccard": jaccard(selected, attention),
                            "logit_concept_jaccard": jaccard(selected, concept),
                            "selected_unique": selected.unique().numel(),
                            "bbox_patch_count": int(bbox_mask.sum()),
                        }
                    )
    localization_fields = [
        "image_id",
        "split",
        "selector",
        "K",
        "selection_seed",
        "inside_fraction",
        "bbox_patch_recall",
        "bbox_patch_iou",
        "pointing_game",
        "llm_attention_jaccard",
        "logit_concept_jaccard",
        "selected_unique",
        "bbox_patch_count",
    ]
    write_csv(
        args.output_dir / "localization_metrics.csv",
        localization_rows,
        localization_fields,
    )
    grouped_localization: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for row in localization_rows:
        if row["split"] == "val":
            grouped_localization[(str(row["selector"]), int(row["K"]))].append(row)
    localization_means = {
        key: {
            measure: statistics.mean(float(item[measure]) for item in group)
            for measure in (
                "inside_fraction",
                "bbox_patch_recall",
                "bbox_patch_iou",
                "pointing_game",
                "llm_attention_jaccard",
                "logit_concept_jaccard",
            )
        }
        for key, group in grouped_localization.items()
    }
    localization_summary: list[dict[str, object]] = []
    localization_measures = [
        "inside_fraction",
        "bbox_patch_recall",
        "bbox_patch_iou",
        "pointing_game",
        "llm_attention_jaccard",
        "logit_concept_jaccard",
    ]
    for (selector, k), group in sorted(grouped_localization.items()):
        row: dict[str, object] = {
            "selector": selector,
            "K": k,
            "unique_images": len({int(item["image_id"]) for item in group}),
            "evaluation_rows": len(group),
            "selection_seeds": max(
                1,
                len(
                    {
                        item["selection_seed"]
                        for item in group
                        if item["selection_seed"] != ""
                    }
                ),
            ),
        }
        for measure in localization_measures:
            row[f"{measure}_mean"] = localization_means[(selector, k)][measure]
        random_mean = localization_means.get(("random", k))
        for measure in ("inside_fraction", "bbox_patch_iou", "pointing_game"):
            row[f"{measure}_delta_vs_random"] = (
                ""
                if random_mean is None
                else float(row[f"{measure}_mean"]) - random_mean[measure]
            )
        localization_summary.append(row)
    write_csv(
        args.output_dir / "localization_summary.csv",
        localization_summary,
        [
            "selector",
            "K",
            "unique_images",
            "evaluation_rows",
            "selection_seeds",
            *(f"{measure}_mean" for measure in localization_measures),
            "inside_fraction_delta_vs_random",
            "bbox_patch_iou_delta_vs_random",
            "pointing_game_delta_vs_random",
        ],
    )

    cache_config_path = args.cache_dir / "extraction_config.json"
    cache_config = (
        json.loads(cache_config_path.read_text(encoding="utf-8"))
        if cache_config_path.is_file()
        else None
    )
    evaluation_config = {
        "schema_version": 2,
        "git_commit": current_git_commit(REPO_ROOT),
        "manifest": str(args.manifest.resolve()),
        "cache_dir": str(args.cache_dir.resolve()),
        "cache_config": cache_config,
        "selectors": list(selectors),
        "k_values": k_values,
        "selection_seeds": selection_seeds,
        "probe_seeds": probe_seeds,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "device": args.device,
        "num_classes": len(labels),
        "official_test_images_used": 0,
        "bounding_boxes_used_for": "evaluation_only",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "evaluation_config.json").write_text(
        json.dumps(evaluation_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "notes.md").write_text(
        "# Phase 01B notes\n\n"
        "This rerun compares frozen, same-backbone localization signals. CUB "
        "bounding boxes are used only for evaluation. `logit_concept` uses the "
        "same fixed concept vocabulary for every image and never receives the "
        "per-image species label. Selection seeds and probe seeds are recorded "
        "separately.\n",
        encoding="utf-8",
    )
    print(f"Phase 01B results written to {args.output_dir}")


if __name__ == "__main__":
    main()
