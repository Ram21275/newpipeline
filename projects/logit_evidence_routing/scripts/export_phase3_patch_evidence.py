#!/usr/bin/env python3
"""Export exact attribute-probe patch evidence from the reusable Phase 2 cache."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from lger.cub import certainty_policy_target  # noqa: E402
from lger.phase4 import load_part_vocabulary, load_policy, read_json, require  # noqa: E402
from lger.probe_localization import exact_patch_margin, patch_evidence_summary  # noqa: E402
from lger.stage_cache import REQUIRED_STAGE_NAMES, atomic_json_write, config_digest  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    require(bool(rows), f"refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-cache", type=Path, required=True)
    parser.add_argument("--phase3-dir", type=Path, required=True)
    parser.add_argument("--policy", type=Path, default=PROJECT / "configs/phase4_localization.json")
    parser.add_argument("--part-vocabulary", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--stages", nargs="+", default=list(REQUIRED_STAGE_NAMES))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--k", type=int, nargs="+", default=[16, 32])
    args = parser.parse_args()

    stages = tuple(dict.fromkeys(args.stages))
    seeds = tuple(sorted(set(args.seeds)))
    k_values = tuple(sorted(set(args.k)))
    require(stages and set(stages).issubset(REQUIRED_STAGE_NAMES), "unknown stage")
    require(seeds and k_values and min(k_values) > 0, "invalid seeds or K values")
    policy = load_policy(args.policy)
    run_config = read_json(args.stage_cache / "run_config.json")
    index = read_json(args.stage_cache / "index.json")
    validation = read_json(args.stage_cache / "validation_report.json")
    digest = config_digest(run_config)
    phase3 = read_json(args.phase3_dir / "phase3_run_report.json")
    require(
        digest == policy["cache_config_digest"] == index["config_digest"] == validation["config_digest"]
        and index["complete"] is True
        and validation["status"] == "PASS"
        and validation["official_test_images"] == 0
        and phase3["schema_version"] == 2
        and phase3["status"] == "PASS"
        and phase3["cache_config_digest"] == digest
        and phase3["official_test_images_used"] == 0,
        "validated corrected development cache and Phase 3R are required",
    )
    parameter_path = args.phase3_dir / "probe_parameters.safetensors"
    parameter_index = read_json(args.phase3_dir / "probe_parameters.json")
    require(
        parameter_index["prompt_id"] == phase3["prompt_id"]
        and parameter_index["maximum_reconstruction_abs_error"] <= 1e-4,
        "Phase 3R probe parameter gate failed",
    )
    vocabulary_path = args.part_vocabulary or Path(run_config["dataset"]["root"]) / "parts/parts.txt"
    part_names = load_part_vocabulary(vocabulary_path, policy)
    attributes = {int(row["attribute_id"]): row for row in policy["attributes"]}
    score_rows = read_csv(args.phase3_dir / "decision_probe_scores.csv")
    recorded_scores = {
        (row["decision_id"], row["stage"], int(row["seed"])): float(row["probe_margin"])
        for row in score_rows
        if row["split"] == args.split
        and row["pooling"] == "mean"
        and row["control"] == "primary"
        and row["observed"] == "1"
        and row["stage"] in stages
        and int(row["seed"]) in seeds
    }

    try:
        from safetensors import safe_open
    except ImportError as error:  # pragma: no cover - Kaggle dependency guard
        raise RuntimeError("probe localization export requires safetensors") from error

    parameter_tensors: dict[str, torch.Tensor] = {}
    with safe_open(parameter_path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
        require(
            metadata.get("prompt_id") == phase3["prompt_id"],
            "probe parameter metadata differs",
        )
        for name in handle.keys():
            parameter_tensors[name] = handle.get_tensor(name).float()

    rows: list[dict[str, object]] = []
    maximum_error = 0.0
    for shard in index["shards"]:
        metadata_path = args.stage_cache / shard["metadata_path"]
        require(sha256(metadata_path) == shard["metadata_sha256"], "stage metadata hash differs")
        shard_metadata = read_json(metadata_path)
        with safe_open(args.stage_cache / shard["tensor_path"], framework="pt", device="cpu") as handle:
            for item in shard_metadata["records"]:
                packed = item["packed_record"]
                image = packed["image"]
                if image["development_split"] != args.split:
                    continue
                require(image["official_split"] == "train", "official-test record encountered")
                selected_ids = [int(value) for value in image["selected_attribute_ids"]]
                require(set(selected_ids) == set(attributes), "selected attribute identity differs")
                selected_labels = {
                    int(row["attribute_id"]): row
                    for row in image["attributes"]
                    if int(row["attribute_id"]) in attributes
                }
                part_by_id = {int(row["part_id"]): row for row in image["parts"]}
                grid_size = tuple(int(value) for value in packed["spatial"]["grid_size"])
                image_size = tuple(int(value) for value in packed["spatial"]["processed_image_size_hw"])
                bbox = tuple(float(value) for value in image["bbox_model_xyxy"])
                for stage in stages:
                    tensor_key = packed["stages"][stage].get("__stage_cache_tensor__")
                    require(isinstance(tensor_key, str), "stage tensor reference is missing")
                    patch_states = handle.get_tensor(tensor_key).float()
                    for seed in seeds:
                        prefix = f"primary/mean/{stage}/seed_{seed}"
                        mean = parameter_tensors[f"{prefix}/normalization_mean"]
                        scale = parameter_tensors[f"{prefix}/normalization_scale"]
                        weights = parameter_tensors[f"{prefix}/classifier_weight"]
                        biases = parameter_tensors[f"{prefix}/classifier_bias"]
                        thresholds = parameter_tensors[f"{prefix}/thresholds"]
                        require(weights.shape[0] == len(selected_ids), "probe attribute dimension differs")
                        for column, attribute_id in enumerate(selected_ids):
                            target, _ = certainty_policy_target(selected_labels[attribute_id])
                            if target is None:
                                continue
                            relevant_part_ids = {
                                part_names[name]
                                for name in attributes[attribute_id]["relevant_parts"]
                            }
                            points = [
                                tuple(float(value) for value in part["model_xy"])
                                for part_id, part in part_by_id.items()
                                if part_id in relevant_part_ids
                                and part["visible"]
                                and part["model_xy"] is not None
                            ]
                            patch_margin = exact_patch_margin(
                                patch_states,
                                normalization_mean=mean,
                                normalization_scale=scale,
                                weight=weights[column],
                                bias=biases[column],
                                threshold=thresholds[column],
                            )
                            summary = patch_evidence_summary(
                                patch_margin,
                                grid_size=grid_size,
                                image_size=image_size,
                                bbox_xyxy=bbox,
                                landmark_points=points,
                                k_values=k_values,
                            )
                            decision_id = f"{int(image['image_id'])}::{attribute_id}::{phase3['prompt_id']}"
                            expected = recorded_scores[(decision_id, stage, seed)]
                            error = abs(float(summary["probe_margin_reconstructed"]) - expected)
                            maximum_error = max(maximum_error, error)
                            require(error <= 2e-4, "patch evidence does not reconstruct Phase 3 score")
                            rows.append(
                                {
                                    "decision_id": decision_id,
                                    "image_id": int(image["image_id"]),
                                    "split": args.split,
                                    "attribute_id": attribute_id,
                                    "attribute_name": attributes[attribute_id]["name"],
                                    "attribute_group": attributes[attribute_id]["group"],
                                    "prompt_id": phase3["prompt_id"],
                                    "target": int(target),
                                    "stage": stage,
                                    "hidden_state_index": packed["stage_metadata"][stage]["hidden_state_index"],
                                    "pooling": "mean",
                                    "control": "primary",
                                    "seed": seed,
                                    "recorded_probe_margin": expected,
                                    "reconstruction_abs_error": error,
                                    **{
                                        key: json.dumps(value) if key.endswith("_indices") else value
                                        for key, value in summary.items()
                                    },
                                }
                            )

    require(rows and maximum_error <= 2e-4, "probe patch evidence export is empty or inconsistent")
    expected_keys = set(recorded_scores)
    actual_keys = {(row["decision_id"], row["stage"], int(row["seed"])) for row in rows}
    require(actual_keys == expected_keys, "probe patch evidence coverage differs from decision scores")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "probe_patch_evidence.csv"
    write_csv(output_path, rows)
    report = {
        "schema_version": 1,
        "status": "PASS",
        "split": args.split,
        "rows": len(rows),
        "decisions": len({row["decision_id"] for row in rows}),
        "stages": list(stages),
        "seeds": list(seeds),
        "k_values": list(k_values),
        "prompt_id": phase3["prompt_id"],
        "maximum_reconstruction_abs_error": maximum_error,
        "probe_parameters_sha256": sha256(parameter_path),
        "decision_probe_scores_sha256": sha256(args.phase3_dir / "decision_probe_scores.csv"),
        "probe_patch_evidence_sha256": sha256(output_path),
        "official_test_images_used": 0,
        "interpretation": "exact linear-probe decomposition; localization and concentration are diagnostic, not causal",
    }
    atomic_json_write(report, args.output_dir / "probe_patch_evidence_report.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
