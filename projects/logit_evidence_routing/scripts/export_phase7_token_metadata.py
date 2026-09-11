#!/usr/bin/env python3
"""Export a bounded, matched Phase 7 token-evidence cohort from Phase 2/3R/6."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from lger.localization import patch_centers_in_box  # noqa: E402
from lger.phase4 import load_policy, read_json, require  # noqa: E402
from lger.phase7_bridge import (  # noqa: E402
    CAUSAL_STAGE_ORDER,
    resolve_intervention_stages,
    select_matched_causal_cohort,
)
from lger.probe_localization import exact_patch_margin  # noqa: E402
from lger.stage_cache import atomic_json_write, config_digest  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    require(bool(rows), f"input CSV is empty: {path}")
    return rows


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
    parser.add_argument("--phase6-dir", type=Path, required=True)
    parser.add_argument(
        "--policy", type=Path, default=PROJECT / "configs/phase4_localization.json"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--max-per-outcome-per-attribute", type=int, default=5)
    parser.add_argument("--cohort-seed", type=int, default=424242)
    args = parser.parse_args()

    policy = load_policy(args.policy)
    run_config = read_json(args.stage_cache / "run_config.json")
    index = read_json(args.stage_cache / "index.json")
    validation = read_json(args.stage_cache / "validation_report.json")
    phase3 = read_json(args.phase3_dir / "phase3_run_report.json")
    phase6 = read_json(args.phase6_dir / "phase6_run_report.json")
    transition = read_json(args.phase6_dir / "transition_decision.json")
    digest = config_digest(run_config)
    require(
        digest == policy["cache_config_digest"] == index["config_digest"]
        == validation["config_digest"] == phase3["cache_config_digest"]
        and index["complete"] is True
        and validation["status"] == phase3["status"] == phase6["status"] == "PASS"
        and validation["official_test_images"] == 0
        and phase3["official_test_images_used"] == phase6["official_test_images_used"] == 0,
        "validated development-only Phase 2/3R/6 inputs are required",
    )
    require(
        phase6["selected_transition"] == transition["selected_transition"],
        "Phase 6 transition artifacts disagree",
    )
    selected_stage, neighbor_stage = resolve_intervention_stages(
        str(transition["selected_transition"])
    )
    stages = (selected_stage, neighbor_stage)
    seeds = tuple(sorted(set(args.seeds)))
    require(seeds, "at least one probe seed is required")

    joint_rows = read_csv(args.phase6_dir / "joint_decisions.csv")
    cohort = select_matched_causal_cohort(
        joint_rows,
        max_per_outcome_per_attribute=args.max_per_outcome_per_attribute,
        seed=args.cohort_seed,
    )
    cohort_by_decision = {str(row["decision_id"]): row for row in cohort}

    try:
        from safetensors import safe_open
    except ImportError as error:  # pragma: no cover - Kaggle dependency guard
        raise RuntimeError("Phase 7 token export requires safetensors") from error

    parameter_path = args.phase3_dir / "probe_parameters.safetensors"
    parameters: dict[str, torch.Tensor] = {}
    with safe_open(parameter_path, framework="pt", device="cpu") as handle:
        for name in handle.keys():
            parameters[name] = handle.get_tensor(name).float()

    output_rows: list[dict[str, object]] = []
    found: set[str] = set()
    stage_means: dict[str, torch.Tensor] = {}
    for shard in index["shards"]:
        metadata_path = args.stage_cache / shard["metadata_path"]
        require(sha256(metadata_path) == shard["metadata_sha256"], "stage metadata hash differs")
        shard_metadata = read_json(metadata_path)
        tensor_path = args.stage_cache / shard["tensor_path"]
        require(sha256(tensor_path) == shard["tensor_sha256"], "stage tensor hash differs")
        with safe_open(tensor_path, framework="pt", device="cpu") as handle:
            for item in shard_metadata["records"]:
                packed = item["packed_record"]
                image = packed["image"]
                if image["development_split"] != "val":
                    continue
                require(image["official_split"] == "train", "official-test record encountered")
                image_id = int(image["image_id"])
                selected_ids = [int(value) for value in image["selected_attribute_ids"]]
                bbox = tuple(float(value) for value in image["bbox_model_xyxy"])
                grid_size = tuple(int(value) for value in packed["spatial"]["grid_size"])
                image_size = tuple(
                    int(value) for value in packed["spatial"]["processed_image_size_hw"]
                )
                box_mask = patch_centers_in_box(grid_size, image_size, bbox)
                for column, attribute_id in enumerate(selected_ids):
                    decision_id = f"{image_id}::{attribute_id}::{phase3['prompt_id']}"
                    joint = cohort_by_decision.get(decision_id)
                    if joint is None:
                        continue
                    found.add(decision_id)
                    target = int(joint["target"])
                    require(target in (0, 1), "Phase 7 target must be binary")
                    for stage in stages:
                        tensor_key = packed["stages"][stage].get("__stage_cache_tensor__")
                        require(isinstance(tensor_key, str), "stage tensor reference is missing")
                        patch_states = handle.get_tensor(tensor_key).float()
                        seed_scores: list[torch.Tensor] = []
                        for seed in seeds:
                            prefix = f"primary/mean/{stage}/seed_{seed}"
                            mean = parameters[f"{prefix}/normalization_mean"]
                            scale = parameters[f"{prefix}/normalization_scale"]
                            weights = parameters[f"{prefix}/classifier_weight"]
                            biases = parameters[f"{prefix}/classifier_bias"]
                            thresholds = parameters[f"{prefix}/thresholds"]
                            require(column < weights.shape[0], "probe attribute order differs")
                            score = exact_patch_margin(
                                patch_states,
                                normalization_mean=mean,
                                normalization_scale=scale,
                                weight=weights[column],
                                bias=biases[column],
                                threshold=thresholds[column],
                            )
                            seed_scores.append(score if target else -score)
                            previous = stage_means.setdefault(stage, mean)
                            require(
                                previous.shape == mean.shape
                                and torch.allclose(previous, mean, atol=1e-6, rtol=1e-6),
                                f"development training mean differs across seeds for {stage}",
                            )
                        evidence = torch.stack(seed_scores).mean(dim=0)
                        norms = patch_states.norm(dim=-1)
                        require(
                            evidence.shape == norms.shape == box_mask.shape,
                            "token evidence does not align with cached patches",
                        )
                        for token_index in range(evidence.numel()):
                            output_rows.append(
                                {
                                    "decision_id": decision_id,
                                    "image_id": image_id,
                                    "attribute_id": attribute_id,
                                    "target": target,
                                    "phase5_failure": int(joint["phase5_failure"]),
                                    "stage": stage,
                                    "token_index": token_index,
                                    "evidence_score": float(evidence[token_index]),
                                    "bird_box_membership": (
                                        "inside" if bool(box_mask[token_index]) else "outside"
                                    ),
                                    "hidden_state_norm": float(norms[token_index]),
                                    "probe_seed_aggregation": "mean",
                                    "probe_seeds": json.dumps(list(seeds)),
                                }
                            )

    require(found == set(cohort_by_decision), "cache does not cover the selected Phase 7 cohort")
    require(set(stage_means) == set(stages), "replacement means do not cover both stages")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    token_path = args.output_dir / "token_metadata.csv"
    write_csv(token_path, output_rows)
    from safetensors.torch import save_file

    mean_path = args.output_dir / "development_train_stage_means.safetensors"
    temporary = mean_path.with_suffix(mean_path.suffix + ".tmp")
    save_file(
        {stage: tensor.contiguous() for stage, tensor in stage_means.items()},
        str(temporary),
        metadata={"source": "phase3r_training_feature_mean", "split": "train"},
    )
    temporary.replace(mean_path)
    cohort_path = args.output_dir / "causal_cohort.csv"
    write_csv(cohort_path, cohort)
    counts = Counter((int(row["attribute_id"]), int(row["phase5_failure"])) for row in cohort)
    report = {
        "schema_version": 1,
        "status": "PASS",
        "purpose": "phase7_bounded_matched_token_metadata",
        "selected_transition": transition["selected_transition"],
        "selected_stage": selected_stage,
        "neighbor_stage": neighbor_stage,
        "stage_order": list(CAUSAL_STAGE_ORDER),
        "decisions": len(cohort),
        "images": len({int(row["image_id"]) for row in cohort}),
        "attributes": len({int(row["attribute_id"]) for row in cohort}),
        "token_rows": len(output_rows),
        "max_per_outcome_per_attribute": args.max_per_outcome_per_attribute,
        "cohort_seed": args.cohort_seed,
        "probe_seeds": list(seeds),
        "outcome_counts_by_attribute": {
            f"{attribute_id}:{failure}": count
            for (attribute_id, failure), count in sorted(counts.items())
        },
        "artifacts": {
            token_path.name: sha256(token_path),
            cohort_path.name: sha256(cohort_path),
            mean_path.name: sha256(mean_path),
        },
        "phase6_transition_sha256": sha256(args.phase6_dir / "transition_decision.json"),
        "official_test_images_used": 0,
    }
    atomic_json_write(report, args.output_dir / "token_metadata_report.json")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
