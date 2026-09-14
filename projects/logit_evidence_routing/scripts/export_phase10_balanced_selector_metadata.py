#!/usr/bin/env python3
"""Export cached selector scores for every balanced CUB development decision."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from lger.localization import patch_centers_in_box  # noqa: E402
from lger.phase4 import read_json, require  # noqa: E402
from lger.stage_cache import atomic_json_write, config_digest  # noqa: E402


SELECTORS = ("vision_cls_attention", "logit_concept")
SELECTED_STAGE = "vision.late"
NEIGHBOR_STAGE = "projector.output"


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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    require(bool(rows), "refusing to write empty selector metadata")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def resolve_localizer_cache(path: Path) -> Path:
    for candidate in (path, path / "cache"):
        if (candidate / "extraction_config.json").is_file() and (candidate / "records").is_dir():
            return candidate
    raise RuntimeError("corrected Phase 1B cache requires extraction_config.json and records/")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-cache", type=Path, required=True)
    parser.add_argument("--localizer-cache", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--behavior-decisions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    localizer_cache = resolve_localizer_cache(args.localizer_cache)
    run_config_path = args.stage_cache / "run_config.json"
    index_path = args.stage_cache / "index.json"
    validation_path = args.stage_cache / "validation_report.json"
    localizer_config_path = localizer_cache / "extraction_config.json"
    run_config = read_json(run_config_path)
    index = read_json(index_path)
    validation = read_json(validation_path)
    localizer_config = read_json(localizer_config_path)
    digest = config_digest(run_config)
    require(
        index["complete"] is True
        and index["config_digest"] == validation["config_digest"] == digest
        and validation["status"] == "PASS"
        and validation["official_test_images"] == 0
        and localizer_config.get("official_test_images_used", 0) == 0,
        "validated development-only Phase 1/2 caches are required",
    )

    manifest_rows = read_csv(args.manifest)
    decisions: dict[int, list[dict[str, Any]]] = defaultdict(list)
    seen: set[str] = set()
    target_counts = {0: 0, 1: 0}
    for row in manifest_rows:
        decision_id = str(row["decision_id"])
        require(decision_id not in seen, "manifest decision IDs must be unique")
        seen.add(decision_id)
        target = int(row["target"])
        require(row["split"] == "val" and row["official_split"] == "train"
                and target in (0, 1), "manifest must be development validation only")
        target_counts[target] += 1
        decisions[int(row["image_id"])].append({**row, "target": target})
    require(target_counts[0] == target_counts[1] and target_counts[0] > 0,
            "manifest must be target balanced")

    behavior_rows = read_csv(args.behavior_decisions)
    behavior = {}
    for row in behavior_rows:
        if row.get("control") == "image":
            decision_id = str(row["decision_id"])
            require(decision_id not in behavior, "duplicate correct-image behavior row")
            behavior[decision_id] = row
    require(set(behavior) == seen, "correct-image behavior rows must exactly cover the manifest")

    try:
        from safetensors import safe_open
    except ImportError as error:  # pragma: no cover - Kaggle dependency guard
        raise RuntimeError("balanced selector export requires safetensors") from error

    output: list[dict[str, Any]] = []
    found_images: set[int] = set()
    localizer_hashes: dict[str, str] = {}
    patch_count: int | None = None
    for shard in index["shards"]:
        metadata_path = args.stage_cache / shard["metadata_path"]
        tensor_path = args.stage_cache / shard["tensor_path"]
        require(sha256(metadata_path) == shard["metadata_sha256"], "stage metadata hash differs")
        require(sha256(tensor_path) == shard["tensor_sha256"], "stage tensor hash differs")
        metadata = read_json(metadata_path)
        with safe_open(tensor_path, framework="pt", device="cpu") as tensors:
            for item in metadata["records"]:
                packed = item["packed_record"]
                image = packed["image"]
                image_id = int(image["image_id"])
                if image_id not in decisions:
                    continue
                require(image["development_split"] == "val" and image["official_split"] == "train",
                        "official-test or non-validation image encountered")
                tensor_key = packed["stages"][SELECTED_STAGE].get("__stage_cache_tensor__")
                require(isinstance(tensor_key, str), "selected-stage tensor reference is missing")
                states = tensors.get_tensor(tensor_key).float()
                norms = states.norm(dim=-1)
                grid = tuple(int(value) for value in packed["spatial"]["grid_size"])
                image_size = tuple(int(value) for value in packed["spatial"]["processed_image_size_hw"])
                box = tuple(float(value) for value in image["bbox_model_xyxy"])
                inside = patch_centers_in_box(grid, image_size, box)
                require(norms.shape == inside.shape, "stage tokens do not align with the spatial grid")
                patch_count = int(norms.numel()) if patch_count is None else patch_count
                require(int(norms.numel()) == patch_count, "patch count changes across images")

                localizer_path = localizer_cache / "records" / f"{image_id:05d}.pt"
                require(localizer_path.is_file(), f"localizer cache is missing image {image_id}")
                localizer = torch.load(localizer_path, map_location="cpu", weights_only=False)
                require(int(localizer["image_id"]) == image_id, "localizer image identity differs")
                score_maps = {
                    method: torch.as_tensor(localizer["score_maps"][method]).float().flatten()
                    for method in SELECTORS
                }
                require(all(score.shape == norms.shape and bool(torch.isfinite(score).all())
                            for score in score_maps.values()),
                        "selector maps do not align with selected-stage tokens")
                localizer_hashes[f"localizer_record_{image_id}"] = sha256(localizer_path)
                found_images.add(image_id)
                for decision in decisions[image_id]:
                    behavior_row = behavior[str(decision["decision_id"])]
                    failure = 1 - int(behavior_row["margin_correct"])
                    for method, scores in score_maps.items():
                        for token_index in range(scores.numel()):
                            output.append({
                                "decision_id": decision["decision_id"],
                                "image_id": image_id,
                                "class_id": int(decision["class_id"]),
                                "attribute_id": int(decision["attribute_id"]),
                                "target": int(decision["target"]),
                                "phase5_failure": failure,
                                "selection_method": method,
                                "stage": SELECTED_STAGE,
                                "token_index": token_index,
                                "evidence_score": float(scores[token_index]),
                                "bird_box_membership": (
                                    "inside" if bool(inside[token_index]) else "outside"
                                ),
                                "hidden_state_norm": float(norms[token_index]),
                                "score_origin": "phase1_cached_selector_map",
                            })

    require(found_images == set(decisions), "stage/localizer caches do not cover the manifest")
    require(patch_count is not None and len(output) == len(manifest_rows) * len(SELECTORS) * patch_count,
            "unexpected balanced selector row count")
    output_path = args.output_dir / "selector_token_metadata.csv"
    write_csv(output_path, output)
    report = {
        "schema_version": 1,
        "status": "PASS",
        "purpose": "phase10_cache_only_balanced_selector_metadata",
        "development_only": True,
        "phase8_protocol_unchanged": True,
        "official_test_images_used": 0,
        "model_extraction_performed": False,
        "selected_stage": SELECTED_STAGE,
        "neighbor_stage": NEIGHBOR_STAGE,
        "selector_methods": list(SELECTORS),
        "decisions": len(manifest_rows),
        "target_counts": {str(key): value for key, value in target_counts.items()},
        "images": len(found_images),
        "patch_count": patch_count,
        "token_rows": len(output),
        "artifacts": {output_path.name: sha256(output_path)},
        "source_hashes": {
            "stage_run_config": sha256(run_config_path),
            "stage_index": sha256(index_path),
            "stage_validation": sha256(validation_path),
            "localizer_extraction_config": sha256(localizer_config_path),
            "manifest": sha256(args.manifest),
            "behavior_decisions": sha256(args.behavior_decisions),
            **localizer_hashes,
        },
    }
    atomic_json_write(report, args.output_dir / "selector_metadata_report.json")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
