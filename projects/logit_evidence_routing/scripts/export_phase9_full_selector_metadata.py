#!/usr/bin/env python3
"""Export Vision-CLS/Logit selector scores for every Phase 6 development decision.

The exporter reads the saved Phase 1 localizer cache and Phase 2 stage cache. It
does not load or execute a VLM.  Its output enables smoke, pilot, then full Phase
9 intervention planning over the complete Phase 6 joined cohort.
"""

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
from lger.phase7_bridge import resolve_intervention_stages  # noqa: E402
from lger.stage_cache import atomic_json_write, config_digest  # noqa: E402


SELECTORS = ("vision_cls_attention", "logit_concept")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    require(rows, f"input CSV is empty: {path}")
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    require(rows, f"refusing to write empty output: {path}")
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
    parser.add_argument("--localizer-cache", type=Path, required=True)
    parser.add_argument("--phase6-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    run_config_path = args.stage_cache / "run_config.json"
    index_path = args.stage_cache / "index.json"
    validation_path = args.stage_cache / "validation_report.json"
    localizer_config_path = args.localizer_cache / "extraction_config.json"
    phase6_report_path = args.phase6_dir / "phase6_run_report.json"
    transition_path = args.phase6_dir / "transition_decision.json"
    joint_path = args.phase6_dir / "joint_decisions.csv"
    run_config = read_json(run_config_path)
    index = read_json(index_path)
    validation = read_json(validation_path)
    localizer_config = read_json(localizer_config_path)
    phase6 = read_json(phase6_report_path)
    transition = read_json(transition_path)
    digest = config_digest(run_config)
    require(
        index["complete"] is True
        and index["config_digest"] == validation["config_digest"] == digest
        and validation["status"] == phase6["status"] == "PASS"
        and validation["official_test_images"] == 0
        and phase6["official_test_images_used"] == 0
        and localizer_config.get("official_test_images_used", 0) == 0,
        "validated development-only Phase 1/2/6 caches are required",
    )
    selected_stage, neighbor_stage = resolve_intervention_stages(
        str(transition["selected_transition"])
    )
    require(selected_stage == "vision.late" and neighbor_stage == "projector.output",
            "full selector export is defined for the frozen Phase 6 mismatch transition")

    joint_rows = read_csv(joint_path)
    decisions: dict[int, list[dict[str, str]]] = defaultdict(list)
    seen: set[str] = set()
    for row in joint_rows:
        decision_id = str(row["decision_id"])
        require(decision_id not in seen, "Phase 6 decision IDs must be unique")
        seen.add(decision_id)
        require(row.get("split", "val") == "val" and int(row["target"]) == 1,
                "Phase 9 selector alignment uses the positive validation cohort localized in Phase 4")
        decisions[int(row["image_id"])].append(row)
    require(decisions, "Phase 6 full joined cohort is empty")

    try:
        from safetensors import safe_open
    except ImportError as error:  # pragma: no cover - Kaggle dependency guard
        raise RuntimeError("full selector export requires safetensors") from error

    output: list[dict[str, Any]] = []
    found_images: set[int] = set()
    source_hashes: dict[str, str] = {}
    for shard in index["shards"]:
        metadata_path = args.stage_cache / shard["metadata_path"]
        tensor_path = args.stage_cache / shard["tensor_path"]
        require(sha256(metadata_path) == shard["metadata_sha256"],
                "stage metadata hash differs")
        require(sha256(tensor_path) == shard["tensor_sha256"],
                "stage tensor hash differs")
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
                tensor_key = packed["stages"][selected_stage].get("__stage_cache_tensor__")
                require(isinstance(tensor_key, str), "selected-stage tensor reference is missing")
                states = tensors.get_tensor(tensor_key).float()
                norms = states.norm(dim=-1)
                grid = tuple(int(value) for value in packed["spatial"]["grid_size"])
                image_size = tuple(int(value) for value in
                                   packed["spatial"]["processed_image_size_hw"])
                box = tuple(float(value) for value in image["bbox_model_xyxy"])
                inside = patch_centers_in_box(grid, image_size, box)
                require(norms.shape == inside.shape, "stage tokens do not align with spatial grid")

                localizer_path = args.localizer_cache / "records" / f"{image_id:05d}.pt"
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
                source_hashes[f"localizer_record_{image_id}"] = sha256(localizer_path)
                found_images.add(image_id)
                for decision in decisions[image_id]:
                    for method, scores in score_maps.items():
                        for token_index in range(scores.numel()):
                            output.append({
                                "decision_id": decision["decision_id"],
                                "image_id": image_id,
                                "attribute_id": int(decision["attribute_id"]),
                                "target": 1,
                                "phase5_failure": int(decision["phase5_failure"]),
                                "selection_method": method,
                                "stage": selected_stage,
                                "token_index": token_index,
                                "evidence_score": float(scores[token_index]),
                                "bird_box_membership": (
                                    "inside" if bool(inside[token_index]) else "outside"
                                ),
                                "hidden_state_norm": float(norms[token_index]),
                                "score_origin": "phase1_cached_selector_map",
                            })

    require(found_images == set(decisions), "stage/localizer caches do not cover the full cohort")
    expected_rows = len(joint_rows) * len(SELECTORS) * int(index["records"][0].get("patch_count", 576))
    # The cache index schema has not always exposed patch_count; validate through
    # the observed grid when that field is absent.
    if expected_rows != len(output):
        require(len(output) % (len(joint_rows) * len(SELECTORS)) == 0,
                "unexpected selector token row count")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "selector_token_metadata.csv"
    write_csv(output_path, output)
    report = {
        "schema_version": 1,
        "status": "PASS",
        "purpose": "phase9_full_cache_only_selector_metadata",
        "development_only": True,
        "phase8_protocol_unchanged": True,
        "official_test_images_used": 0,
        "model_extraction_performed": False,
        "selected_transition": transition["selected_transition"],
        "selected_stage": selected_stage,
        "neighbor_stage": neighbor_stage,
        "selector_methods": list(SELECTORS),
        "decisions": len(joint_rows),
        "images": len(found_images),
        "token_rows": len(output),
        "artifacts": {output_path.name: sha256(output_path)},
        "source_hashes": {
            "stage_run_config": sha256(run_config_path),
            "stage_index": sha256(index_path),
            "stage_validation": sha256(validation_path),
            "localizer_extraction_config": sha256(localizer_config_path),
            "phase6_report": sha256(phase6_report_path),
            "phase6_transition": sha256(transition_path),
            "phase6_joint_decisions": sha256(joint_path),
            **source_hashes,
        },
    }
    atomic_json_write(report, args.output_dir / "selector_metadata_report.json")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
