#!/usr/bin/env python3
"""Render a fixed diverse Phase 4 gallery across every selected attribute group."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import torch


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from lger.dense_clip import FrozenDenseClip  # noqa: E402
from lger.phase4 import (  # noqa: E402
    join_cached_record,
    load_development_metadata,
    load_part_vocabulary,
    load_policy,
    read_json,
    require,
    sha256,
)
from lger.phase4_plots import plot_image  # noqa: E402
from lger.stage_cache import atomic_json_write  # noqa: E402


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    require(bool(rows), f"input CSV is empty: {path}")
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def select_pairs(eligibility: list[dict[str, str]], policy: dict[str, Any], per_group: int) \
        -> list[tuple[int, int, str, str]]:
    """Round-robin groups/attributes without consulting any measured score."""

    eligible = {(int(row["image_id"]), int(row["attribute_id"]))
                for row in eligibility if row["split"] == "val" and row["reason"] == "eligible"}
    groups: list[str] = []
    for attribute in policy["attributes"]:
        if attribute["group"] not in groups:
            groups.append(attribute["group"])
    output: list[tuple[int, int, str, str]] = []
    used_images: set[int] = set()
    for group in groups:
        attributes = [item for item in policy["attributes"] if item["group"] == group]
        chosen = 0
        for attribute in attributes:
            candidates = sorted(image_id for image_id, attribute_id in eligible
                                if attribute_id == int(attribute["attribute_id"])
                                and image_id not in used_images)
            if not candidates:
                continue
            image_id = candidates[0]
            output.append((image_id, int(attribute["attribute_id"]), group, attribute["name"]))
            used_images.add(image_id)
            chosen += 1
            if chosen == per_group:
                break
        require(chosen == per_group,
                f"attribute group {group} lacks {per_group} distinct eligible validation examples")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-cache", type=Path, required=True)
    parser.add_argument("--localizer-cache", type=Path, required=True)
    parser.add_argument("--phase4-dir", type=Path, required=True)
    parser.add_argument("--policy", type=Path,
                        default=PROJECT / "configs/phase4_localization.json")
    parser.add_argument("--part-vocabulary", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-group", type=int, default=2)
    args = parser.parse_args()
    require(args.per_group > 0, "per-group must be positive")

    policy = load_policy(args.policy)
    stage_config, records = load_development_metadata(args.stage_cache, policy)
    validation = read_json(args.stage_cache / "validation_report.json")
    localizer_config = read_json(args.localizer_cache / "extraction_config.json")
    phase4 = read_json(args.phase4_dir / "phase4_run_report.json")
    require(validation["status"] == phase4["status"] == "PASS"
            and validation["official_test_images"] == 0
            and phase4["official_test_images_used"] == 0
            and localizer_config.get("official_test_images_used", 0) == 0,
            "validated development-only caches and Phase 4 run are required")
    vocabulary = args.part_vocabulary or Path(stage_config["dataset"]["root"]) / "parts/parts.txt"
    part_names = load_part_vocabulary(vocabulary, policy)
    selection = select_pairs(
        read_csv(args.phase4_dir / "attribute_eligibility.csv"), policy, args.per_group
    )
    by_image = {int(record["image"]["image_id"]): record for record in records}
    require(all(image_id in by_image for image_id, *_ in selection),
            "selected gallery image is missing from the stage cache")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    scorer = None
    rows = []
    figures = []
    recomputed = 0
    for image_id, attribute_id, group, name in selection:
        record = by_image[image_id]
        pixels, cached_scores, source_hash = join_cached_record(args.localizer_cache, record)
        candidates = (
            args.phase4_dir / "dense_scores" / f"{image_id:05d}.pt",
            args.output_dir / "dense_scores" / f"{image_id:05d}.pt",
        )
        dense_path = next((path for path in candidates if path.is_file()), None)
        if dense_path is not None:
            saved = torch.load(dense_path, map_location="cpu", weights_only=True)
            require(int(saved["image_id"]) == image_id and saved["source_sha256"] == source_hash,
                    "retained dense score identity differs")
            dense = saved["scores"]
            dense_source = "retained_phase4_dense_scores"
        else:
            require(torch.cuda.is_available(),
                    "missing dense scores require the small gallery recomputation on Kaggle CUDA")
            if scorer is None:
                scorer = FrozenDenseClip.from_pretrained(policy, stage_config["spatial_preprocessing"])
            dense, diagnostics = scorer.score(pixels, validate_global=(recomputed == 0))
            dense_path = candidates[1]
            dense_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = dense_path.with_suffix(".pt.tmp")
            torch.save({"image_id": image_id, "source_sha256": source_hash,
                        "scores": dense, "diagnostics": diagnostics}, temporary)
            temporary.replace(dense_path)
            dense_source = "gallery_only_dense_clip_recomputation"
            recomputed += 1
        created = plot_image(
            args.output_dir, record, pixels, cached_scores, dense, policy, part_names,
            attribute_id=attribute_id, include_object=False,
        )
        figures.extend(created)
        rows.append({
            "image_id": image_id, "attribute_id": attribute_id,
            "attribute_group": group, "attribute_name": name,
            "selection_rule": "first_distinct_eligible_validation_image_in_policy_order",
            "dense_source": dense_source, "figure": created[0],
        })

    selection_path = args.output_dir / "gallery_selection.csv"
    write_csv(selection_path, rows)
    report = {
        "schema_version": 1, "status": "PASS",
        "purpose": "phase4_diverse_attribute_qualitative_gallery",
        "development_only": True, "official_test_images_used": 0,
        "selection_uses_performance_metrics": False,
        "attribute_groups": len({row["attribute_group"] for row in rows}),
        "panels": len(rows), "per_group": args.per_group,
        "dense_scores_recomputed": recomputed,
        "source_hashes": {
            "phase4_report": sha256(args.phase4_dir / "phase4_run_report.json"),
            "eligibility": sha256(args.phase4_dir / "attribute_eligibility.csv"),
            "policy": sha256(args.policy),
        },
        "artifacts": {
            selection_path.name: sha256(selection_path),
            **{name: sha256(args.output_dir / name) for name in figures},
        },
    }
    atomic_json_write(report, args.output_dir / "gallery_report.json")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
