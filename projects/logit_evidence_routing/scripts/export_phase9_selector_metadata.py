#!/usr/bin/env python3
"""Export cached Phase 6 selector maps for the matched Phase 7 development cohort.

This is a cache-only bridge: it reads the already extracted Phase 1 selector
maps and Phase 7 probe metadata.  It never loads or runs the VLM.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from lger.phase4 import read_json, require  # noqa: E402
from lger.stage_cache import atomic_json_write  # noqa: E402


SELECTORS = ("vision_cls_attention", "logit_concept", "probe")


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
    parser.add_argument("--localizer-cache", type=Path, required=True)
    parser.add_argument("--phase7-bridge-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    bridge_report_path = args.phase7_bridge_dir / "token_metadata_report.json"
    bridge = read_json(bridge_report_path)
    localizer_config_path = args.localizer_cache / "extraction_config.json"
    localizer_config = read_json(localizer_config_path)
    require(
        bridge["status"] == "PASS"
        and bridge["official_test_images_used"] == 0
        and localizer_config.get("official_test_images_used", 0) == 0,
        "development-only Phase 7 bridge and localizer cache are required",
    )
    selected_stage = str(bridge["selected_stage"])
    probe_rows = [
        row for row in read_csv(args.phase7_bridge_dir / "token_metadata.csv")
        if row["stage"] == selected_stage
    ]
    require(probe_rows, "Phase 7 probe token metadata does not cover the selected stage")

    by_decision: dict[str, list[dict[str, str]]] = {}
    for row in probe_rows:
        by_decision.setdefault(row["decision_id"], []).append(row)
    output: list[dict[str, Any]] = []
    cache_hashes: dict[int, str] = {}
    for decision_id, rows in sorted(by_decision.items()):
        identity = rows[0]
        require(all(
            row["image_id"] == identity["image_id"]
            and row["attribute_id"] == identity["attribute_id"]
            and row["target"] == identity["target"]
            and row["phase5_failure"] == identity["phase5_failure"]
            for row in rows
        ), f"probe metadata identity changes within {decision_id}")
        require(int(identity["target"]) == 1,
                "Phase 9 selector alignment is defined on Phase 4 eligible positive decisions")
        rows = sorted(rows, key=lambda row: int(row["token_index"]))
        image_id = int(identity["image_id"])
        cache_path = args.localizer_cache / "records" / f"{image_id:05d}.pt"
        require(cache_path.is_file(), f"cached selector record is missing: {cache_path}")
        saved = torch.load(cache_path, map_location="cpu", weights_only=False)
        require(int(saved["image_id"]) == image_id, "localizer cache image identity differs")
        score_maps = saved.get("score_maps", {})
        require(all(method in score_maps for method in SELECTORS[:2]),
                "localizer cache lacks the selected Phase 6 score maps")
        cache_hashes.setdefault(image_id, sha256(cache_path))

        scores = {
            "vision_cls_attention": torch.as_tensor(score_maps["vision_cls_attention"]).flatten(),
            "logit_concept": torch.as_tensor(score_maps["logit_concept"]).flatten(),
            "probe": torch.tensor([float(row["evidence_score"]) for row in rows]),
        }
        token_indices = [int(row["token_index"]) for row in rows]
        require(token_indices == list(range(len(rows))),
                "Phase 9 requires a complete zero-based patch-token universe")
        require(all(score.numel() == len(rows) and bool(torch.isfinite(score).all())
                    for score in scores.values()),
                "selector scores do not align with probe patch tokens")
        for method in SELECTORS:
            for position, row in enumerate(rows):
                output.append({
                    "decision_id": decision_id,
                    "image_id": image_id,
                    "attribute_id": int(identity["attribute_id"]),
                    "target": 1,
                    "phase5_failure": int(identity["phase5_failure"]),
                    "selection_method": method,
                    "stage": selected_stage,
                    "token_index": position,
                    "evidence_score": float(scores[method][position]),
                    "bird_box_membership": row["bird_box_membership"],
                    "hidden_state_norm": float(row["hidden_state_norm"]),
                    "score_origin": (
                        "phase3r_mean_probe_margin" if method == "probe"
                        else "phase1_cached_selector_map"
                    ),
                })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "selector_token_metadata.csv"
    write_csv(output_path, output)
    report = {
        "schema_version": 1,
        "status": "PASS",
        "purpose": "phase9_cache_only_selector_metadata",
        "development_only": True,
        "phase8_protocol_unchanged": True,
        "official_test_images_used": 0,
        "model_extraction_performed": False,
        "selected_stage": selected_stage,
        "selector_methods": list(SELECTORS),
        "decisions": len(by_decision),
        "images": len(cache_hashes),
        "token_rows": len(output),
        "artifacts": {output_path.name: sha256(output_path)},
        "source_hashes": {
            "phase7_bridge_report": sha256(bridge_report_path),
            "localizer_extraction_config": sha256(localizer_config_path),
            "localizer_records": {str(key): value for key, value in sorted(cache_hashes.items())},
        },
    }
    atomic_json_write(report, args.output_dir / "selector_metadata_report.json")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
