#!/usr/bin/env python3
"""Select the frozen official-test image manifest after the Phase 8 freeze gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from lger.cub import load_cub_records  # noqa: E402


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-protocol", type=Path, required=True)
    parser.add_argument("--stage-cache", type=Path, required=True,
                        help="Development cache used only to recover the fixed pilot classes")
    parser.add_argument("--cub-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    frozen = json.loads(args.frozen_protocol.read_text(encoding="utf-8"))
    canonical = dict(frozen)
    recorded_digest = canonical.pop("protocol_digest", None)
    if frozen.get("status") != "FROZEN_BEFORE_OFFICIAL_TEST_ACCESS" or digest(canonical) != recorded_digest:
        raise RuntimeError("Phase 8 protocol is absent, unfrozen, or has changed")
    config = frozen["phase8_config"]
    if config["sampling_frame"] != "official_test_images_from_development_pilot_classes":
        raise RuntimeError("unsupported Phase 8 sampling frame")

    development_ids = {
        int(row["image_id"])
        for row in json.loads((args.stage_cache / "index.json").read_text())["records"]
    }
    records = load_cub_records(args.cub_root)
    by_id = {record.image_id: record for record in records}
    if not development_ids or any(by_id[image_id].official_split != "train" for image_id in development_ids):
        raise RuntimeError("development cache identity is not official-training-only")
    classes = sorted({by_id[image_id].label for image_id in development_ids})
    by_class: dict[int, list[object]] = defaultdict(list)
    for record in records:
        if record.official_split == "test" and record.label in classes:
            by_class[record.label].append(record)
    per_class = int(config["images_per_class"])
    rng = random.Random(int(config["sampling_seed"]))
    selected = []
    for class_id in classes:
        candidates = sorted(by_class[class_id], key=lambda row: row.image_id)
        if len(candidates) < per_class:
            raise RuntimeError(f"class {class_id} has fewer than {per_class} official-test images")
        selected.extend(rng.sample(candidates, per_class))
    selected.sort(key=lambda row: (row.label, row.image_id))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.output_dir / "official_test_manifest.csv"
    with manifest.with_suffix(".csv.tmp").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("image_id", "relative_path", "class_id", "class_name", "official_split"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows({
            "image_id": row.image_id,
            "relative_path": row.relative_path,
            "class_id": row.label,
            "class_name": row.class_name,
            "official_split": row.official_split,
        } for row in selected)
    manifest.with_suffix(".csv.tmp").replace(manifest)
    report = {
        "schema_version": 1,
        "status": "PASS",
        "protocol_digest": recorded_digest,
        "classes": len(classes),
        "images_per_class": per_class,
        "selected_official_test_images": len(selected),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "test_images_evaluated": 0,
    }
    temporary = args.output_dir / "manifest_report.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output_dir / "manifest_report.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
