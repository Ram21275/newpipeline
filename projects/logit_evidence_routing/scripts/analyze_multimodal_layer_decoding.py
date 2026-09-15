#!/usr/bin/env python3
"""Analyze paired DeCo, DoLa, final-layer, and visual-control results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from lger.layer_decoding_analysis import analyze_layer_decoding  # noqa: E402
from lger.stage_cache import atomic_json_write  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"input CSV is empty: {path}")
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write an empty table: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260915)
    args = parser.parse_args()

    result = analyze_layer_decoding(
        read_csv(args.input),
        samples=args.bootstrap_samples,
        seed=args.seed,
        confidence_level=args.confidence_level,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "multimodal_layer_decoding_summary.csv": result.pop("summary"),
        "multimodal_layer_decoding_paired_inference.csv": result.pop(
            "paired_inference"
        ),
        "multimodal_layer_decoding_visual_specificity.csv": result.pop(
            "visual_specificity_inference"
        ),
        "multimodal_layer_selection.csv": result.pop("layer_selection"),
    }
    for name, table in tables.items():
        write_csv(args.output_dir / name, table)
    result["source_sha256"] = sha256(args.input)
    result["artifacts"] = {
        name: sha256(args.output_dir / name) for name in tables
    }
    atomic_json_write(
        result, args.output_dir / "multimodal_layer_decoding_analysis.json"
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "decisions": result["decision_count"],
                "images": result["image_count"],
                "official_test_images_used": result["official_test_images_used"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
