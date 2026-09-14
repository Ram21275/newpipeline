#!/usr/bin/env python3
"""Run Priority 0 inference and influence checks on retained CUB artifacts."""

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

from lger.paper_robustness import (  # noqa: E402
    group_robustness,
    image_influence,
    phase9_family_rows,
    replication_family_rows,
    simultaneous_image_bootstrap,
)
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
        raise RuntimeError(f"refusing to write empty output: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase9-outcomes", type=Path, required=True)
    parser.add_argument("--phase9-plan", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--replication", nargs=2, action="append", metavar=("LABEL", "CSV"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260914)
    args = parser.parse_args()

    manifest = read_csv(args.manifest)
    plan = json.loads(args.phase9_plan.read_text(encoding="utf-8"))
    phase9_rows = phase9_family_rows(read_csv(args.phase9_outcomes), plan, manifest)
    replication_paths = {label: Path(path) for label, path in args.replication}
    if len(replication_paths) != len(args.replication):
        raise RuntimeError("replication labels must be unique")
    replication_rows = replication_family_rows(
        {label: read_csv(path) for label, path in replication_paths.items()}, manifest
    )

    simultaneous = simultaneous_image_bootstrap(
        phase9_rows, samples=args.bootstrap_samples, seed=args.seed,
        confidence_level=args.confidence_level,
    )
    phase9_groups = group_robustness(
        phase9_rows, samples=args.bootstrap_samples, seed=args.seed,
        confidence_level=args.confidence_level,
    )
    replication_groups = group_robustness(
        replication_rows, samples=args.bootstrap_samples, seed=args.seed + 1,
        confidence_level=args.confidence_level,
    )
    per_image, influence = image_influence(phase9_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "phase9_decision_effects.csv": phase9_rows,
        "phase9_simultaneous_inference.csv": simultaneous,
        "phase9_group_robustness.csv": phase9_groups,
        "phase9_per_image_effects.csv": per_image,
        "phase9_influence_summary.csv": influence,
        "replication_decision_effects.csv": replication_rows,
        "replication_group_robustness.csv": replication_groups,
    }
    for name, rows in tables.items():
        write_csv(args.output_dir / name, rows)
    source_hashes = {
        "phase9_outcomes": sha256(args.phase9_outcomes),
        "phase9_plan": sha256(args.phase9_plan),
        "manifest": sha256(args.manifest),
        **{f"replication_{label}": sha256(path) for label, path in replication_paths.items()},
    }
    report = {
        "schema_version": 1,
        "status": "PASS",
        "purpose": "priority0_submission_robustness",
        "development_only": True,
        "official_test_images_used": 0,
        "model_extraction_performed": False,
        "bootstrap_samples": args.bootstrap_samples,
        "confidence_level": args.confidence_level,
        "simultaneous_method": "common-image nonparametric max-t bootstrap",
        "group_method": "equal-group estimate with nonparametric group bootstrap",
        "phase9_family_size": len({row["estimand"] for row in phase9_rows}),
        "replication_family_size": len({row["estimand"] for row in replication_rows}),
        "source_hashes": source_hashes,
        "artifacts": {name: sha256(args.output_dir / name) for name in tables},
    }
    atomic_json_write(report, args.output_dir / "priority0_robustness_report.json")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
