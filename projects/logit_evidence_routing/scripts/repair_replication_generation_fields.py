#!/usr/bin/env python3
"""Recompute binary generation fields from retained text without model execution."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from lger.phase5 import strict_parse_binary  # noqa: E402
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
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def repair_rows(rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    required = {"decision_id", "target", "control", "generated_text",
                "parsed_answer", "generation_correct"}
    if not all(required <= set(row) for row in rows):
        raise RuntimeError(f"replication rows lack required fields: {sorted(required)}")
    repaired: list[dict[str, Any]] = []
    parsed_counts: Counter[str] = Counter()
    changed_rows = 0
    for source in rows:
        row: dict[str, Any] = dict(source)
        target = int(source["target"])
        if target not in (0, 1):
            raise RuntimeError("target must be binary")
        parsed = strict_parse_binary(source["generated_text"])
        parsed_value: bool | str = parsed if parsed is not None else ""
        correct: int | str = int(parsed == bool(target)) if parsed is not None else ""
        if "generation_parseable" in row:
            row["generation_parseable"] = int(parsed is not None)
        if str(row.get("parsed_answer", "")) != str(parsed_value) \
                or str(row.get("generation_correct", "")) != str(correct):
            changed_rows += 1
        row["parsed_answer"] = parsed_value
        row["generation_correct"] = correct
        parsed_counts["unparseable" if parsed is None else "yes" if parsed else "no"] += 1
        repaired.append(row)
    return repaired, {
        "rows": len(repaired),
        "changed_rows": changed_rows,
        "parsed_counts": dict(parsed_counts),
        "parseable_rows": len(repaired) - parsed_counts["unparseable"],
        "generation_correct_rows": sum(
            int(row["generation_correct"]) for row in repaired
            if row["generation_correct"] != ""
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs=2, action="append", metavar=("LABEL", "CSV"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest = read_csv(args.manifest)
    manifest_ids = {row["decision_id"] for row in manifest}
    if len(manifest_ids) != len(manifest):
        raise RuntimeError("manifest decision IDs are not unique")
    if any(row.get("official_split") != "train" or row.get("split") != "val"
           for row in manifest):
        raise RuntimeError("manifest must contain development validation rows from official train only")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports: dict[str, Any] = {}
    artifacts: dict[str, str] = {}
    source_hashes = {"manifest": sha256(args.manifest)}
    labels: set[str] = set()
    for label, raw_path in args.input:
        if label in labels:
            raise RuntimeError("input labels must be unique")
        labels.add(label)
        path = Path(raw_path)
        rows = read_csv(path)
        if {row["decision_id"] for row in rows} != manifest_ids:
            raise RuntimeError(f"{label} decisions do not exactly match the audited manifest")
        repaired, summary = repair_rows(rows)
        output = args.output_dir / f"{label}_replication_vqa_decisions.csv"
        write_csv(output, repaired)
        reports[label] = summary
        source_hashes[f"input_{label}"] = sha256(path)
        artifacts[output.name] = sha256(output)

    report = {
        "schema_version": 1,
        "status": "PASS",
        "purpose": "cpu_reanalysis_of_retained_free_generation_text",
        "model_extraction_performed": False,
        "development_only": True,
        "official_test_images_used": 0,
        "boolean_no_preserved": True,
        "models": reports,
        "source_hashes": source_hashes,
        "artifacts": artifacts,
    }
    atomic_json_write(report, args.output_dir / "generation_reanalysis_report.json")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
