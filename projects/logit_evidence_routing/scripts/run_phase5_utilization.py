#!/usr/bin/env python3
"""Validate and aggregate precomputed Phase 5 utilization decision records."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
PHASE5_PATH = PROJECT / "src/lger/phase5.py"
SPEC = importlib.util.spec_from_file_location("_lger_phase5_cli", PHASE5_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load Phase 5 core: {PHASE5_PATH}")
PHASE5 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PHASE5)
Phase5ValidationError = PHASE5.Phase5ValidationError
validate_phase5_records = PHASE5.validate_phase5_records


def read_records(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.casefold()
    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    if suffix in {".jsonl", ".ndjson"}:
        records = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise Phase5ValidationError(f"JSONL line {line_number} is not an object")
            records.append(value)
        return records
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("records")
        if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
            raise Phase5ValidationError("JSON input must be a list or an object containing records")
        return payload
    raise Phase5ValidationError("records must be .csv, .json, .jsonl, or .ndjson")


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        raise Phase5ValidationError(f"refusing to write empty CSV: {path.name}")
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise Phase5ValidationError(f"inconsistent CSV schema: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def file_identity(path: Path) -> dict[str, Any]:
    return {
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True,
                        help="Precomputed .csv/.json/.jsonl decision-control measurements")
    parser.add_argument("--config", type=Path,
                        default=PROJECT / "configs/phase5_utilization.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    gate_path = args.output_dir / "phase5_gate.json"
    run_report_path = args.output_dir / "phase5_run_report.json"
    gate_path.unlink(missing_ok=True)
    run_report_path.unlink(missing_ok=True)
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        records = read_records(args.records)
        metrics, summaries, deltas, report = validate_phase5_records(records, config)
        outputs = {
            "decision_metrics.csv": metrics,
            "utilization_summary.csv": summaries,
            "paired_control_deltas.csv": deltas,
        }
        for name, rows in outputs.items():
            atomic_csv(rows, args.output_dir / name)
        report["input_records"] = str(args.records.resolve())
        report["input_records_sha256"] = hashlib.sha256(args.records.read_bytes()).hexdigest()
        report["config"] = str(args.config.resolve())
        report["config_sha256"] = hashlib.sha256(args.config.read_bytes()).hexdigest()
        report["artifact_manifest"] = {
            name: file_identity(args.output_dir / name) for name in outputs
        }
        atomic_json(report, run_report_path)
        atomic_json({
            "schema_version": 1,
            "status": "PASS",
            "protocol_digest": report["protocol_digest"],
            "decisions": report["decisions"],
            "measurement_rows": report["measurement_rows"],
            "official_test_split_untouched": True,
            "run_report_sha256": file_identity(run_report_path)["sha256"],
        }, gate_path)
    except Exception as exc:
        atomic_json({
            "schema_version": 1,
            "status": "FAIL",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "official_test_split_untouched": None,
        }, gate_path)
        raise
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
