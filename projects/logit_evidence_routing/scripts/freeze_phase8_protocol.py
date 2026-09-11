#!/usr/bin/env python3
"""Freeze the official-test protocol only after all development gates pass."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase3-dir", type=Path, required=True)
    parser.add_argument("--phase4-dir", type=Path, required=True)
    parser.add_argument("--phase5-dir", type=Path, required=True)
    parser.add_argument("--phase5-analysis-dir", type=Path, required=True)
    parser.add_argument("--phase6-dir", type=Path, required=True)
    parser.add_argument("--phase7-bridge-dir", type=Path, required=True)
    parser.add_argument("--phase7-plan", type=Path, required=True)
    parser.add_argument("--phase7-dir", type=Path, required=True)
    parser.add_argument("--phase7-analysis", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=PROJECT / "configs/phase8_final.json"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reports = {
        "phase3r": args.phase3_dir / "phase3_run_report.json",
        "phase4": args.phase4_dir / "phase4_validation_report.json",
        "phase5_extraction": args.phase5_dir / "phase5_run_report.json",
        "phase5_analysis": args.phase5_analysis_dir / "phase5_run_report.json",
        "phase6": args.phase6_dir / "phase6_run_report.json",
        "phase6_transition": args.phase6_dir / "transition_decision.json",
        "phase7_bridge": args.phase7_bridge_dir / "token_metadata_report.json",
        "phase7_plan": args.phase7_plan,
        "phase7_extraction": args.phase7_dir / "phase7_extraction_report.json",
        "phase7_analysis": args.phase7_analysis,
    }
    values = {name: read_json(path) for name, path in reports.items()}
    status_reports = (
        "phase3r", "phase4", "phase5_extraction", "phase5_analysis",
        "phase6", "phase7_bridge", "phase7_extraction",
    )
    if any(values[name].get("status") != "PASS" for name in status_reports):
        raise RuntimeError("every development execution/validation report must pass")
    if any(values[name].get("official_test_images_used", 0) != 0 for name in status_reports):
        raise RuntimeError("official test evidence was accessed before protocol freeze")
    transition = values["phase6_transition"].get("selected_transition")
    if transition == "mixed_or_null":
        raise RuntimeError("cannot freeze a causal confirmation for a mixed/null transition")
    if transition != values["phase6"].get("selected_transition"):
        raise RuntimeError("Phase 6 transition artifacts disagree")
    if values["phase7_plan"].get("decision_count") != values["phase7_analysis"].get("decision_count"):
        raise RuntimeError("Phase 7 plan and complete aggregation have different coverage")
    if values["phase7_plan"].get("intervention_count") != values["phase7_analysis"].get("outcome_count"):
        raise RuntimeError("Phase 7 plan does not have exact outcome coverage")
    if values["phase7_analysis"].get("bootstrap_samples") != 10000:
        raise RuntimeError("Phase 7 requires 10,000 development bootstrap samples before freeze")

    config = read_json(args.config)
    if config.get("schema_version") != 1 or config.get("test_access_requires_frozen_protocol") is not True:
        raise RuntimeError("unsupported Phase 8 policy")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "FROZEN_BEFORE_OFFICIAL_TEST_ACCESS",
        "purpose": config["purpose"],
        "development_selected_transition": transition,
        "selected_stage": values["phase7_plan"]["selected_stage"],
        "neighbor_stage": values["phase7_plan"]["neighbor_stage"],
        "selection_k": values["phase7_plan"]["k"],
        "replacement": values["phase7_plan"]["replacement"],
        "matched_random_seeds": values["phase7_plan"]["matched_random_seeds"],
        "phase8_config": config,
        "development_artifacts": {
            name: {"path": str(path.resolve()), "sha256": sha256(path)}
            for name, path in reports.items()
        },
        "official_test_images_used_at_freeze": 0,
        "allowed_confirmatory_claim": (
            "The predeclared selected-stage top-evidence intervention has a larger "
            "correct-answer margin effect than its matched-random control on held-out images."
        ),
        "null_policy": "retain and report a null-compatible result without changing the protocol",
    }
    payload["protocol_digest"] = canonical_digest(payload)
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output.exists():
        existing = read_json(args.output)
        if existing != payload:
            raise RuntimeError("a different Phase 8 protocol is already frozen at this path")
        print(json.dumps({"status": "ALREADY_FROZEN", "protocol_digest": payload["protocol_digest"]}))
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(serialized, encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({"status": payload["status"], "protocol_digest": payload["protocol_digest"]}))


if __name__ == "__main__":
    main()
