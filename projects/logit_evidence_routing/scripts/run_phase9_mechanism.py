#!/usr/bin/env python3
"""Plan or aggregate the development-only Phase 9 mechanism disambiguation."""

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

from lger.phase9 import (  # noqa: E402
    Phase9ValidationError,
    aggregate_selector_interventions,
    build_selector_intervention_plan,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise Phase9ValidationError(f"input file does not exist: {path}")
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise Phase9ValidationError(f"input table is empty: {path}")
        for row in rows:
            if "token_index" in row:
                row["token_index"] = int(row["token_index"])
        return rows
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Phase9ValidationError(f"cannot read JSON input: {path}") from error
    rows = payload if isinstance(payload, list) else payload.get("records", payload.get("outcomes"))
    if not isinstance(rows, list) or not rows or not all(isinstance(row, dict) for row in rows):
        raise Phase9ValidationError("JSON input must be a nonempty record list")
    return rows


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Phase9ValidationError(f"cannot read JSON: {path}") from error
    if not isinstance(payload, dict):
        raise Phase9ValidationError("expected a JSON object")
    return payload


def atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--token-metadata", type=Path, required=True)
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--selected-stage", default="vision.late")
    plan.add_argument("--neighbor-stage", default="projector.output")
    plan.add_argument("--stage-order", nargs="+", default=[
        "vision.early", "vision.middle", "vision.late", "projector.output",
        "llm.early", "llm.middle", "llm.late",
    ])
    plan.add_argument("--selectors", nargs="+", default=[
        "vision_cls_attention", "logit_concept",
    ])
    plan.add_argument("--k", type=int, default=32)
    plan.add_argument("--random-seeds", type=int, nargs="+", default=[0, 1, 2])
    plan.add_argument("--seed", type=int, default=909)
    plan.add_argument("--norm-quantiles", type=int, default=4)
    plan.add_argument("--include-low-evidence", action="store_true")
    plan.add_argument("--omit-global-control", action="store_true")

    aggregate = commands.add_parser("aggregate")
    aggregate.add_argument("--outcomes", type=Path, required=True)
    aggregate.add_argument("--plan", type=Path, required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    aggregate.add_argument("--bootstrap-samples", type=int, default=10000)
    aggregate.add_argument("--confidence-level", type=float, default=0.95)
    aggregate.add_argument("--seed", type=int, default=20260913)
    return result


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    if args.command == "plan":
        payload = build_selector_intervention_plan(
            read_records(args.token_metadata),
            selected_stage=args.selected_stage,
            neighbor_stage=args.neighbor_stage,
            stage_order=args.stage_order,
            k=args.k,
            selector_methods=args.selectors,
            random_seeds=args.random_seeds,
            seed=args.seed,
            norm_quantiles=args.norm_quantiles,
            include_low_evidence=args.include_low_evidence,
            include_global_control=not args.omit_global_control,
        )
        payload["source_hashes"] = {"token_metadata": sha256(args.token_metadata)}
    else:
        payload = aggregate_selector_interventions(
            read_records(args.outcomes),
            plan=read_json(args.plan),
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
            confidence_level=args.confidence_level,
        )
        payload["source_hashes"] = {
            "outcomes": sha256(args.outcomes),
            "plan": sha256(args.plan),
        }
    atomic_write(args.output, payload)
    print(json.dumps({
        "status": "PASS",
        "command": args.command,
        "output": str(args.output.resolve()),
        "records": payload.get("intervention_count", payload.get("outcome_count")),
        "official_test_images_used": payload["official_test_images_used"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
