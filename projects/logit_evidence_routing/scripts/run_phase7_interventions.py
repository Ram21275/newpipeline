#!/usr/bin/env python3
"""Plan Phase 7 interventions or aggregate precomputed intervention outcomes."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from lger.phase7 import (  # noqa: E402
    Phase7ValidationError,
    aggregate_intervention_outcomes,
    build_intervention_plan,
)


def read_records(path: Path) -> list[dict[str, Any]]:
    """Read records from JSON, JSON Lines, or CSV without schema coercion."""

    if not path.is_file():
        raise Phase7ValidationError(f"input file does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise Phase7ValidationError(f"input table is empty: {path}")
        for row in rows:
            if "token_index" in row:
                try:
                    row["token_index"] = int(row["token_index"])
                except ValueError as error:
                    raise Phase7ValidationError("CSV token_index must be an integer") from error
        return rows
    if suffix in (".jsonl", ".ndjson"):
        rows = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise Phase7ValidationError(f"invalid JSON on line {line_number}: {path}") from error
            if not isinstance(row, dict):
                raise Phase7ValidationError("JSON Lines entries must be objects")
            rows.append(row)
        if not rows:
            raise Phase7ValidationError(f"input table is empty: {path}")
        return rows
    if suffix != ".json":
        raise Phase7ValidationError("input must be .json, .jsonl/.ndjson, or .csv")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise Phase7ValidationError(f"invalid JSON: {path}") from error
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = payload.get("records", payload.get("outcomes"))
    else:
        rows = None
    if not isinstance(rows, list) or not rows or not all(isinstance(row, dict) for row in rows):
        raise Phase7ValidationError("JSON input must be a nonempty record list or contain records/outcomes")
    return rows


def read_plan(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Phase7ValidationError(f"cannot read intervention plan: {path}") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
        raise Phase7ValidationError("plan JSON must contain records")
    return payload


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="generate a deterministic intervention plan")
    plan.add_argument("--token-metadata", type=Path, required=True)
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--selected-stage", required=True)
    plan.add_argument("--neighbor-stage", required=True)
    plan.add_argument("--stage-order", nargs="+", required=True)
    plan.add_argument("--k", type=int, required=True)
    plan.add_argument("--seed", type=int, default=0)
    plan.add_argument("--random-seeds", type=int, nargs="+", default=[0, 1, 2])
    plan.add_argument("--norm-quantiles", type=int, default=4)

    aggregate = commands.add_parser("aggregate", help="aggregate precomputed intervention outcomes")
    aggregate.add_argument("--outcomes", type=Path, required=True)
    aggregate.add_argument("--plan", type=Path)
    aggregate.add_argument("--output", type=Path, required=True)
    aggregate.add_argument("--bootstrap-samples", type=int, default=10000)
    aggregate.add_argument("--confidence-level", type=float, default=0.95)
    aggregate.add_argument("--seed", type=int, default=314159)
    return result


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    if args.command == "plan":
        payload = build_intervention_plan(
            read_records(args.token_metadata),
            selected_stage=args.selected_stage,
            neighbor_stage=args.neighbor_stage,
            stage_order=args.stage_order,
            k=args.k,
            seed=args.seed,
            random_replicates=len(args.random_seeds),
            random_seeds=args.random_seeds,
            norm_quantiles=args.norm_quantiles,
        )
    else:
        plan = read_plan(args.plan) if args.plan else None
        payload = aggregate_intervention_outcomes(
            read_records(args.outcomes),
            plan=plan,
            bootstrap_samples=args.bootstrap_samples,
            confidence_level=args.confidence_level,
            seed=args.seed,
        )
    atomic_write_json(args.output, payload)
    print(json.dumps({
        "status": "PASS",
        "command": args.command,
        "output": str(args.output.resolve()),
        "records": payload.get("intervention_count", payload.get("outcome_count")),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
