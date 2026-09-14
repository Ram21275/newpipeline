#!/usr/bin/env python3
"""Plan or aggregate the balanced-target and K-dose development extension."""

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

from lger.phase10 import aggregate_priority1, build_priority1_plan  # noqa: E402
from lger.stage_cache import atomic_json_write  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"input CSV is empty: {path}")
    # csv.DictReader serializes every field as text.  Phase 7's audited matcher
    # intentionally requires token indices to be real integers, so normalize at
    # this file boundary just as the Phase 9 CLI does.
    for row in rows:
        if "token_index" in row:
            try:
                row["token_index"] = int(row["token_index"])
            except (TypeError, ValueError) as error:
                raise RuntimeError(f"token_index must be an integer in {path}") from error
    return rows


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--token-metadata", type=Path, required=True)
    plan.add_argument("--config", type=Path, default=PROJECT / "configs/phase10_priority1.json")
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--seed", type=int, default=909)
    plan.add_argument("--norm-quantiles", type=int, default=4)

    aggregate = commands.add_parser("aggregate")
    aggregate.add_argument("--outcomes", type=Path, required=True)
    aggregate.add_argument("--plan", type=Path, required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    aggregate.add_argument("--bootstrap-samples", type=int, default=10000)
    aggregate.add_argument("--confidence-level", type=float, default=0.95)
    aggregate.add_argument("--seed", type=int, default=20260914)
    return result


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    if args.command == "plan":
        config = json.loads(args.config.read_text(encoding="utf-8"))
        if not (config.get("development_only") is True
                and config.get("phase8_protocol_unchanged") is True
                and config.get("official_test_images_used") == 0):
            raise RuntimeError("Priority 1 config must remain development-only")
        payload = build_priority1_plan(
            read_csv(args.token_metadata),
            selected_stage=str(config["selected_stage"]),
            neighbor_stage=str(config["neighbor_stage"]),
            stage_order=["vision.early", "vision.middle", "vision.late", "projector.output",
                         "llm.early", "llm.middle", "llm.late"],
            selector_methods=config["selector_methods"],
            balanced_k=int(config["balanced_k"]),
            dose_selector=str(config["dose_selector"]),
            dose_k_values=config["dose_k_values"],
            random_seeds=config["matched_random_seeds"],
            seed=args.seed,
            norm_quantiles=args.norm_quantiles,
        )
        payload["source_hashes"] = {
            "token_metadata": sha256(args.token_metadata),
            "config": sha256(args.config),
        }
    else:
        plan = json.loads(args.plan.read_text(encoding="utf-8"))
        payload = aggregate_priority1(
            read_csv(args.outcomes), plan=plan,
            bootstrap_samples=args.bootstrap_samples, seed=args.seed,
            confidence_level=args.confidence_level,
        )
        payload["source_hashes"] = {
            "outcomes": sha256(args.outcomes),
            "plan": sha256(args.plan),
        }
    atomic_json_write(payload, args.output)
    print(json.dumps({
        "status": payload["status"], "command": args.command,
        "output": str(args.output.resolve()),
        "decisions": payload.get("decision_count"),
        "records": payload.get("intervention_count", payload.get("outcome_count")),
        "official_test_images_used": payload["official_test_images_used"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
