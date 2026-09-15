#!/usr/bin/env python3
"""Find the largest K with exact bird-box and norm-matched random controls."""

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

from lger.phase9 import Phase9ValidationError, build_selector_intervention_plan  # noqa: E402
from lger.stage_cache import atomic_json_write  # noqa: E402


STAGE_ORDER = (
    "vision.early", "vision.middle", "vision.late", "projector.output",
    "llm.early", "llm.middle", "llm.late",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_selector_rows(path: Path, selector: str, stage: str) -> dict[int, list[dict[str, Any]]]:
    output: dict[int, list[dict[str, Any]]] = {0: [], 1: []}
    with path.open(newline="", encoding="utf-8") as handle:
        for source in csv.DictReader(handle):
            if source.get("selection_method") != selector or source.get("stage") != stage:
                continue
            row: dict[str, Any] = dict(source)
            row["token_index"] = int(row["token_index"])
            target = int(row["target"])
            if target not in output:
                raise RuntimeError("target must be binary")
            output[target].append(row)
    if not all(output.values()):
        raise RuntimeError("selector metadata lacks one target stratum")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selector", default="vision_cls_attention")
    parser.add_argument("--selected-stage", default="vision.late")
    parser.add_argument("--neighbor-stage", default="projector.output")
    parser.add_argument("--max-k", type=int, default=64)
    parser.add_argument("--min-k", type=int, default=32)
    parser.add_argument("--random-seeds", type=int, nargs="+", default=(0, 1, 2))
    parser.add_argument("--seed", type=int, default=909)
    parser.add_argument("--norm-quantiles", type=int, default=4)
    args = parser.parse_args()
    if args.min_k <= 0 or args.max_k < args.min_k:
        raise RuntimeError("invalid K search range")
    by_target = read_selector_rows(
        args.token_metadata, args.selector, args.selected_stage
    )
    attempts = []
    selected_k = None
    selected_quantiles: dict[str, int] = {}
    for k in range(args.max_k, args.min_k - 1, -1):
        target_results = {}
        passed = True
        for target in (0, 1):
            try:
                plan = build_selector_intervention_plan(
                    by_target[target], selected_stage=args.selected_stage,
                    neighbor_stage=args.neighbor_stage, stage_order=STAGE_ORDER,
                    k=k, selector_methods=(args.selector,),
                    random_seeds=args.random_seeds, seed=args.seed,
                    norm_quantiles=args.norm_quantiles, include_global_control=False,
                )
                resolved_quantiles = int(plan[
                    "matching_norm_quantiles_by_selector"
                ][args.selector])
                target_results[str(target)] = {
                    "status": "PASS",
                    "spatial_match": "exact_bird_box_membership",
                    "matching_norm_quantiles": resolved_quantiles,
                    "requested_norm_quantiles": args.norm_quantiles,
                    "exact_spatial_and_norm_pass": resolved_quantiles == args.norm_quantiles,
                }
                passed = passed and resolved_quantiles == args.norm_quantiles
            except Phase9ValidationError as error:
                passed = False
                target_results[str(target)] = {
                    "status": "FAIL",
                    "reason": str(error),
                }
        attempts.append({
            "k": k,
            "targets": target_results,
            "exact_spatial_and_norm_pass": passed,
        })
        print(f"K={k}: {'PASS' if passed else 'FAIL'}", flush=True)
        if passed:
            selected_k = k
            selected_quantiles = {
                target: int(value["matching_norm_quantiles"])
                for target, value in target_results.items()
            }
            break
    if selected_k is None:
        raise RuntimeError("no exact spatial-and-norm matched K exists in the search range")
    report = {
        "schema_version": 1,
        "status": "PASS",
        "purpose": "largest_exact_spatial_and_norm_matched_k_search",
        "development_only": True,
        "official_test_images_used": 0,
        "selector": args.selector,
        "selected_stage": args.selected_stage,
        "neighbor_stage": args.neighbor_stage,
        "max_k_requested": args.max_k,
        "min_k_requested": args.min_k,
        "largest_feasible_exact_k": selected_k,
        "norm_quantiles_by_target": selected_quantiles,
        "random_seeds": list(args.random_seeds),
        "search_order": "descending exhaustive until first pass",
        "attempts": attempts,
        "source_sha256": sha256(args.token_metadata),
        "model_forwards_performed": False,
    }
    atomic_json_write(report, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
