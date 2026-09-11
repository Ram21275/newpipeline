#!/usr/bin/env python3
"""Run the frozen Phase 7 hidden-state interventions on Kaggle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

import torch
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from lger.hf_utilization import HiddenIntervention, HfLlavaDecisionRunner  # noqa: E402
from lger.phase4 import read_json, require  # noqa: E402
from lger.phase5 import strict_parse_binary  # noqa: E402
from lger.stage_cache import atomic_json_write, config_digest, write_or_validate_config  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    require(bool(rows), f"input CSV is empty: {path}")
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    require(bool(rows), f"refusing to write empty table: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def generated_correct(text: str | None, target: int) -> bool:
    parsed = strict_parse_binary(text or "")
    return parsed is not None and int(parsed) == target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "development"), required=True)
    parser.add_argument(
        "--config", type=Path, default=PROJECT / "configs/phase5_utilization.json"
    )
    parser.add_argument(
        "--phase7-config", type=Path, default=PROJECT / "configs/phase7_intervention.json"
    )
    parser.add_argument("--phase5-dir", type=Path, required=True)
    parser.add_argument("--phase7-bridge-dir", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--cub-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke-dir", type=Path)
    parser.add_argument("--baseline-tolerance", type=float, default=5e-3)
    args = parser.parse_args()

    config = read_json(args.config)
    phase7_config = read_json(args.phase7_config)
    phase5 = read_json(args.phase5_dir / "phase5_run_report.json")
    bridge = read_json(args.phase7_bridge_dir / "token_metadata_report.json")
    plan = read_json(args.plan)
    require(
        phase5["status"] == bridge["status"] == "PASS"
        and phase5["mode"] == "development"
        and phase5["official_test_images_used"] == bridge["official_test_images_used"] == 0
        and plan["purpose"] == "phase7_targeted_causal_intervention_plan"
        and plan["replacement"] == "development_train_stage_mean"
        and plan["selected_stage"] == bridge["selected_stage"]
        and plan["neighbor_stage"] == bridge["neighbor_stage"]
        and plan["decision_count"] == bridge["decisions"]
        and plan["k"] == phase7_config["selection_k"]
        and plan["replacement"] == phase7_config["replacement"]
        and plan["matched_random_seeds"] == phase7_config["matched_random_seeds"]
        and phase7_config["official_test_images_used"] == 0,
        "Phase 7 requires compatible development-only Phase 5/bridge/plan artifacts",
    )
    manifest_rows = read_csv(args.phase5_dir / "decision_manifest.csv")
    manifest = {row["decision_id"]: row for row in manifest_rows}
    phase5_rows = {
        row["decision_id"]: row
        for row in read_csv(args.phase5_dir / "phase5_decisions.csv")
        if row["control"] == "image"
    }
    require(
        all(record["decision_id"] in manifest and record["decision_id"] in phase5_rows
            for record in plan["records"]),
        "Phase 7 plan contains a decision outside Phase 5",
    )
    require(args.cub_root.is_dir(), "CUB root does not exist")

    try:
        from safetensors import safe_open
    except ImportError as error:  # pragma: no cover - Kaggle dependency guard
        raise RuntimeError("Phase 7 extraction requires safetensors") from error
    mean_path = args.phase7_bridge_dir / "development_train_stage_means.safetensors"
    means: dict[str, torch.Tensor] = {}
    with safe_open(mean_path, framework="pt", device="cpu") as handle:
        for stage in handle.keys():
            means[stage] = handle.get_tensor(stage).float()
    require(
        set(means) == {plan["selected_stage"], plan["neighbor_stage"]},
        "replacement means do not match the planned stages",
    )

    policy = {
        "schema_version": 1,
        "purpose": "phase7_causal_intervention_extraction",
        "model": config["model"],
        "revision": config["revision"],
        "quantization": config["quantization"],
        "phase5_report_sha256": sha256(args.phase5_dir / "phase5_run_report.json"),
        "bridge_report_sha256": sha256(args.phase7_bridge_dir / "token_metadata_report.json"),
        "plan_sha256": sha256(args.plan),
        "phase7_config_sha256": sha256(args.phase7_config),
        "means_sha256": sha256(mean_path),
        "baseline_tolerance": args.baseline_tolerance,
        "official_test_images_used": 0,
    }
    policy_digest = config_digest(policy)
    records = list(plan["records"])
    if args.mode == "smoke":
        records = records[:1]
    else:
        require(args.smoke_dir is not None, "--smoke-dir is required for development mode")
        smoke = read_json(args.smoke_dir / "phase7_extraction_report.json")
        require(
            smoke["status"] == "PASS"
            and smoke["mode"] == "smoke"
            and smoke["policy_digest"] == policy_digest,
            "matching Phase 7 smoke is required",
        )

    identity = {**policy, "mode": args.mode, "intervention_ids": [r["intervention_id"] for r in records]}
    protocol_digest = config_digest(identity)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_or_validate_config(args.output_dir / "evaluation_config.json", identity)
    (args.output_dir / "phase7_extraction_report.json").unlink(missing_ok=True)
    output_record_dir = args.output_dir / "records"
    output_record_dir.mkdir(exist_ok=True)

    runner = HfLlavaDecisionRunner.from_pretrained(
        str(config["model"]),
        revision=str(config["revision"]),
        quantization=str(config["quantization"]),
        positive_answer=str(config["positive_answer"]),
        negative_answer=str(config["negative_answer"]),
        attention_layer_offset=int(config["attention_layer_offset"]),
    )
    require(runner.resolved_revision == config["revision"], "resolved model revision differs")
    baselines: dict[str, object] = {}
    output_rows: list[dict[str, object]] = []
    start = time.monotonic()
    for position, record in enumerate(records, 1):
        destination = output_record_dir / f"{record['intervention_id']}.json"
        if destination.is_file():
            saved = read_json(destination)
            require(
                saved["protocol_digest"] == protocol_digest
                and saved["intervention_id"] == record["intervention_id"],
                "Phase 7 resume identity differs",
            )
            output_rows.append(saved["row"])
            continue
        try:
            decision = manifest[record["decision_id"]]
            target = int(decision["target"])
            image_path = args.cub_root / "images" / decision["relative_path"]
            with Image.open(image_path) as opened:
                image = opened.convert("RGB")
                baseline = baselines.get(record["decision_id"])
                if baseline is None:
                    baseline = runner.evaluate(
                        image,
                        decision["prompt_text"],
                        include_attention=False,
                        generate=True,
                        max_new_tokens=int(config["max_new_tokens"]),
                    )
                    baselines[record["decision_id"]] = baseline
                    expected = float(phase5_rows[record["decision_id"]]["correct_answer_margin"])
                    actual = baseline.answer_margin if target else -baseline.answer_margin
                    require(
                        abs(actual - expected) <= args.baseline_tolerance,
                        "Phase 7 baseline does not reproduce Phase 5 within tolerance",
                    )
                intervention = HiddenIntervention(
                    stage=record["stage"],
                    patch_indices=torch.tensor(record["token_indices"], dtype=torch.long),
                    replacement=means[record["stage"]],
                )
                changed = runner.evaluate(
                    image,
                    decision["prompt_text"],
                    include_attention=False,
                    generate=True,
                    max_new_tokens=int(config["max_new_tokens"]),
                    intervention=intervention,
                )
            before_margin = baseline.answer_margin if target else -baseline.answer_margin
            after_margin = changed.answer_margin if target else -changed.answer_margin
            row = {
                "decision_id": record["decision_id"],
                "image_id": record["image_id"],
                "attribute_id": int(decision["attribute_id"]),
                "target": target,
                "intervention_id": record["intervention_id"],
                "stage": record["stage"],
                "stage_role": record["stage_role"],
                "intervention_type": record["intervention_type"],
                "replicate": record["replicate"],
                "correct_answer_teacher_forced_margin_before": before_margin,
                "correct_answer_teacher_forced_margin_after": after_margin,
                "generated_correct_before": generated_correct(baseline.generated_text, target),
                "generated_correct_after": generated_correct(changed.generated_text, target),
                "generated_text_before": baseline.generated_text or "",
                "generated_text_after": changed.generated_text or "",
            }
            atomic_json_write(
                {
                    "schema_version": 1,
                    "protocol_digest": protocol_digest,
                    "intervention_id": record["intervention_id"],
                    "row": row,
                },
                destination,
            )
            output_rows.append(row)
        except Exception as error:
            atomic_json_write(
                {
                    "schema_version": 1,
                    "status": "FAIL",
                    "mode": args.mode,
                    "protocol_digest": protocol_digest,
                    "failed_position": position,
                    "failed_intervention_id": record["intervention_id"],
                    "completed_interventions": len(output_rows),
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                    "resumable": True,
                    "official_test_images_used": 0,
                },
                args.output_dir / "phase7_failure_report.json",
            )
            raise
        print(
            f"[{position}/{len(records)}] {record['intervention_id']} "
            f"elapsed={time.monotonic() - start:.1f}s",
            flush=True,
        )

    require(len(output_rows) == len(records), "Phase 7 outputs are incomplete")
    output_path = args.output_dir / "intervention_outcomes.csv"
    write_csv(output_path, output_rows)
    report = {
        "schema_version": 1,
        "status": "PASS",
        "mode": args.mode,
        "policy_digest": policy_digest,
        "protocol_digest": protocol_digest,
        "interventions": len(records),
        "decisions": len({row["decision_id"] for row in records}),
        "selected_stage": plan["selected_stage"],
        "neighbor_stage": plan["neighbor_stage"],
        "runtime_seconds": time.monotonic() - start,
        "outcomes_sha256": sha256(output_path),
        "official_test_images_used": 0,
    }
    atomic_json_write(report, args.output_dir / "phase7_extraction_report.json")
    (args.output_dir / "phase7_failure_report.json").unlink(missing_ok=True)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
