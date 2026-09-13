#!/usr/bin/env python3
"""Run smoke, pilot, or full LLaVA Phase 9 selector interventions on Kaggle."""

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
from lger.phase9 import PURPOSE_PLAN  # noqa: E402
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
    require(rows, f"refusing to write empty output: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def choose_records(
    records: list[dict[str, object]], mode: str, pilot_decisions: int
) -> list[dict[str, object]]:
    decision_ids = sorted({str(row["decision_id"]) for row in records})
    if mode == "smoke":
        chosen = set(decision_ids[:1])
    elif mode == "pilot":
        require(pilot_decisions > 1, "pilot_decisions must exceed the one-decision smoke")
        chosen = set(decision_ids[:pilot_decisions])
    else:
        chosen = set(decision_ids)
    return [row for row in records if str(row["decision_id"]) in chosen]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "pilot", "full"), required=True)
    parser.add_argument("--phase5-config", type=Path,
                        default=PROJECT / "configs/phase5_utilization.json")
    parser.add_argument("--phase9-config", type=Path,
                        default=PROJECT / "configs/phase9_mechanism.json")
    parser.add_argument("--phase5-dir", type=Path, required=True)
    parser.add_argument("--phase7-bridge-dir", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--cub-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke-dir", type=Path)
    parser.add_argument("--pilot-dir", type=Path)
    parser.add_argument("--pilot-decisions", type=int, default=20)
    parser.add_argument("--baseline-tolerance", type=float, default=5e-3)
    args = parser.parse_args()

    model_config = read_json(args.phase5_config)
    phase9_config = read_json(args.phase9_config)
    phase5 = read_json(args.phase5_dir / "phase5_run_report.json")
    bridge = read_json(args.phase7_bridge_dir / "token_metadata_report.json")
    plan = read_json(args.plan)
    require(
        phase5["status"] == bridge["status"] == plan["status"] == "PASS"
        and phase5["official_test_images_used"] == bridge["official_test_images_used"]
        == plan["official_test_images_used"] == phase9_config["official_test_images_used"] == 0
        and plan["purpose"] == PURPOSE_PLAN
        and plan["development_only"] is True
        and plan["phase8_protocol_unchanged"] is True
        and phase9_config["phase8_protocol_unchanged"] is True
        and plan["selected_stage"] == bridge["selected_stage"]
        and plan["neighbor_stage"] == bridge["neighbor_stage"]
        and plan["selection_k"] == phase9_config["selection_k"]
        and plan["selector_methods"] == phase9_config["selector_methods"]
        and plan["replacement"] == phase9_config["replacement"]
        and plan["matched_random_seeds"] == phase9_config["matched_random_seeds"],
        "compatible development-only Phase 5/7/9 inputs are required",
    )
    require(args.cub_root.is_dir(), "CUB root does not exist")
    manifest = {row["decision_id"]: row for row in read_csv(args.phase5_dir / "decision_manifest.csv")}
    phase5_image = {
        row["decision_id"]: row
        for row in read_csv(args.phase5_dir / "phase5_decisions.csv")
        if row["control"] == "image"
    }
    require(all(str(row["decision_id"]) in manifest and str(row["decision_id"]) in phase5_image
                for row in plan["records"]),
            "Phase 9 plan includes a decision absent from Phase 5")

    try:
        from safetensors import safe_open
    except ImportError as error:  # pragma: no cover - Kaggle dependency guard
        raise RuntimeError("Phase 9 extraction requires safetensors") from error
    mean_path = args.phase7_bridge_dir / "development_train_stage_means.safetensors"
    means: dict[str, torch.Tensor] = {}
    with safe_open(mean_path, framework="pt", device="cpu") as handle:
        for stage in handle.keys():
            means[stage] = handle.get_tensor(stage).float()
    require(set(means) == {plan["selected_stage"], plan["neighbor_stage"]},
            "replacement means do not match Phase 9 stages")

    policy = {
        "schema_version": 1,
        "purpose": "phase9_llava_selector_intervention_extraction",
        "development_only": True,
        "phase8_protocol_unchanged": True,
        "model": model_config["model"],
        "revision": model_config["revision"],
        "quantization": model_config["quantization"],
        "phase5_report_sha256": sha256(args.phase5_dir / "phase5_run_report.json"),
        "bridge_report_sha256": sha256(args.phase7_bridge_dir / "token_metadata_report.json"),
        "plan_sha256": sha256(args.plan),
        "phase9_config_sha256": sha256(args.phase9_config),
        "means_sha256": sha256(mean_path),
        "baseline_tolerance": args.baseline_tolerance,
        "generated_outcome_computed": False,
        "official_test_images_used": 0,
    }
    policy_digest = config_digest(policy)

    if args.mode in ("pilot", "full"):
        require(args.smoke_dir is not None, "--smoke-dir is required after smoke mode")
        smoke = read_json(args.smoke_dir / "phase9_extraction_report.json")
        require(smoke["status"] == "PASS" and smoke["mode"] == "smoke"
                and smoke["policy_digest"] == policy_digest,
                "a passing matching Phase 9 smoke is required")
    if args.mode == "full":
        require(args.pilot_dir is not None, "--pilot-dir is required for the full run")
        pilot = read_json(args.pilot_dir / "phase9_extraction_report.json")
        require(pilot["status"] == "PASS" and pilot["mode"] == "pilot"
                and pilot["policy_digest"] == policy_digest,
                "a passing matching Phase 9 pilot is required")

    records = choose_records(list(plan["records"]), args.mode, args.pilot_decisions)
    identity = {
        **policy,
        "mode": args.mode,
        "pilot_decisions": args.pilot_decisions,
        "intervention_ids": [row["intervention_id"] for row in records],
    }
    protocol_digest = config_digest(identity)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_or_validate_config(args.output_dir / "evaluation_config.json", identity)
    (args.output_dir / "phase9_extraction_report.json").unlink(missing_ok=True)
    record_dir = args.output_dir / "records"
    record_dir.mkdir(exist_ok=True)

    runner = HfLlavaDecisionRunner.from_pretrained(
        str(model_config["model"]),
        revision=str(model_config["revision"]),
        quantization=str(model_config["quantization"]),
        positive_answer=str(model_config["positive_answer"]),
        negative_answer=str(model_config["negative_answer"]),
        attention_layer_offset=int(model_config["attention_layer_offset"]),
    )
    require(runner.resolved_revision == model_config["revision"], "resolved model revision differs")

    baselines: dict[str, object] = {}
    output_rows: list[dict[str, object]] = []
    start = time.monotonic()
    for position, record in enumerate(records, 1):
        destination = record_dir / f"{record['intervention_id']}.json"
        if destination.is_file():
            saved = read_json(destination)
            require(saved["protocol_digest"] == protocol_digest, "resume identity differs")
            output_rows.append(saved["row"])
            continue
        try:
            decision_id = str(record["decision_id"])
            decision = manifest[decision_id]
            target = int(decision["target"])
            require(target == int(record["target"]), "Phase 9 target differs from manifest")
            image_path = args.cub_root / "images" / decision["relative_path"]
            with Image.open(image_path) as opened:
                image = opened.convert("RGB")
                baseline = baselines.get(decision_id)
                if baseline is None:
                    baseline = runner.evaluate(
                        image, decision["prompt_text"], include_attention=False, generate=False
                    )
                    baselines[decision_id] = baseline
                    signed = baseline.answer_margin if target else -baseline.answer_margin
                    expected = float(phase5_image[decision_id]["correct_answer_margin"])
                    require(abs(signed - expected) <= args.baseline_tolerance,
                            "Phase 9 baseline does not reproduce Phase 5")
                changed = runner.evaluate(
                    image,
                    decision["prompt_text"],
                    include_attention=False,
                    generate=False,
                    intervention=HiddenIntervention(
                        stage=str(record["stage"]),
                        patch_indices=torch.tensor(record["token_indices"], dtype=torch.long),
                        replacement=means[str(record["stage"])],
                    ),
                )
            before = baseline.answer_margin if target else -baseline.answer_margin
            after = changed.answer_margin if target else -changed.answer_margin
            row = {
                "decision_id": decision_id,
                "image_id": record["image_id"],
                "attribute_id": int(record["attribute_id"]),
                "target": target,
                "phase5_failure": int(record["phase5_failure"]),
                "intervention_id": record["intervention_id"],
                "stage": record["stage"],
                "stage_role": record["stage_role"],
                "selection_method": record["selection_method"],
                "intervention_type": record["intervention_type"],
                "replicate": int(record["replicate"]),
                "correct_answer_teacher_forced_margin_before": before,
                "correct_answer_teacher_forced_margin_after": after,
            }
            atomic_json_write({
                "schema_version": 1,
                "protocol_digest": protocol_digest,
                "intervention_id": record["intervention_id"],
                "row": row,
            }, destination)
            output_rows.append(row)
        except Exception as error:
            atomic_json_write({
                "schema_version": 1,
                "status": "FAIL",
                "mode": args.mode,
                "policy_digest": policy_digest,
                "protocol_digest": protocol_digest,
                "failed_position": position,
                "failed_intervention_id": record["intervention_id"],
                "completed_interventions": len(output_rows),
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "resumable": True,
                "official_test_images_used": 0,
            }, args.output_dir / "phase9_failure_report.json")
            raise
        if position % 25 == 0 or position == len(records):
            print(f"[{position}/{len(records)}] elapsed={time.monotonic() - start:.1f}s", flush=True)

    require(len(output_rows) == len(records), "Phase 9 outcomes are incomplete")
    output_path = args.output_dir / "intervention_outcomes.csv"
    write_csv(output_path, output_rows)
    report = {
        "schema_version": 1,
        "status": "PASS",
        "mode": args.mode,
        "development_only": True,
        "phase8_protocol_unchanged": True,
        "policy_digest": policy_digest,
        "protocol_digest": protocol_digest,
        "interventions": len(records),
        "decisions": len({str(row["decision_id"]) for row in records}),
        "images": len({str(row["image_id"]) for row in records}),
        "selected_stage": plan["selected_stage"],
        "neighbor_stage": plan["neighbor_stage"],
        "runtime_seconds": time.monotonic() - start,
        "outcomes_sha256": sha256(output_path),
        "official_test_images_used": 0,
    }
    atomic_json_write(report, args.output_dir / "phase9_extraction_report.json")
    (args.output_dir / "phase9_failure_report.json").unlink(missing_ok=True)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
