#!/usr/bin/env python3
"""Run resumable balanced-target and K-dose LLaVA interventions on Kaggle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch
from PIL import Image


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from lger.hf_utilization import HiddenIntervention, HfLlavaDecisionRunner  # noqa: E402
from lger.phase10 import PURPOSE_PLAN, require  # noqa: E402
from lger.stage_cache import atomic_json_write, config_digest, write_or_validate_config  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), f"expected JSON object: {path}")
    return payload


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    require(bool(rows), f"input CSV is empty: {path}")
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    require(bool(rows), f"refusing to write empty output: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def choose_records(records: list[dict[str, Any]], mode: str, pilot_decisions: int) -> list[dict[str, Any]]:
    identities: dict[str, tuple[int, int]] = {}
    for row in records:
        identities[str(row["decision_id"])] = (int(row["target"]), int(row["attribute_id"]))
    ordered = sorted(identities, key=lambda value: (identities[value][1], identities[value][0], value))
    if mode == "full":
        chosen = set(ordered)
    elif mode == "smoke":
        chosen = {
            next(value for value in ordered if identities[value][0] == 0),
            next(value for value in ordered if identities[value][0] == 1),
        }
    else:
        require(pilot_decisions >= 4, "pilot_decisions must be at least four")
        strata: dict[tuple[int, int], list[str]] = {}
        for decision_id in ordered:
            target, attribute = identities[decision_id]
            strata.setdefault((attribute, target), []).append(decision_id)
        selected: list[str] = []
        position = 0
        while len(selected) < min(pilot_decisions, len(ordered)):
            added = False
            for key in sorted(strata):
                if position < len(strata[key]):
                    selected.append(strata[key][position])
                    added = True
                    if len(selected) == min(pilot_decisions, len(ordered)):
                        break
            require(added, "could not construct a stratified pilot")
            position += 1
        chosen = set(selected)
    return [row for row in records if str(row["decision_id"]) in chosen]


def reusable_rows(paths: list[Path], plan_index: dict[tuple[str, str], dict[str, Any]]) \
        -> dict[tuple[str, str], dict[str, Any]]:
    output: dict[tuple[str, str], dict[str, Any]] = {}
    for path in paths:
        for source in read_csv(path):
            key = (str(source["decision_id"]), str(source["intervention_id"]))
            if key not in plan_index:
                continue
            require(key not in output, "reused intervention outcome is duplicated")
            planned = plan_index[key]
            for field in ("image_id", "attribute_id", "target", "stage", "selection_method",
                          "intervention_type", "replicate"):
                require(str(source[field]) == str(planned[field]),
                        f"reused outcome contradicts planned {field}")
            output[key] = dict(source)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "pilot", "full"), required=True)
    parser.add_argument("--model-config", type=Path,
                        default=PROJECT / "configs/phase5_utilization.json")
    parser.add_argument("--experiment-config", type=Path,
                        default=PROJECT / "configs/phase10_priority1.json")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--behavior-decisions", type=Path, required=True)
    parser.add_argument("--means-file", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--cub-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reuse-outcomes", type=Path, action="append", default=[])
    parser.add_argument("--smoke-dir", type=Path)
    parser.add_argument("--pilot-dir", type=Path)
    parser.add_argument("--pilot-decisions", type=int, default=20)
    parser.add_argument("--baseline-tolerance", type=float, default=5e-3)
    args = parser.parse_args()

    model_config = read_json(args.model_config)
    experiment_config = read_json(args.experiment_config)
    plan = read_json(args.plan)
    require(
        plan["status"] == "PASS" and plan["purpose"] == PURPOSE_PLAN
        and plan["development_only"] is True and plan["phase8_protocol_unchanged"] is True
        and plan["official_test_images_used"] == 0
        and experiment_config["development_only"] is True
        and experiment_config["phase8_protocol_unchanged"] is True
        and experiment_config["official_test_images_used"] == 0
        and plan["selected_stage"] == experiment_config["selected_stage"]
        and plan["neighbor_stage"] == experiment_config["neighbor_stage"]
        and plan["selector_methods"] == experiment_config["selector_methods"]
        and plan["balanced_k"] == experiment_config["balanced_k"]
        and plan["dose_k_values"] == experiment_config["dose_k_values"]
        and plan["matched_random_seeds"] == experiment_config["matched_random_seeds"],
        "compatible development-only Priority 1 plan and config are required",
    )
    require(args.cub_root.is_dir(), "CUB root does not exist")
    manifest_rows = read_csv(args.manifest)
    manifest = {row["decision_id"]: row for row in manifest_rows}
    require(len(manifest) == len(manifest_rows), "manifest decision IDs are not unique")
    require(all(row["split"] == "val" and row["official_split"] == "train"
                for row in manifest_rows), "manifest must exclude the official test split")
    behavior_rows = read_csv(args.behavior_decisions)
    behavior = {row["decision_id"]: row for row in behavior_rows if row["control"] == "image"}
    require(set(behavior) == set(manifest), "behavior baselines must exactly cover the manifest")

    try:
        from safetensors import safe_open
    except ImportError as error:  # pragma: no cover - Kaggle dependency guard
        raise RuntimeError("Priority 1 extraction requires safetensors") from error
    means: dict[str, torch.Tensor] = {}
    with safe_open(args.means_file, framework="pt", device="cpu") as handle:
        for stage in handle.keys():
            means[stage] = handle.get_tensor(stage).float()
    require(set(means) == {plan["selected_stage"], plan["neighbor_stage"]},
            "replacement means do not match the intervention stages")

    plan_records = list(plan["records"])
    plan_index = {(str(row["decision_id"]), str(row["intervention_id"])): row
                  for row in plan_records}
    reuse = reusable_rows(args.reuse_outcomes, plan_index)
    policy = {
        "schema_version": 1,
        "purpose": "phase10_llava_priority1_intervention_extraction",
        "development_only": True,
        "phase8_protocol_unchanged": True,
        "model": model_config["model"],
        "revision": model_config["revision"],
        "quantization": model_config["quantization"],
        "model_config_sha256": sha256(args.model_config),
        "experiment_config_sha256": sha256(args.experiment_config),
        "manifest_sha256": sha256(args.manifest),
        "behavior_decisions_sha256": sha256(args.behavior_decisions),
        "means_sha256": sha256(args.means_file),
        "plan_sha256": sha256(args.plan),
        "reuse_outcome_hashes": [sha256(path) for path in args.reuse_outcomes],
        "reuse_rule": "decision_id and intervention_id plus all design fields must match",
        "baseline_tolerance": args.baseline_tolerance,
        "official_test_images_used": 0,
    }
    policy_digest = config_digest(policy)
    if args.mode in ("pilot", "full"):
        require(args.smoke_dir is not None, "--smoke-dir is required after smoke mode")
        smoke = read_json(args.smoke_dir / "phase10_extraction_report.json")
        require(smoke["status"] == "PASS" and smoke["mode"] == "smoke"
                and smoke["policy_digest"] == policy_digest,
                "a passing matching Priority 1 smoke is required")
    if args.mode == "full":
        require(args.pilot_dir is not None, "--pilot-dir is required for full mode")
        pilot = read_json(args.pilot_dir / "phase10_extraction_report.json")
        require(pilot["status"] == "PASS" and pilot["mode"] == "pilot"
                and pilot["policy_digest"] == policy_digest,
                "a passing matching Priority 1 pilot is required")

    records = choose_records(plan_records, args.mode, args.pilot_decisions)
    identity = {**policy, "mode": args.mode, "pilot_decisions": args.pilot_decisions,
                "intervention_ids": [row["intervention_id"] for row in records]}
    protocol_digest = config_digest(identity)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_or_validate_config(args.output_dir / "evaluation_config.json", identity)
    (args.output_dir / "phase10_extraction_report.json").unlink(missing_ok=True)
    record_dir = args.output_dir / "records"
    record_dir.mkdir(exist_ok=True)

    require(any((str(row["decision_id"]), str(row["intervention_id"])) not in reuse
                for row in records), "selected mode contains no new interventions")
    print(
        f"Loading pinned LLaVA for Priority 1 {args.mode} ({len(records)} interventions; "
        f"{sum((str(row['decision_id']), str(row['intervention_id'])) in reuse for row in records)} reusable).",
        flush=True,
    )
    runner = HfLlavaDecisionRunner.from_pretrained(
        str(model_config["model"]), revision=str(model_config["revision"]),
        quantization=str(model_config["quantization"]),
        positive_answer=str(model_config["positive_answer"]),
        negative_answer=str(model_config["negative_answer"]),
        attention_layer_offset=int(model_config["attention_layer_offset"]),
    )
    require(runner.resolved_revision == model_config["revision"], "resolved model revision differs")

    baselines: dict[str, Any] = {}
    output_rows: list[dict[str, Any]] = []
    reused_count = 0
    executed_count = 0
    start = time.monotonic()
    progress_interval = 1 if args.mode == "smoke" else 10 if args.mode == "pilot" else 100
    for position, record in enumerate(records, 1):
        destination = record_dir / f"{record['intervention_id']}.json"
        if destination.is_file():
            saved = read_json(destination)
            require(saved["protocol_digest"] == protocol_digest, "resume identity differs")
            output_rows.append(saved["row"])
            reused_count += int(saved.get("source") == "reused_retained_outcome")
            executed_count += int(saved.get("source") == "new_model_forward")
            continue
        key = (str(record["decision_id"]), str(record["intervention_id"]))
        if key in reuse:
            row = {**reuse[key], "selection_k": int(record["selection_k"])}
            atomic_json_write({
                "schema_version": 1, "protocol_digest": protocol_digest,
                "intervention_id": record["intervention_id"],
                "source": "reused_retained_outcome", "row": row,
            }, destination)
            output_rows.append(row)
            reused_count += 1
            continue
        try:
            decision_id = str(record["decision_id"])
            decision = manifest[decision_id]
            target = int(decision["target"])
            require(target == int(record["target"]), "plan target differs from manifest")
            with Image.open(args.cub_root / "images" / decision["relative_path"]) as opened:
                image = opened.convert("RGB")
                baseline = baselines.get(decision_id)
                if baseline is None:
                    baseline = runner.evaluate(
                        image, decision["prompt_text"], include_attention=False, generate=False
                    )
                    baselines[decision_id] = baseline
                    signed = baseline.answer_margin if target else -baseline.answer_margin
                    expected = float(behavior[decision_id]["correct_answer_margin"])
                    require(abs(signed - expected) <= args.baseline_tolerance,
                            "Priority 1 baseline does not reproduce retained behavior")
                changed = runner.evaluate(
                    image, decision["prompt_text"], include_attention=False, generate=False,
                    intervention=HiddenIntervention(
                        stage=str(record["stage"]),
                        patch_indices=torch.tensor(record["token_indices"], dtype=torch.long),
                        replacement=means[str(record["stage"])],
                    ),
                )
            before = baseline.answer_margin if target else -baseline.answer_margin
            after = changed.answer_margin if target else -changed.answer_margin
            row = {
                "decision_id": decision_id, "image_id": record["image_id"],
                "attribute_id": int(record["attribute_id"]), "target": target,
                "phase5_failure": int(record["phase5_failure"]),
                "intervention_id": record["intervention_id"], "stage": record["stage"],
                "stage_role": record["stage_role"],
                "selection_method": record["selection_method"],
                "selection_k": int(record["selection_k"]),
                "intervention_type": record["intervention_type"],
                "replicate": int(record["replicate"]),
                "correct_answer_teacher_forced_margin_before": before,
                "correct_answer_teacher_forced_margin_after": after,
            }
            atomic_json_write({
                "schema_version": 1, "protocol_digest": protocol_digest,
                "intervention_id": record["intervention_id"],
                "source": "new_model_forward", "row": row,
            }, destination)
            output_rows.append(row)
            executed_count += 1
        except Exception as error:
            atomic_json_write({
                "schema_version": 1, "status": "FAIL", "mode": args.mode,
                "policy_digest": policy_digest, "protocol_digest": protocol_digest,
                "failed_position": position, "failed_intervention_id": record["intervention_id"],
                "completed_interventions": len(output_rows), "error_type": type(error).__name__,
                "error": str(error), "traceback": traceback.format_exc(), "resumable": True,
                "official_test_images_used": 0,
            }, args.output_dir / "phase10_failure_report.json")
            raise
        if position % progress_interval == 0 or position == len(records):
            print(f"[{position}/{len(records)}] elapsed={time.monotonic() - start:.1f}s", flush=True)

    require(len(output_rows) == len(records), "Priority 1 outcomes are incomplete")
    output_path = args.output_dir / "intervention_outcomes.csv"
    write_csv(output_path, output_rows)
    report = {
        "schema_version": 1, "status": "PASS", "mode": args.mode,
        "development_only": True, "phase8_protocol_unchanged": True,
        "policy_digest": policy_digest, "protocol_digest": protocol_digest,
        "interventions": len(records), "reused_interventions": reused_count,
        "executed_interventions": executed_count,
        "decisions": len({str(row["decision_id"]) for row in records}),
        "target_counts": {
            str(target): len({str(row["decision_id"]) for row in records
                              if int(row["target"]) == target}) for target in (0, 1)
        },
        "images": len({str(row["image_id"]) for row in records}),
        "selected_stage": plan["selected_stage"], "neighbor_stage": plan["neighbor_stage"],
        "runtime_seconds": time.monotonic() - start, "outcomes_sha256": sha256(output_path),
        "official_test_images_used": 0,
    }
    atomic_json_write(report, args.output_dir / "phase10_extraction_report.json")
    (args.output_dir / "phase10_failure_report.json").unlink(missing_ok=True)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
