#!/usr/bin/env python3
"""Run restart-safe answer-vocabulary and prompt-order controls on Kaggle."""

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

from PIL import Image


PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from lger.answer_robustness import render_condition_prompt, validate_conditions  # noqa: E402
from lger.hf_utilization import HfLlavaDecisionRunner  # noqa: E402
from lger.phase5 import strict_parse_binary  # noqa: E402
from lger.reproducibility import current_git_commit  # noqa: E402
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
    if not rows:
        raise RuntimeError(f"input CSV is empty: {path}")
    return rows


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty output: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def choose_decisions(
    rows: list[dict[str, str]], mode: str, pilot_decisions: int
) -> list[dict[str, str]]:
    ordered = sorted(rows, key=lambda row: (
        int(row["attribute_id"]), int(row["target"]), str(row["decision_id"])
    ))
    if mode == "full":
        return ordered
    if mode == "smoke":
        return [next(row for row in ordered if int(row["target"]) == target)
                for target in (0, 1)]
    if pilot_decisions < 6:
        raise RuntimeError("pilot_decisions must be at least six")
    strata: dict[tuple[int, int], list[dict[str, str]]] = {}
    for row in ordered:
        strata.setdefault((int(row["attribute_id"]), int(row["target"])), []).append(row)
    selected: list[dict[str, str]] = []
    position = 0
    while len(selected) < min(pilot_decisions, len(ordered)):
        added = False
        for key in sorted(strata):
            if position < len(strata[key]):
                selected.append(strata[key][position])
                added = True
                if len(selected) == min(pilot_decisions, len(ordered)):
                    break
        if not added:
            raise RuntimeError("could not construct stratified answer-token pilot")
        position += 1
    return selected


def runner_from_config(config: dict[str, Any]) -> Any:
    adapter = str(config.get("adapter", "llava"))
    kwargs = {
        "revision": str(config["revision"]),
        "quantization": str(config["quantization"]),
        "positive_answer": "yes",
        "negative_answer": "no",
        "attention_layer_offset": int(config["attention_layer_offset"]),
    }
    if adapter == "llava":
        return HfLlavaDecisionRunner.from_pretrained(str(config["model"]), **kwargs)
    if adapter == "qwen2_5_vl":
        from lger.hf_qwen import HfQwenDecisionRunner
        return HfQwenDecisionRunner.from_pretrained(str(config["model"]), **kwargs)
    raise RuntimeError(f"unsupported adapter: {adapter}")


def result_row(
    decision: dict[str, str], condition: dict[str, str], control: str,
    prompt: str, result: Any, adapter: str,
) -> dict[str, Any]:
    target = int(decision["target"])
    tie = result.answer_margin == 0.0
    prediction: int | str = "" if tie else int(result.answer_margin > 0)
    parsed = strict_parse_binary(
        result.generated_text or "", condition["positive_answer"], condition["negative_answer"]
    )
    evaluated_image = {
        "image": decision["image_id"],
        "prompt_only": "",
        "image_shuffled": decision["shuffled_image_id"],
    }[control]
    return {
        "schema_version": 1,
        "decision_id": decision["decision_id"],
        "image_id": decision["image_id"],
        "class_id": decision["class_id"],
        "attribute_id": decision["attribute_id"],
        "attribute_name": decision.get("attribute_name", ""),
        "target": target,
        "condition_id": condition["condition_id"],
        "vocabulary": condition["vocabulary"],
        "answer_order": condition["answer_order"],
        "prompt_text": prompt,
        "positive_answer": condition["positive_answer"],
        "negative_answer": condition["negative_answer"],
        "control": control,
        "evaluated_image_id": evaluated_image,
        "adapter": adapter,
        "positive_log_likelihood": result.positive_log_likelihood,
        "negative_log_likelihood": result.negative_log_likelihood,
        "semantic_positive_minus_negative_margin": result.answer_margin,
        "correct_answer_margin": result.answer_margin if target else -result.answer_margin,
        "margin_tie": int(tie),
        "margin_prediction": prediction,
        "margin_correct": int(not tie and prediction == target),
        "generated_text": result.generated_text or "",
        "parsed_semantic_positive": parsed if parsed is not None else "",
        "generation_correct": int(parsed == bool(target)) if parsed is not None else "",
        "positive_token_ids": json.dumps(result.positive_token_ids),
        "negative_token_ids": json.dumps(result.negative_token_ids),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "pilot", "full"), required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path,
                        default=PROJECT / "configs/answer_token_robustness.json")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke-dir", type=Path)
    parser.add_argument("--pilot-dir", type=Path)
    parser.add_argument("--pilot-decisions", type=int)
    args = parser.parse_args()

    model_config = read_json(args.model_config)
    experiment = read_json(args.experiment_config)
    if not (experiment.get("schema_version") == 1
            and experiment.get("development_only") is True
            and experiment.get("official_test_images_used") == 0):
        raise RuntimeError("answer-token experiment must remain development-only")
    conditions = validate_conditions(experiment.get("conditions", []))
    controls = tuple(str(value) for value in experiment.get("controls", []))
    if controls != ("image", "prompt_only", "image_shuffled"):
        raise RuntimeError("unsupported answer-token visual controls")
    manifest = read_csv(args.manifest)
    required = {
        "decision_id", "image_id", "class_id", "attribute_id", "attribute_name",
        "attribute_phrase", "target", "relative_path", "shuffled_image_id",
        "shuffled_relative_path", "split", "official_split",
    }
    if not all(required <= set(row) for row in manifest):
        raise RuntimeError(f"manifest lacks required columns: {sorted(required)}")
    if not all(row["split"] == "val" and row["official_split"] == "train" for row in manifest):
        raise RuntimeError("answer-token robustness must exclude official-test images")
    pilot_decisions = args.pilot_decisions or int(experiment["pilot_decisions"])
    decisions = choose_decisions(manifest, args.mode, pilot_decisions)
    adapter = str(model_config.get("adapter", "llava"))
    policy = {
        "schema_version": 1,
        "purpose": experiment["purpose"],
        "development_only": True,
        "git_commit": current_git_commit(REPO),
        "adapter": adapter,
        "model": model_config["model"],
        "revision": model_config["revision"],
        "quantization": model_config["quantization"],
        "model_config_sha256": sha256(args.model_config),
        "experiment_config_sha256": sha256(args.experiment_config),
        "manifest_sha256": sha256(args.manifest),
        "official_test_images_used": 0,
    }
    policy_digest = config_digest(policy)
    if args.mode in ("pilot", "full"):
        if args.smoke_dir is None:
            raise RuntimeError("--smoke-dir is required after smoke mode")
        smoke = read_json(args.smoke_dir / "answer_token_robustness_report.json")
        if not (smoke.get("status") == "PASS" and smoke.get("mode") == "smoke"
                and smoke.get("policy_digest") == policy_digest):
            raise RuntimeError("a passing matching answer-token smoke is required")
    if args.mode == "full":
        if args.pilot_dir is None:
            raise RuntimeError("--pilot-dir is required for full mode")
        pilot = read_json(args.pilot_dir / "answer_token_robustness_report.json")
        if not (pilot.get("status") == "PASS" and pilot.get("mode") == "pilot"
                and pilot.get("policy_digest") == policy_digest):
            raise RuntimeError("a passing matching answer-token pilot is required")

    identity = {
        **policy,
        "mode": args.mode,
        "pilot_decisions": pilot_decisions,
        "decision_ids": [row["decision_id"] for row in decisions],
        "condition_ids": [row["condition_id"] for row in conditions],
        "controls": list(controls),
    }
    protocol_digest = config_digest(identity)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_or_validate_config(args.output_dir / "evaluation_config.json", identity)
    record_dir = args.output_dir / "records"
    record_dir.mkdir(exist_ok=True)
    (args.output_dir / "answer_token_robustness_report.json").unlink(missing_ok=True)

    print(
        f"Loading {model_config['model']} once for {len(decisions)} decisions x "
        f"{len(conditions)} conditions x {len(controls)} controls.", flush=True,
    )
    runner = runner_from_config(model_config)
    if runner.resolved_revision != model_config["revision"]:
        raise RuntimeError("resolved model revision differs from the frozen config")
    tokenizations = {}
    for condition in conditions:
        runner.set_answer_pair(condition["positive_answer"], condition["negative_answer"])
        tokenizations[condition["condition_id"]] = {
            "positive": list(runner.positive_token_ids),
            "negative": list(runner.negative_token_ids),
        }
    print("All answer pairs tokenize and round-trip.", flush=True)

    output: list[dict[str, Any]] = []
    start = time.monotonic()
    completed = 0
    total = len(decisions) * len(conditions)
    for decision in decisions:
        with Image.open(args.image_root / decision["relative_path"]) as opened:
            image = opened.convert("RGB")
        with Image.open(args.image_root / decision["shuffled_relative_path"]) as opened:
            shuffled = opened.convert("RGB")
        images = {"image": image, "prompt_only": None, "image_shuffled": shuffled}
        for condition in conditions:
            record_id = hashlib.sha256(
                f"{decision['decision_id']}::{condition['condition_id']}".encode("utf-8")
            ).hexdigest()[:24]
            destination = record_dir / f"{record_id}.json"
            if destination.is_file():
                saved = read_json(destination)
                if saved.get("protocol_digest") != protocol_digest:
                    raise RuntimeError("answer-token resume identity differs")
                output.extend(saved["rows"])
                completed += 1
                continue
            try:
                runner.set_answer_pair(
                    condition["positive_answer"], condition["negative_answer"]
                )
                prompt = render_condition_prompt(condition, decision["attribute_phrase"])
                measured = {
                    control: runner.evaluate(
                        images[control], prompt, include_attention=False, generate=True,
                        max_new_tokens=int(experiment["max_new_tokens"]),
                    ) for control in controls
                }
                rows = [result_row(
                    decision, condition, control, prompt, measured[control], adapter
                ) for control in controls]
                atomic_json_write({
                    "schema_version": 1,
                    "protocol_digest": protocol_digest,
                    "decision_id": decision["decision_id"],
                    "condition_id": condition["condition_id"],
                    "rows": rows,
                }, destination)
                output.extend(rows)
                completed += 1
            except Exception as error:
                atomic_json_write({
                    "schema_version": 1,
                    "status": "FAIL",
                    "mode": args.mode,
                    "protocol_digest": protocol_digest,
                    "failed_decision_id": decision["decision_id"],
                    "failed_condition_id": condition["condition_id"],
                    "completed_decision_conditions": completed,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                    "resumable": True,
                    "official_test_images_used": 0,
                }, args.output_dir / "answer_token_robustness_failure.json")
                raise
            if completed % (1 if args.mode == "smoke" else 12) == 0 or completed == total:
                elapsed = time.monotonic() - start
                rate = elapsed / completed
                remaining = rate * (total - completed)
                print(
                    f"[{completed}/{total}] elapsed={elapsed:.1f}s "
                    f"estimated_remaining={remaining:.1f}s", flush=True,
                )

    expected_rows = len(decisions) * len(conditions) * len(controls)
    if len(output) != expected_rows:
        raise RuntimeError("answer-token result coverage is incomplete")
    output_path = args.output_dir / "answer_token_robustness_decisions.csv"
    write_csv(output_path, output)
    report = {
        "schema_version": 1,
        "status": "PASS",
        "mode": args.mode,
        "development_only": True,
        "policy_digest": policy_digest,
        "protocol_digest": protocol_digest,
        "adapter": adapter,
        "model": model_config["model"],
        "resolved_revision": runner.resolved_revision,
        "decisions": len(decisions),
        "images": len({row["image_id"] for row in decisions}),
        "target_counts": {
            str(target): sum(int(row["target"]) == target for row in decisions)
            for target in (0, 1)
        },
        "conditions": len(conditions),
        "controls": list(controls),
        "result_rows": len(output),
        "tokenizations": tokenizations,
        "runtime_seconds": time.monotonic() - start,
        "seconds_per_decision_condition": (time.monotonic() - start) / total,
        "outcomes_sha256": sha256(output_path),
        "official_test_images_used": 0,
    }
    atomic_json_write(report, args.output_dir / "answer_token_robustness_report.json")
    (args.output_dir / "answer_token_robustness_failure.json").unlink(missing_ok=True)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
