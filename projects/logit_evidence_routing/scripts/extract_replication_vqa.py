#!/usr/bin/env python3
"""Run smoke, pilot, or full binary-VQA controls with LLaVA or Qwen2.5-VL."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image


PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from lger.hf_utilization import HfLlavaDecisionRunner  # noqa: E402
from lger.phase5 import strict_parse_binary  # noqa: E402
from lger.reproducibility import current_git_commit  # noqa: E402
from lger.stage_cache import atomic_json_write, config_digest, write_or_validate_config  # noqa: E402


CONTROLS = ("image", "prompt_only", "image_shuffled", "opposite_label_image")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"input manifest is empty: {path}")
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty output: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def runner_from_config(config: dict[str, Any]) -> Any:
    kwargs = {
        "revision": str(config["revision"]),
        "quantization": str(config["quantization"]),
        "positive_answer": str(config["positive_answer"]),
        "negative_answer": str(config["negative_answer"]),
        "attention_layer_offset": int(config["attention_layer_offset"]),
    }
    if config["adapter"] == "llava":
        return HfLlavaDecisionRunner.from_pretrained(str(config["model"]), **kwargs)
    if config["adapter"] == "qwen2_5_vl":
        from lger.hf_qwen import HfQwenDecisionRunner

        return HfQwenDecisionRunner.from_pretrained(str(config["model"]), **kwargs)
    raise RuntimeError(f"unsupported VLM adapter: {config['adapter']}")


def choose_decisions(rows: list[dict[str, str]], mode: str, pilot_decisions: int) -> list[dict[str, str]]:
    ordered = sorted(rows, key=lambda row: (
        int(row["attribute_id"]), int(row["target"]), str(row["decision_id"])
    ))
    if mode == "full":
        return ordered
    if mode == "smoke":
        positive = next(row for row in ordered if int(row["target"]) == 1)
        negative = next(row for row in ordered if int(row["target"]) == 0)
        return [positive, negative]
    if pilot_decisions < 4:
        raise RuntimeError("pilot_decisions must be at least four")
    # Round-robin attributes and labels rather than taking a convenient prefix.
    strata: dict[tuple[int, int], list[dict[str, str]]] = {}
    for row in ordered:
        strata.setdefault((int(row["attribute_id"]), int(row["target"])), []).append(row)
    selected = []
    position = 0
    keys = sorted(strata)
    while len(selected) < min(pilot_decisions, len(ordered)):
        added = False
        for key in keys:
            if position < len(strata[key]):
                selected.append(strata[key][position])
                added = True
                if len(selected) == min(pilot_decisions, len(ordered)):
                    break
        if not added:
            break
        position += 1
    return selected


def result_row(decision: dict[str, str], control: str, result: Any, adapter: str) -> dict[str, Any]:
    target = int(decision["target"])
    tie = result.answer_margin == 0.0
    prediction: int | str = "" if tie else int(result.answer_margin > 0)
    parsed = strict_parse_binary(result.generated_text or "")
    evaluated_image = {
        "image": decision["image_id"],
        "image_shuffled": decision["shuffled_image_id"],
        "opposite_label_image": decision["opposite_label_image_id"],
        "prompt_only": "",
    }[control]
    return {
        "schema_version": 1,
        "decision_id": decision["decision_id"],
        "image_id": decision["image_id"],
        "attribute_id": decision["attribute_id"],
        "attribute_name": decision.get("attribute_name", ""),
        "target": target,
        "control": control,
        "evaluated_image_id": evaluated_image,
        "adapter": adapter,
        "positive_log_likelihood": result.positive_log_likelihood,
        "negative_log_likelihood": result.negative_log_likelihood,
        "answer_margin": result.answer_margin,
        "correct_answer_margin": result.answer_margin if target else -result.answer_margin,
        "margin_tie": int(tie),
        "margin_prediction": prediction,
        "margin_correct": int(not tie and prediction == target),
        "generated_text": result.generated_text or "",
        "parsed_answer": parsed or "",
        "generation_correct": int((parsed == "yes") == bool(target)) if parsed is not None else "",
        "attention_entropy": result.attention_entropy if result.attention_entropy is not None else "",
        "attention_effective_tokens": (
            result.attention_effective_tokens if result.attention_effective_tokens is not None else ""
        ),
        "positive_token_ids": json.dumps(result.positive_token_ids),
        "negative_token_ids": json.dumps(result.negative_token_ids),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "pilot", "full"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke-dir", type=Path)
    parser.add_argument("--pilot-dir", type=Path)
    parser.add_argument("--pilot-decisions", type=int)
    args = parser.parse_args()

    config = read_json(args.config)
    if (
        config.get("schema_version") != 1
        or config.get("adapter") not in ("llava", "qwen2_5_vl")
        or tuple(config.get("controls", ())) != CONTROLS
        or config.get("official_test_images_used") != 0
    ):
        raise RuntimeError("unsupported development replication config")
    rows = read_csv(args.manifest)
    required = {
        "decision_id", "image_id", "attribute_id", "target", "prompt_text",
        "relative_path", "shuffled_image_id", "shuffled_relative_path",
        "opposite_label_image_id", "opposite_label_relative_path", "opposite_label_target",
    }
    if not all(required <= set(row) for row in rows):
        raise RuntimeError(f"replication manifest lacks required columns: {sorted(required)}")
    if any(int(row["opposite_label_target"]) != 1 - int(row["target"]) for row in rows):
        raise RuntimeError("opposite-label control does not reverse the queried attribute target")
    pilot_decisions = args.pilot_decisions or int(config.get("pilot_decisions", 40))
    decisions = choose_decisions(rows, args.mode, pilot_decisions)

    policy = {
        "schema_version": 1,
        "purpose": config["purpose"],
        "git_commit": current_git_commit(REPO),
        "config": config,
        "config_sha256": sha256(args.config),
        "manifest_sha256": sha256(args.manifest),
        "image_root": str(args.image_root.resolve()),
        "official_test_images_used": 0,
    }
    policy_digest = config_digest(policy)
    if args.mode in ("pilot", "full"):
        if args.smoke_dir is None:
            raise RuntimeError("--smoke-dir is required after smoke mode")
        smoke = read_json(args.smoke_dir / "replication_vqa_report.json")
        if smoke.get("status") != "PASS" or smoke.get("mode") != "smoke" \
                or smoke.get("policy_digest") != policy_digest:
            raise RuntimeError("a passing matching replication smoke is required")
    if args.mode == "full":
        if args.pilot_dir is None:
            raise RuntimeError("--pilot-dir is required for full mode")
        pilot = read_json(args.pilot_dir / "replication_vqa_report.json")
        if pilot.get("status") != "PASS" or pilot.get("mode") != "pilot" \
                or pilot.get("policy_digest") != policy_digest:
            raise RuntimeError("a passing matching replication pilot is required")

    identity = {
        **policy,
        "mode": args.mode,
        "pilot_decisions": pilot_decisions,
        "decision_ids": [row["decision_id"] for row in decisions],
    }
    protocol_digest = config_digest(identity)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_or_validate_config(args.output_dir / "evaluation_config.json", identity)
    record_dir = args.output_dir / "records"
    record_dir.mkdir(exist_ok=True)
    (args.output_dir / "replication_vqa_report.json").unlink(missing_ok=True)
    print(
        f"Loading {config['model']} for {args.mode} replication "
        f"({len(decisions)} decisions); the first load may download model files.",
        flush=True,
    )
    runner = runner_from_config(config)
    if runner.resolved_revision != config["revision"]:
        raise RuntimeError("resolved model revision differs from the frozen config")
    print(f"Model loaded at revision {runner.resolved_revision}.", flush=True)

    output: list[dict[str, Any]] = []
    start = time.monotonic()
    progress_interval = 1 if args.mode == "smoke" else 10
    for position, decision in enumerate(decisions, 1):
        destination = record_dir / (hashlib.sha256(
            str(decision["decision_id"]).encode()).hexdigest()[:20] + ".json")
        if destination.is_file():
            saved = read_json(destination)
            if saved["protocol_digest"] != protocol_digest:
                raise RuntimeError("replication resume identity differs")
            output.extend(saved["rows"])
            if position % progress_interval == 0 or position == len(decisions):
                print(
                    f"[{position}/{len(decisions)}] resumed "
                    f"elapsed={time.monotonic() - start:.1f}s",
                    flush=True,
                )
            continue
        try:
            images = {}
            for control, field in (
                ("image", "relative_path"),
                ("image_shuffled", "shuffled_relative_path"),
                ("opposite_label_image", "opposite_label_relative_path"),
            ):
                with Image.open(args.image_root / decision[field]) as opened:
                    images[control] = opened.convert("RGB")
            measured = {}
            for control in CONTROLS:
                measured[control] = runner.evaluate(
                    None if control == "prompt_only" else images[control],
                    decision["prompt_text"],
                    include_attention=(control == "image" and bool(
                        config.get("question_conditioned_llm_attention", False)
                    )),
                    generate=True,
                    max_new_tokens=int(config["max_new_tokens"]),
                )
            decision_rows = [result_row(decision, control, measured[control], config["adapter"])
                             for control in CONTROLS]
            atomic_json_write({
                "schema_version": 1,
                "protocol_digest": protocol_digest,
                "decision_id": decision["decision_id"],
                "rows": decision_rows,
            }, destination)
            output.extend(decision_rows)
        except Exception as error:
            atomic_json_write({
                "schema_version": 1,
                "status": "FAIL",
                "mode": args.mode,
                "protocol_digest": protocol_digest,
                "failed_position": position,
                "failed_decision_id": decision["decision_id"],
                "completed_decisions": len(output) // len(CONTROLS),
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "resumable": True,
                "official_test_images_used": 0,
            }, args.output_dir / "replication_vqa_failure_report.json")
            raise
        if position % progress_interval == 0 or position == len(decisions):
            print(f"[{position}/{len(decisions)}] elapsed={time.monotonic() - start:.1f}s", flush=True)

    if len(output) != len(decisions) * len(CONTROLS):
        raise RuntimeError("replication control outputs are incomplete")
    output_path = args.output_dir / "replication_vqa_decisions.csv"
    write_csv(output_path, output)
    report = {
        "schema_version": 1,
        "status": "PASS",
        "mode": args.mode,
        "adapter": config["adapter"],
        "model": config["model"],
        "resolved_revision": runner.resolved_revision,
        "policy_digest": policy_digest,
        "protocol_digest": protocol_digest,
        "decisions": len(decisions),
        "control_rows": len(output),
        "target_counts": dict(Counter(str(row["target"]) for row in decisions)),
        "margin_ties_by_control": {
            control: sum(int(row["margin_tie"]) for row in output if row["control"] == control)
            for control in CONTROLS
        },
        "positive_token_ids": list(runner.positive_token_ids),
        "negative_token_ids": list(runner.negative_token_ids),
        "runtime_seconds": time.monotonic() - start,
        "artifacts": {
            "evaluation_config.json": sha256(args.output_dir / "evaluation_config.json"),
            output_path.name: sha256(output_path),
        },
        "official_test_images_used": 0,
    }
    atomic_json_write(report, args.output_dir / "replication_vqa_report.json")
    (args.output_dir / "replication_vqa_failure_report.json").unlink(missing_ok=True)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
