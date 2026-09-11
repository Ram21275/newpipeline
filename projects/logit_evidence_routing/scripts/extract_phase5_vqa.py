#!/usr/bin/env python3
"""Collect frozen CUB-VQA answer margins and controls for Phase 5 on Kaggle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from lger.hf_utilization import HfLlavaDecisionRunner  # noqa: E402
from lger.phase4 import load_development_metadata, load_policy, read_json, require  # noqa: E402
from lger.reproducibility import current_git_commit  # noqa: E402
from lger.stage_cache import atomic_json_write, config_digest, write_or_validate_config  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strict_binary_parse(text: str | None) -> str | None:
    """Parse only an unambiguous leading yes/no answer."""

    if text is None:
        return None
    normalized = " ".join(text.strip().casefold().split())
    if not normalized:
        return None
    first = normalized.split(maxsplit=1)[0].strip(".,!?;:")
    return first if first in {"yes", "no"} else None


def attribute_phrase(text: str) -> str:
    prefix = "a photo of a bird with "
    normalized = text.strip()
    if not normalized.casefold().startswith(prefix) or not normalized.endswith("."):
        raise RuntimeError(f"cannot derive a fixed binary question from {text!r}")
    phrase = normalized[len(prefix) : -1].strip()
    if not phrase:
        raise RuntimeError("attribute phrase is empty")
    return phrase


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    require(bool(rows), f"refusing to write empty table: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def build_decisions(
    records: list[dict[str, object]],
    eligibility_path: Path,
    policy: dict[str, object],
    config: dict[str, object],
) -> list[dict[str, object]]:
    """Freeze a label-balanced validation cohort before any VLM call."""

    images = {int(row["image"]["image_id"]): row["image"] for row in records}
    attributes = {int(row["attribute_id"]): row for row in policy["attributes"]}
    grouped: dict[int, dict[str, list[dict[str, str]]]] = defaultdict(
        lambda: {"grounded_positive": [], "grounded_negative": []}
    )
    for row in read_csv(eligibility_path):
        if row["split"] != config["development_evaluation_split"]:
            continue
        image_id = int(row["image_id"])
        attribute_id = int(row["attribute_id"])
        require(image_id in images and attribute_id in attributes, "eligibility identity differs")
        visible = int(row["relevant_visible_in_crop_parts"]) > 0
        if row["reason"] == "eligible":
            grouped[attribute_id]["grounded_positive"].append(row)
        elif row["reason"] == "observed_negative" and visible:
            grouped[attribute_id]["grounded_negative"].append(row)

    rng = random.Random(int(config["label_balancing_seed"]))
    chosen: list[dict[str, object]] = []
    prompt_id = str(config["prompt_id"])
    for attribute_id in sorted(attributes):
        positive = sorted(grouped[attribute_id]["grounded_positive"], key=lambda r: int(r["image_id"]))
        negative = sorted(grouped[attribute_id]["grounded_negative"], key=lambda r: int(r["image_id"]))
        require(positive and negative, f"attribute {attribute_id} lacks one grounded label cohort")
        balanced_count = min(len(positive), len(negative))
        sampled_positive = rng.sample(positive, balanced_count)
        sampled_negative = rng.sample(negative, balanced_count)
        for cohort, target, rows in (
            ("grounded_positive", 1, sampled_positive),
            ("grounded_negative", 0, sampled_negative),
        ):
            for row in rows:
                image_id = int(row["image_id"])
                attribute = attributes[attribute_id]
                phrase = attribute_phrase(str(attribute["text"]))
                chosen.append(
                    {
                        "decision_id": f"{image_id}::{attribute_id}::{prompt_id}",
                        "image_id": image_id,
                        "split": row["split"],
                        "class_id": int(images[image_id]["class_id"]),
                        "relative_path": str(images[image_id]["relative_path"]),
                        "attribute_id": attribute_id,
                        "attribute_name": str(attribute["name"]),
                        "attribute_group": str(attribute["group"]),
                        "attribute_phrase": phrase,
                        "prompt_id": prompt_id,
                        "prompt_text": str(config["prompt_template"]).format(
                            attribute_phrase=phrase
                        ),
                        "cohort": cohort,
                        "target": target,
                        "certainty_name": row["certainty_name"],
                        "relevant_visible_in_crop_parts": int(
                            row["relevant_visible_in_crop_parts"]
                        ),
                        "official_split": "train",
                    }
                )

    # An attribute-wise cyclic derangement keeps questions/labels fixed while
    # replacing the visual evidence.  Rotation is deterministic but seed-dependent.
    by_attribute: dict[int, list[dict[str, object]]] = defaultdict(list)
    for decision in chosen:
        by_attribute[int(decision["attribute_id"])].append(decision)
    shuffle_seed = int(config["image_shuffle_seed"])
    for attribute_id, rows in by_attribute.items():
        ordered = sorted(rows, key=lambda row: (int(row["image_id"]), int(row["target"])))
        require(len(ordered) > 1, f"attribute {attribute_id} cannot be deranged")
        shift = 1 + (shuffle_seed + attribute_id) % (len(ordered) - 1)
        rotated = ordered[shift:] + ordered[:shift]
        for destination, source in zip(ordered, rotated):
            require(destination["image_id"] != source["image_id"], "image shuffle is not a derangement")
            destination["shuffled_image_id"] = int(source["image_id"])
            destination["shuffled_relative_path"] = str(source["relative_path"])
            destination["shuffled_class_id"] = int(source["class_id"])
    decision_ids = [str(row["decision_id"]) for row in chosen]
    require(len(decision_ids) == len(set(decision_ids)), "duplicate Phase 5 decisions")
    return sorted(chosen, key=lambda row: (int(row["attribute_id"]), str(row["cohort"]), int(row["image_id"])))


def output_row(
    decision: dict[str, object],
    control: str,
    output: object,
) -> dict[str, object]:
    target = int(decision["target"])
    prediction = int(output.answer_margin >= 0)
    parsed = strict_binary_parse(output.generated_text)
    return {
        "schema_version": 1,
        "decision_id": decision["decision_id"],
        "image_id": decision["image_id"],
        "split": decision["split"],
        "attribute_id": decision["attribute_id"],
        "attribute_name": decision["attribute_name"],
        "attribute_group": decision["attribute_group"],
        "attribute_phrase": decision["attribute_phrase"],
        "prompt_id": decision["prompt_id"],
        "prompt_text": decision["prompt_text"],
        "cohort": decision["cohort"],
        "target": target,
        "certainty_name": decision["certainty_name"],
        "relevant_visible_in_crop_parts": decision[
            "relevant_visible_in_crop_parts"
        ],
        "official_split": decision["official_split"],
        "control": control,
        "evaluated_image_id": (
            decision["shuffled_image_id"] if control == "image_shuffled" else decision["image_id"]
        ) if control != "prompt_only" else "",
        "positive_answer": "yes",
        "negative_answer": "no",
        "positive_log_likelihood": output.positive_log_likelihood,
        "negative_log_likelihood": output.negative_log_likelihood,
        "answer_margin": output.answer_margin,
        "correct_answer_margin": output.answer_margin if target else -output.answer_margin,
        "margin_prediction": prediction,
        "margin_correct": int(prediction == target),
        "generated_text": output.generated_text or "",
        "parsed_answer": parsed or "",
        "generation_parseable": int(parsed is not None) if control == "image" else "",
        "generation_correct": int((parsed == "yes") == bool(target)) if parsed is not None and control == "image" else "",
        "attention_entropy": output.attention_entropy if output.attention_entropy is not None else "",
        "attention_effective_tokens": output.attention_effective_tokens if output.attention_effective_tokens is not None else "",
        "attention_scores": json.dumps(output.attention_scores.tolist()) if output.attention_scores is not None else "",
        "positive_token_ids": json.dumps(output.positive_token_ids),
        "negative_token_ids": json.dumps(output.negative_token_ids),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "development"), required=True)
    parser.add_argument("--config", type=Path, default=PROJECT / "configs/phase5_utilization.json")
    parser.add_argument("--phase4-policy", type=Path, default=PROJECT / "configs/phase4_localization.json")
    parser.add_argument("--stage-cache", type=Path, required=True)
    parser.add_argument(
        "--phase3-dir",
        type=Path,
        help="Optional completed Phase 3R directory. Omit only when Phase 3R is running concurrently.",
    )
    parser.add_argument("--phase4-dir", type=Path, required=True)
    parser.add_argument(
        "--phase4-review",
        type=Path,
        default=PROJECT / "reports/development_20260911/phase4_review_decision.json",
    )
    parser.add_argument("--search-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--cub-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke-dir", type=Path)
    args = parser.parse_args()
    require(Path.cwd().resolve() == PROJECT.resolve(), "Run from projects/logit_evidence_routing")

    config = read_json(args.config)
    policy = load_policy(args.phase4_policy)
    require(
        config["schema_version"] == 1
        and config["cache_config_digest"] == policy["cache_config_digest"]
        and config["controls"] == ["image", "prompt_only", "image_shuffled"]
        and config["label_balancing"]
        == "attribute_stratified_without_replacement_to_smaller_grounded_label_count"
        and isinstance(config["label_balancing_seed"], int)
        and config["official_test_images_used"] == 0,
        "unsupported Phase 5 policy",
    )
    cfg2, records = load_development_metadata(args.stage_cache, policy)
    review = read_json(args.phase4_review)
    phase4 = read_json(args.phase4_dir / "phase4_validation_report.json")
    phase3_report_sha256 = None
    phase3_concurrent = args.phase3_dir is None
    if args.phase3_dir is not None:
        phase3 = read_json(args.phase3_dir / "phase3_run_report.json")
        require(
            phase3.get("schema_version") == 2
            and phase3["status"] == "PASS"
            and phase3["cache_config_digest"] == config["cache_config_digest"]
            and phase3["prompt_id"] == config["prompt_id"]
            and phase3["certainty_policy_overrides_masked"] == review["unapproved_cached_targets_masked"]
            and phase3["official_test_images_used"] == 0,
            "corrected Phase 3R artifacts are incompatible",
        )
        require(
            (args.phase3_dir / "decision_probe_scores.csv").is_file()
            and (args.phase3_dir / "probe_parameters.safetensors").is_file(),
            "Phase 3R decision/probe artifacts are missing",
        )
        phase3_report_sha256 = sha256(args.phase3_dir / "phase3_run_report.json")
    require(
        review["status"] == "PASS_DESCRIPTIVE_ONLY"
        and review["numeric_review_complete"] is True
        and review["qualitative_panels_reviewed"] is True
        and review["official_test_images_used"] == 0
        and sha256(args.phase4_dir / "bundle_manifest.json") == review["bundle_manifest_sha256"]
        and phase4["status"] == "PASS"
        and phase4["images"] == 240
        and phase4["official_test_images_used"] == 0,
        "reviewed full Phase 4 development bundle is required",
    )
    for key in ("model", "resolved_revision", "quantization"):
        config_key = "revision" if key == "resolved_revision" else key
        require(cfg2[key] == config[config_key], f"Phase 2/5 model provenance mismatch: {key}")

    cub_root = args.cub_root
    if cub_root is None:
        cached_root = Path(str(cfg2["dataset"]["root"]))
        if cached_root.is_dir():
            cub_root = cached_root
        else:
            from lger.cub import discover_cub_root

            cub_root = discover_cub_root(args.search_root)
    decisions = build_decisions(
        records,
        args.phase4_dir / "attribute_eligibility.csv",
        policy,
        config,
    )
    if args.mode == "smoke":
        first_positive = next(row for row in decisions if row["target"] == 1)
        first_negative = next(row for row in decisions if row["target"] == 0)
        decisions = [first_positive, first_negative]

    policy_identity = {
        "schema_version": 1,
        "purpose": config["purpose"],
        "git_commit": current_git_commit(REPO),
        "config": config,
        "config_sha256": sha256(args.config),
        "phase4_policy_sha256": sha256(args.phase4_policy),
        "stage_cache_digest": config["cache_config_digest"],
        "phase3_report_sha256": phase3_report_sha256,
        "phase3_concurrent": phase3_concurrent,
        "phase4_bundle_manifest_sha256": sha256(args.phase4_dir / "bundle_manifest.json"),
        "phase4_review_sha256": sha256(args.phase4_review),
        "cub_root": str(cub_root.resolve()),
        "official_test_images_used": 0,
        "cross_phase_analysis_blocked_until_phase3r": phase3_concurrent,
    }
    policy_digest = config_digest(policy_identity)
    identity = {
        **policy_identity,
        "mode": args.mode,
        "decision_ids": [row["decision_id"] for row in decisions],
        "policy_digest": policy_digest,
    }
    protocol_digest = config_digest(identity)
    if args.mode == "development":
        require(args.smoke_dir is not None, "--smoke-dir is required for development mode")
        smoke = read_json(args.smoke_dir / "phase5_run_report.json")
        require(
            smoke["status"] == "PASS"
            and smoke["mode"] == "smoke"
            and smoke["policy_digest"] == policy_digest,
            "matching Phase 5 smoke is required",
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_or_validate_config(args.output_dir / "evaluation_config.json", identity)
    manifest_path = args.output_dir / "decision_manifest.csv"
    write_csv(manifest_path, decisions)
    (args.output_dir / "phase5_run_report.json").unlink(missing_ok=True)
    record_dir = args.output_dir / "records"
    record_dir.mkdir(exist_ok=True)
    runner = HfLlavaDecisionRunner.from_pretrained(
        str(config["model"]),
        revision=str(config["revision"]),
        quantization=str(config["quantization"]),
        positive_answer=str(config["positive_answer"]),
        negative_answer=str(config["negative_answer"]),
        attention_layer_offset=int(config["attention_layer_offset"]),
    )
    require(runner.resolved_revision == config["revision"], "resolved model revision differs")
    rows: list[dict[str, object]] = []
    start = time.monotonic()
    for position, decision in enumerate(decisions, 1):
        destination = record_dir / f"{int(decision['image_id']):05d}_{int(decision['attribute_id']):03d}.json"
        if destination.is_file():
            saved = read_json(destination)
            require(
                saved["protocol_digest"] == protocol_digest
                and saved["decision_id"] == decision["decision_id"],
                "Phase 5 resume identity differs",
            )
            rows.extend(saved["rows"])
            continue
        try:
            image_path = cub_root / "images" / str(decision["relative_path"])
            shuffled_path = cub_root / "images" / str(decision["shuffled_relative_path"])
            with Image.open(image_path) as opened:
                image = opened.convert("RGB")
                original = runner.evaluate(
                    image,
                    str(decision["prompt_text"]),
                    include_attention=bool(config["question_conditioned_llm_attention"]),
                    generate=True,
                    max_new_tokens=int(config["max_new_tokens"]),
                )
            with Image.open(shuffled_path) as opened:
                shuffled = runner.evaluate(
                    opened.convert("RGB"),
                    str(decision["prompt_text"]),
                    include_attention=False,
                    generate=True,
                    max_new_tokens=int(config["max_new_tokens"]),
                )
            prompt_only = runner.evaluate(
                None,
                str(decision["prompt_text"]),
                include_attention=False,
                generate=True,
                max_new_tokens=int(config["max_new_tokens"]),
            )
            decision_rows = [
                output_row(decision, "image", original),
                output_row(decision, "image_shuffled", shuffled),
                output_row(decision, "prompt_only", prompt_only),
            ]
            atomic_json_write(
                {
                    "schema_version": 1,
                    "protocol_digest": protocol_digest,
                    "decision_id": decision["decision_id"],
                    "rows": decision_rows,
                },
                destination,
            )
            rows.extend(decision_rows)
        except Exception as error:
            atomic_json_write(
                {
                    "schema_version": 1,
                    "status": "FAIL",
                    "mode": args.mode,
                    "protocol_digest": protocol_digest,
                    "failed_position": position,
                    "failed_decision_id": decision["decision_id"],
                    "completed_decisions": len(rows) // 3,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                    "resumable": True,
                    "official_test_images_used": 0,
                },
                args.output_dir / "phase5_failure_report.json",
            )
            raise
        print(
            f"[{position}/{len(decisions)}] {decision['decision_id']} "
            f"elapsed={time.monotonic() - start:.1f}s",
            flush=True,
        )

    require(len(rows) == len(decisions) * 3, "Phase 5 control rows are incomplete")
    require(len({(row["decision_id"], row["control"]) for row in rows}) == len(rows), "duplicate output rows")
    output_path = args.output_dir / "phase5_decisions.csv"
    write_csv(output_path, rows)
    counts = Counter(str(row["cohort"]) for row in decisions)
    report = {
        "schema_version": 1,
        "status": "PASS",
        "mode": args.mode,
        "protocol_digest": protocol_digest,
        "policy_digest": policy_digest,
        "git_commit": identity["git_commit"],
        "decision_id_schema": "image_id::attribute_id::prompt_id",
        "prompt_id": config["prompt_id"],
        "decisions": len(decisions),
        "decision_ids": identity["decision_ids"],
        "control_rows": len(rows),
        "cohort_counts": dict(counts),
        "attributes": len({int(row["attribute_id"]) for row in decisions}),
        "positive_token_ids": list(runner.positive_token_ids),
        "negative_token_ids": list(runner.negative_token_ids),
        "phase3_concurrent": phase3_concurrent,
        "cross_phase_analysis_blocked_until_phase3r": phase3_concurrent,
        "artifact_manifest": {
            "evaluation_config.json": {"sha256": sha256(args.output_dir / "evaluation_config.json")},
            "decision_manifest.csv": {"sha256": sha256(manifest_path)},
            "phase5_decisions.csv": {"sha256": sha256(output_path)},
        },
        "runtime_seconds": time.monotonic() - start,
        "official_test_images_used": 0,
        "interpretation": "answer utilization with language-only and image-shuffled controls; attention is diagnostic only",
    }
    atomic_json_write(report, args.output_dir / "phase5_run_report.json")
    (args.output_dir / "phase5_failure_report.json").unlink(missing_ok=True)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
