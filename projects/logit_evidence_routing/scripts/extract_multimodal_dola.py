#!/usr/bin/env python3
"""Extract restart-safe DeCo and DoLa image--text layer decoding on Kaggle."""

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
REPO = PROJECT.parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from lger.hf_llava import _resolve_language_projection  # noqa: E402
from lger.hf_utilization import HfLlavaDecisionRunner  # noqa: E402
from lger.multimodal_dola import (  # noqa: E402
    dynamic_layer_correction,
    dynamic_layer_contrast,
    semantic_binary_margin,
    stack_projected_layers,
)
from lger.reproducibility import current_git_commit  # noqa: E402
from lger.stage_cache import atomic_json_write, config_digest, write_or_validate_config  # noqa: E402


CONTROLS = ("image", "prompt_only", "image_shuffled")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"input CSV is empty: {path}")
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError("refusing to write an empty DoLa table")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def choose_decisions(
    rows: list[dict[str, str]], mode: str, pilot_decisions: int
) -> list[dict[str, str]]:
    ordered = sorted(
        rows,
        key=lambda row: (int(row["attribute_id"]), int(row["target"]), row["decision_id"]),
    )
    if mode == "development_full":
        return ordered
    if mode == "architecture":
        return ordered[:1]
    if mode == "smoke":
        return [
            next(row for row in ordered if int(row["target"]) == target)
            for target in (0, 1)
        ]
    if pilot_decisions < 6:
        raise RuntimeError("pilot requires at least six decisions")
    strata: dict[tuple[int, int], list[dict[str, str]]] = {}
    for row in ordered:
        strata.setdefault((int(row["attribute_id"]), int(row["target"])), []).append(row)
    selected: list[dict[str, str]] = []
    offset = 0
    while len(selected) < min(pilot_decisions, len(ordered)):
        added = False
        for key in sorted(strata):
            if offset < len(strata[key]):
                selected.append(strata[key][offset])
                added = True
                if len(selected) == min(pilot_decisions, len(ordered)):
                    break
        if not added:
            raise RuntimeError("could not construct a stratified DoLa pilot")
        offset += 1
    return selected


def runner_from_config(config: dict[str, Any]) -> Any:
    kwargs = {
        "revision": str(config["revision"]),
        "quantization": str(config["quantization"]),
        "positive_answer": str(config.get("positive_answer", "yes")),
        "negative_answer": str(config.get("negative_answer", "no")),
        "attention_layer_offset": int(config.get("attention_layer_offset", -2)),
    }
    adapter = str(config.get("adapter", "llava"))
    if adapter == "llava":
        return HfLlavaDecisionRunner.from_pretrained(str(config["model"]), **kwargs)
    if adapter == "qwen2_5_vl":
        from lger.hf_qwen import HfQwenDecisionRunner

        return HfQwenDecisionRunner.from_pretrained(str(config["model"]), **kwargs)
    raise RuntimeError(f"unsupported adapter: {adapter}")


def projections_for_runner(runner: Any, adapter: str) -> tuple[Any, Any]:
    if adapter == "llava":
        return runner.base.final_norm, runner.base.lm_head
    return _resolve_language_projection(runner.model)


def measure(
    runner: Any,
    *,
    adapter: str,
    image: Image.Image | None,
    prompt: str,
    candidate_indices: tuple[int, ...],
    plausibility_alpha: float,
    deco_alpha: float,
    deco_top_p: float,
    deco_top_k: int,
) -> dict[str, Any]:
    inputs, _, _ = runner._prepare(image, prompt)
    with torch.inference_mode():
        outputs = runner.model(
            **inputs,
            output_hidden_states=True,
            output_attentions=False,
            use_cache=False,
            return_dict=True,
        )
    hidden_states = tuple(outputs.hidden_states)
    if len(hidden_states) < 3:
        raise RuntimeError("model did not return embedding, candidate, and mature states")
    invalid = [index for index in candidate_indices if index <= 0 or index >= len(hidden_states) - 1]
    if invalid:
        raise RuntimeError(
            f"candidate indices {invalid} are outside intermediate hidden states "
            f"1..{len(hidden_states) - 2}"
        )
    final_norm, lm_head = projections_for_runner(runner, adapter)
    with torch.inference_mode():
        candidate_logits = stack_projected_layers(
            [hidden_states[index] for index in candidate_indices],
            final_norm=final_norm,
            lm_head=lm_head,
            query_positions=-1,
        ).float()
    mature_logits = outputs.logits[:, -1, :].float().to(candidate_logits.device)
    positive_ids = tuple(int(value) for value in runner.positive_token_ids)
    negative_ids = tuple(int(value) for value in runner.negative_token_ids)
    if len(positive_ids) != 1 or len(negative_ids) != 1:
        raise RuntimeError(
            "DoLa pilot currently requires single-token answer alternatives; use the "
            "separate answer-token likelihood runner for multi-token pairs"
        )
    answer_ids = torch.tensor(
        [positive_ids[0], negative_ids[0]], dtype=torch.long, device=candidate_logits.device
    )
    binary_candidates = candidate_logits.index_select(-1, answer_ids)
    binary_mature = mature_logits.index_select(-1, answer_ids)
    binary = dynamic_layer_contrast(binary_mature, binary_candidates)
    full = dynamic_layer_contrast(
        mature_logits,
        candidate_logits,
        plausibility_alpha=plausibility_alpha,
        masked_value=-1000.0,
    )
    binary_deco = dynamic_layer_correction(
        binary_mature,
        binary_candidates,
        alpha=deco_alpha,
        candidate_token_mask=torch.ones_like(binary_mature, dtype=torch.bool),
    )
    full_deco = dynamic_layer_correction(
        mature_logits,
        candidate_logits,
        alpha=deco_alpha,
        top_p=deco_top_p,
        top_k=deco_top_k,
    )
    binary_selected = int(binary.premature_indices.item())
    full_selected = int(full.premature_indices.item())
    binary_anchor = int(binary_deco.anchor_indices.item())
    full_anchor = int(full_deco.anchor_indices.item())
    ordinary_margin = float((binary_mature[0, 0] - binary_mature[0, 1]).cpu())
    binary_margin = float(
        semantic_binary_margin(
            binary.contrasted_logits, positive_index=0, negative_index=1
        ).item()
    )
    full_margin = float(
        semantic_binary_margin(
            full.contrasted_logits,
            positive_index=positive_ids[0],
            negative_index=negative_ids[0],
        ).item()
    )
    binary_deco_margin = float(
        semantic_binary_margin(
            binary_deco.corrected_logits, positive_index=0, negative_index=1
        ).item()
    )
    full_deco_margin = float(
        semantic_binary_margin(
            full_deco.corrected_logits,
            positive_index=positive_ids[0],
            negative_index=negative_ids[0],
        ).item()
    )
    return {
        "ordinary_semantic_margin": ordinary_margin,
        "binary_dola_semantic_margin": binary_margin,
        "full_vocab_dola_semantic_margin": full_margin,
        "binary_deco_semantic_margin": binary_deco_margin,
        "full_vocab_deco_semantic_margin": full_deco_margin,
        "binary_premature_layer": candidate_indices[binary_selected],
        "full_vocab_premature_layer": candidate_indices[full_selected],
        "binary_deco_anchor_layer": candidate_indices[binary_anchor],
        "full_vocab_deco_anchor_layer": candidate_indices[full_anchor],
        "binary_selected_jsd": float(binary.js_divergences[0, binary_selected].cpu()),
        "full_vocab_selected_jsd": float(full.js_divergences[0, full_selected].cpu()),
        "binary_deco_anchor_probability": float(
            binary_deco.anchor_probabilities.item()
        ),
        "full_vocab_deco_anchor_probability": float(
            full_deco.anchor_probabilities.item()
        ),
        "binary_deco_layer_scores": [
            float(value) for value in binary_deco.layer_candidate_max_probabilities[0].cpu()
        ],
        "full_vocab_deco_layer_scores": [
            float(value) for value in full_deco.layer_candidate_max_probabilities[0].cpu()
        ],
        "full_vocab_deco_candidate_count": int(
            full_deco.candidate_token_mask[0].sum().item()
        ),
        "binary_jsd_by_layer": [float(value) for value in binary.js_divergences[0].cpu()],
        "full_vocab_jsd_by_layer": [float(value) for value in full.js_divergences[0].cpu()],
        "positive_plausible_full_vocab": int(full.plausibility_mask[0, positive_ids[0]].item()),
        "negative_plausible_full_vocab": int(full.plausibility_mask[0, negative_ids[0]].item()),
        "positive_token_ids": list(positive_ids),
        "negative_token_ids": list(negative_ids),
        "hidden_state_count": len(hidden_states),
        "language_layer_count": len(hidden_states) - 1,
        "vocabulary_size": int(mature_logits.shape[-1]),
    }


def result_row(
    decision: dict[str, str], control: str, values: dict[str, Any], adapter: str
) -> dict[str, Any]:
    target = int(decision["target"])
    evaluated = {
        "image": decision["image_id"],
        "prompt_only": "",
        "image_shuffled": decision["shuffled_image_id"],
    }[control]
    ordinary = float(values["ordinary_semantic_margin"])
    binary = float(values["binary_dola_semantic_margin"])
    full = float(values["full_vocab_dola_semantic_margin"])
    binary_deco = float(values["binary_deco_semantic_margin"])
    full_deco = float(values["full_vocab_deco_semantic_margin"])
    sign = 1 if target else -1
    return {
        "schema_version": 1,
        "decision_id": decision["decision_id"],
        "image_id": decision["image_id"],
        "class_id": decision.get("class_id", ""),
        "attribute_id": decision["attribute_id"],
        "attribute_name": decision.get("attribute_name", ""),
        "target": target,
        "control": control,
        "evaluated_image_id": evaluated,
        "adapter": adapter,
        "ordinary_semantic_margin": ordinary,
        "ordinary_correct_margin": sign * ordinary,
        "ordinary_correct": int((ordinary > 0) == bool(target) and ordinary != 0),
        "binary_dola_semantic_margin": binary,
        "binary_dola_correct_margin": sign * binary,
        "binary_dola_correct": int((binary > 0) == bool(target) and binary != 0),
        "full_vocab_dola_semantic_margin": full,
        "full_vocab_dola_correct_margin": sign * full,
        "full_vocab_dola_correct": int((full > 0) == bool(target) and full != 0),
        "binary_deco_semantic_margin": binary_deco,
        "binary_deco_correct_margin": sign * binary_deco,
        "binary_deco_correct": int(
            (binary_deco > 0) == bool(target) and binary_deco != 0
        ),
        "full_vocab_deco_semantic_margin": full_deco,
        "full_vocab_deco_correct_margin": sign * full_deco,
        "full_vocab_deco_correct": int(
            (full_deco > 0) == bool(target) and full_deco != 0
        ),
        "binary_premature_layer": values["binary_premature_layer"],
        "full_vocab_premature_layer": values["full_vocab_premature_layer"],
        "binary_deco_anchor_layer": values["binary_deco_anchor_layer"],
        "full_vocab_deco_anchor_layer": values["full_vocab_deco_anchor_layer"],
        "binary_selected_jsd": values["binary_selected_jsd"],
        "full_vocab_selected_jsd": values["full_vocab_selected_jsd"],
        "binary_deco_anchor_probability": values["binary_deco_anchor_probability"],
        "full_vocab_deco_anchor_probability": values[
            "full_vocab_deco_anchor_probability"
        ],
        "binary_deco_layer_scores": json.dumps(values["binary_deco_layer_scores"]),
        "full_vocab_deco_layer_scores": json.dumps(
            values["full_vocab_deco_layer_scores"]
        ),
        "full_vocab_deco_candidate_count": values[
            "full_vocab_deco_candidate_count"
        ],
        "binary_jsd_by_layer": json.dumps(values["binary_jsd_by_layer"]),
        "full_vocab_jsd_by_layer": json.dumps(values["full_vocab_jsd_by_layer"]),
        "positive_plausible_full_vocab": values["positive_plausible_full_vocab"],
        "negative_plausible_full_vocab": values["negative_plausible_full_vocab"],
        "positive_token_ids": json.dumps(values["positive_token_ids"]),
        "negative_token_ids": json.dumps(values["negative_token_ids"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("architecture", "smoke", "pilot", "development_full"), required=True
    )
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument(
        "--experiment-config", type=Path, default=PROJECT / "configs/multimodal_dola.json"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-start", type=int, required=True)
    parser.add_argument("--candidate-end", type=int, required=True)
    parser.add_argument("--candidate-stride", type=int, default=2)
    parser.add_argument("--pilot-decisions", type=int, default=48)
    parser.add_argument("--layout-report", type=Path,
                        help="required passing Qwen architecture report")
    parser.add_argument("--architecture-dir", type=Path)
    parser.add_argument("--smoke-dir", type=Path)
    parser.add_argument("--pilot-dir", type=Path)
    args = parser.parse_args()

    model_config = read_json(args.model_config)
    experiment = read_json(args.experiment_config)
    if not (
        experiment.get("schema_version") == 1
        and experiment.get("development_only") is True
        and experiment.get("official_test_images_used") == 0
    ):
        raise RuntimeError("multimodal layer decoding must remain development-only")
    manifest = read_csv(args.manifest)
    required = {
        "decision_id", "image_id", "attribute_id", "target", "prompt_text",
        "relative_path", "shuffled_image_id", "shuffled_relative_path",
    }
    if not all(required <= set(row) for row in manifest):
        raise RuntimeError(f"manifest lacks required columns: {sorted(required)}")
    if "official_split" in manifest[0] and any(
        row["official_split"].strip().casefold() != "train" for row in manifest
    ):
        raise RuntimeError("development layer decoding cannot access official-test images")
    if args.candidate_stride <= 0 or args.candidate_start >= args.candidate_end:
        raise RuntimeError("candidate range must be increasing with a positive stride")
    candidate_indices = tuple(
        range(args.candidate_start, args.candidate_end, args.candidate_stride)
    )
    if not candidate_indices:
        raise RuntimeError("candidate layer set is empty")
    decisions = choose_decisions(manifest, args.mode, args.pilot_decisions)
    adapter = str(model_config.get("adapter", "llava"))
    alpha = float(experiment["paper_faithful_sensitivity"]["adaptive_plausibility_alpha"])
    deco_alpha = float(experiment["deco_primary"]["alpha"])
    deco_top_p = float(experiment["deco_primary"]["top_p"])
    deco_top_k = int(experiment["deco_primary"]["top_k"])
    policy = {
        "schema_version": 1,
        "purpose": "multimodal_deco_dola_development_extraction",
        "git_commit": current_git_commit(REPO),
        "adapter": adapter,
        "model": model_config["model"],
        "revision": model_config["revision"],
        "quantization": model_config["quantization"],
        "model_config_sha256": sha256(args.model_config),
        "experiment_config_sha256": sha256(args.experiment_config),
        "manifest_sha256": sha256(args.manifest),
        "candidate_indices": list(candidate_indices),
        "plausibility_alpha": alpha,
        "deco_alpha": deco_alpha,
        "deco_top_p": deco_top_p,
        "deco_top_k": deco_top_k,
        "layout_report_sha256": sha256(args.layout_report) if args.layout_report else None,
        "controls": list(CONTROLS),
        "official_test_images_used": 0,
    }
    policy_digest = config_digest(policy)
    if adapter == "qwen2_5_vl":
        if args.layout_report is None:
            raise RuntimeError(
                "Qwen layer decoding requires --layout-report from the architecture gate"
            )
        layout = read_json(args.layout_report)
        if not (
            layout.get("status") == "PASS"
            and layout.get("model") == model_config["model"]
            and layout.get("resolved_revision") == model_config["revision"]
            and layout.get("official_test_images_used") == 0
        ):
            raise RuntimeError("Qwen layout report is absent, failing, or for another checkpoint")
    if args.mode != "architecture":
        if args.architecture_dir is None:
            raise RuntimeError("--architecture-dir is required after architecture mode")
        prior = read_json(args.architecture_dir / "multimodal_dola_report.json")
        if not (
            prior.get("status") == "PASS" and prior.get("mode") == "architecture"
            and prior.get("policy_digest") == policy_digest
        ):
            raise RuntimeError(
                "a passing policy-matched layer-decoding architecture run is required"
            )
    if args.mode in ("pilot", "development_full"):
        if args.smoke_dir is None:
            raise RuntimeError("--smoke-dir is required for pilot/full modes")
        prior = read_json(args.smoke_dir / "multimodal_dola_report.json")
        if not (
            prior.get("status") == "PASS" and prior.get("mode") == "smoke"
            and prior.get("policy_digest") == policy_digest
        ):
            raise RuntimeError("a passing policy-matched layer-decoding smoke is required")
    if args.mode == "development_full":
        if args.pilot_dir is None:
            raise RuntimeError("--pilot-dir is required for development_full mode")
        prior = read_json(args.pilot_dir / "multimodal_dola_report.json")
        if not (
            prior.get("status") == "PASS" and prior.get("mode") == "pilot"
            and prior.get("policy_digest") == policy_digest
        ):
            raise RuntimeError("a passing policy-matched layer-decoding pilot is required")
    identity = {
        **policy,
        "mode": args.mode,
        "decision_ids": [row["decision_id"] for row in decisions],
    }
    protocol_digest = config_digest(identity)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_or_validate_config(args.output_dir / "evaluation_config.json", identity)
    records = args.output_dir / "records"
    records.mkdir(exist_ok=True)

    print(
        f"Loading {model_config['model']} for {len(decisions)} decisions x "
        f"{len(CONTROLS)} controls; candidates={candidate_indices}.",
        flush=True,
    )
    runner = runner_from_config(model_config)
    if runner.resolved_revision != model_config["revision"]:
        raise RuntimeError("resolved model revision differs from the frozen config")
    output: list[dict[str, Any]] = []
    started = time.monotonic()
    total = len(decisions) * len(CONTROLS)
    completed = 0
    for decision in decisions:
        with Image.open(args.image_root / decision["relative_path"]) as opened:
            image = opened.convert("RGB")
        with Image.open(args.image_root / decision["shuffled_relative_path"]) as opened:
            shuffled = opened.convert("RGB")
        control_images = {"image": image, "prompt_only": None, "image_shuffled": shuffled}
        for control in CONTROLS:
            record_id = hashlib.sha256(
                f"{decision['decision_id']}::{control}".encode("utf-8")
            ).hexdigest()[:24]
            destination = records / f"{record_id}.json"
            if destination.is_file():
                saved = read_json(destination)
                if saved.get("protocol_digest") != protocol_digest:
                    raise RuntimeError("layer-decoding resume identity differs")
                output.append(saved["row"])
                completed += 1
                continue
            try:
                values = measure(
                    runner,
                    adapter=adapter,
                    image=control_images[control],
                    prompt=decision["prompt_text"],
                    candidate_indices=candidate_indices,
                    plausibility_alpha=alpha,
                    deco_alpha=deco_alpha,
                    deco_top_p=deco_top_p,
                    deco_top_k=deco_top_k,
                )
                row = result_row(decision, control, values, adapter)
                atomic_json_write(
                    {"schema_version": 1, "protocol_digest": protocol_digest, "row": row},
                    destination,
                )
                output.append(row)
                completed += 1
                elapsed = time.monotonic() - started
                eta = elapsed / completed * (total - completed)
                print(
                    f"[{completed}/{total}] {decision['decision_id']} {control}; "
                    f"elapsed={elapsed / 60:.1f}m eta={eta / 60:.1f}m",
                    flush=True,
                )
            except Exception as error:
                atomic_json_write(
                    {
                        "schema_version": 1,
                        "protocol_digest": protocol_digest,
                        "status": "FAIL",
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "traceback": traceback.format_exc(),
                    },
                    args.output_dir / "multimodal_dola_report.json",
                )
                raise

    output.sort(key=lambda row: (row["decision_id"], row["control"]))
    write_csv(args.output_dir / "multimodal_dola_decisions.csv", output)
    elapsed = time.monotonic() - started
    layer_counts = sorted({
        int(read_json(path)["row"].get("binary_premature_layer", -1))
        for path in records.glob("*.json")
        if read_json(path).get("row")
    })
    deco_layer_counts = sorted({
        int(read_json(path)["row"].get("binary_deco_anchor_layer", -1))
        for path in records.glob("*.json")
        if read_json(path).get("row")
    })
    report = {
        "schema_version": 1,
        "status": "PASS",
        "mode": args.mode,
        "policy_digest": policy_digest,
        "protocol_digest": protocol_digest,
        "adapter": adapter,
        "resolved_revision": runner.resolved_revision,
        "decisions": len(decisions),
        "rows": len(output),
        "candidate_indices": list(candidate_indices),
        "observed_binary_premature_layers": layer_counts,
        "observed_binary_deco_anchor_layers": deco_layer_counts,
        "runtime_seconds": elapsed,
        "seconds_per_model_evaluation": elapsed / max(total, 1),
        "official_test_images_used": 0,
        "claim_boundary": experiment["claim_boundary"],
    }
    atomic_json_write(report, args.output_dir / "multimodal_dola_report.json")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
