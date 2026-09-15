#!/usr/bin/env python3
"""Prepare or run the frozen CUB-only depth-allocation development pilot.

Preparation reads metadata only. GPU execution is restricted to Kaggle and never
opens an image until official-training and frozen development-val membership pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

import torch
from PIL import Image

from lger.allocation_manifest import select_pilot, table, validate_decisions
from lger.cub import load_cub_certainties, load_cub_image_attribute_labels, load_cub_part_locations
from lger.depth_allocation import (choose_crops, crop_at, grid_centers, inside,
                                   proxy_token_coverage, random_crops, select_depth_map)
from lger.hf_allocation import FULL, measure, prepare_views, selected_native_tokens
from lger.hf_qwen import HfQwenDecisionRunner


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def source_hashes() -> dict:
    paths = [Path(__file__), *sorted((PROJECT / "src" / "lger").glob("*.py"))]
    return {str(p.relative_to(PROJECT)): sha(p) for p in paths}


def candidates(cub: Path, development: list[dict], attributes: dict) -> list[dict]:
    # Verify the official split before considering labels or image access.
    from lger.allocation_manifest import official_metadata
    paths, splits = official_metadata(cub)
    ids = {int(r["image_id"]) for r in development if r["split"] == "val"}
    if not ids or any(splits.get(i) != 1 for i in ids):
        raise ValueError("development metadata contains official-test IDs")
    labels = load_cub_image_attribute_labels(cub, image_ids=ids)
    certainty = {c.certainty_id: c.name for c in load_cub_certainties(cub)}
    rows = []
    for image_id in sorted(ids):
        for label in labels[image_id]:
            if label.attribute_id not in attributes or certainty[label.certainty_id] not in ("probably", "definitely"):
                continue
            rows.append({"decision_id": f"cub-{image_id}-{label.attribute_id}", "image_id": image_id,
                         "attribute_id": label.attribute_id, "target": int(label.is_present),
                         "relative_path": paths[image_id], "certainty": certainty[label.certainty_id]})
    return validate_decisions(rows, development, cub, set(attributes))


def prompt(attribute: dict, pair: list[str], reverse: bool = False) -> str:
    order = pair[::-1] if reverse else pair
    # Same text for both targets; no species name or location annotation.
    return ("Use all supplied views of the same bird. Is this visual attribute present: "
            + attribute["name"].replace("::", ": ").replace("_", " ")
            + f"? Answer only {order[0]} or {order[1]}.")


def read_probes(path: Path | None, evaluation_ids: set[int], revision: str, training_ids: set[int]):
    if path is None:
        return None
    probes = json.loads(path.read_text())
    if probes["model_revision"] != revision or probes["pooling"] != "all_visual_tokens_mean":
        raise ValueError("frozen diagnostic probes have incompatible model or pooling")
    train_ids = {int(i) for i in probes["training_image_ids"]}
    if not train_ids or train_ids & evaluation_ids or not train_ids <= training_ids or probes.get("diagnostic_only") is not True:
        raise ValueError("probe provenance is missing, overlapping, or not diagnostic-only")
    return probes


def probe_scores(probes, attribute_id: int, states: torch.Tensor):
    if probes is None or str(attribute_id) not in probes["attributes"]:
        return None
    values = probes["attributes"][str(attribute_id)]
    weight, center, scale = [torch.tensor(values[key], dtype=torch.float32) for key in ("weight", "center", "scale")]
    bias = torch.tensor(values["bias"], dtype=torch.float32)
    if weight.shape != states.shape or center.shape != states.shape or scale.shape != states.shape or bool((scale <= 0).any()):
        raise ValueError("diagnostic probe dimensions/scales differ from frozen recipe")
    result = (((states - center) / scale) * weight).sum(-1) + bias
    if not bool(torch.isfinite(result).all()):
        raise ValueError("nonfinite diagnostic probe scores")
    return result.tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "smoke", "pilot", "robustness"), required=True)
    parser.add_argument("--config", type=Path, default=PROJECT / "configs/depth_allocation_bridge.json")
    parser.add_argument("--cub-root", type=Path, required=True)
    parser.add_argument("--development-manifest", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--smoke-report", type=Path)
    parser.add_argument("--frozen-probes", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    attributes_path = PROJECT / config["attribute_config"]
    model_path = PROJECT / config["model_config"]
    attributes = {int(a["attribute_id"]): a for a in json.loads(attributes_path.read_text())["attributes"]}
    model_config = json.loads(model_path.read_text())
    development = table(args.development_manifest)
    all_rows = candidates(args.cub_root, development, attributes)
    hashes = {"config": sha(args.config), "attribute_config": sha(attributes_path),
              "model_config": sha(model_path), "development_manifest": sha(args.development_manifest),
              "images_metadata": sha(args.cub_root / "images.txt"),
              "official_split": sha(args.cub_root / "train_test_split.txt"),
              "attribute_labels": sha(args.cub_root / "attributes/image_attribute_labels.txt"),
              "certainty_vocabulary": sha(args.cub_root / "attributes/certainties.txt"),
              "attribute_vocabulary": sha(args.cub_root / "attributes/attributes.txt"),
              "part_names": sha(args.cub_root / "parts/parts.txt"),
              "part_locations": sha(args.cub_root / "parts/part_locs.txt")}
    if args.mode == "prepare":
        if args.plan.exists():
            raise ValueError("refusing to overwrite an existing development plan")
        selected = select_pilot(all_rows, set(attributes))
        args.plan.parent.mkdir(parents=True, exist_ok=True)
        write(args.plan, {"schema_version": 1, "hashes": hashes, "decisions": selected,
                         "expected_strata": len(attributes)*2, "selected_strata": len(selected),
                         "missing_strata": [[a, y] for a in sorted(attributes) for y in (0, 1)
                             if not any(int(r["attribute_id"]) == a and int(r["target"]) == y for r in selected)],
                         "official_test_images_used": 0, "pixels_opened": 0})
        print(f"Prepared {len(selected)} decisions without opening images: {args.plan}")
        return
    plan = json.loads(args.plan.read_text())
    if plan["hashes"] != hashes or plan["decisions"] != select_pilot(all_rows, set(attributes)):
        raise ValueError("plan inputs or deterministic cohort have changed")
    rows = validate_decisions(plan["decisions"], development, args.cub_root, set(attributes))
    if args.output_dir is None or args.output_dir.exists():
        raise ValueError("supply a new output directory; existing runs are immutable")
    if not Path("/kaggle/working").is_dir() or not torch.cuda.is_available():
        raise RuntimeError("real extraction is restricted to Kaggle GPU; local work is synthetic/metadata only")
    import transformers
    if transformers.__version__ != model_config["transformers_version"]:
        raise RuntimeError("exact pinned Transformers version is required")
    current_sources = source_hashes()
    if args.mode != "smoke":
        if args.smoke_report is None:
            raise ValueError("a matching passed smoke report is required")
        smoke = json.loads(args.smoke_report.read_text())
        if (smoke["status"] != "PASS" or smoke["plan_sha256"] != sha(args.plan)
                or smoke["source_hashes"] != current_sources):
            raise ValueError("smoke gate does not match current plan/source")
    if args.mode == "smoke":
        rows = [next(r for r in rows if int(r["target"]) == y) for y in (0, 1)]
    if args.mode == "robustness":
        rows = rows[:12]  # fixed attribute/target order, never selected by model success
    from lger.allocation_manifest import official_metadata
    _, official_splits = official_metadata(args.cub_root)
    training_ids = {int(r["image_id"]) for r in development
                    if r["split"] == "train" and official_splits.get(int(r["image_id"])) == 1}
    probes = read_probes(args.frozen_probes, {int(r["image_id"]) for r in plan["decisions"]},
                         model_config["revision"], training_ids)
    runner = HfQwenDecisionRunner.from_pretrained(model_config["model"], revision=model_config["revision"],
                                               quantization=model_config["quantization"])
    if runner.resolved_revision != model_config["revision"]:
        raise RuntimeError("model resolved to a different revision")
    locations = load_cub_part_locations(args.cub_root)
    part_names = {}
    for line in (args.cub_root / "parts/parts.txt").read_text().splitlines():
        key, value = line.split(maxsplit=1)
        part_names[int(key)] = value.replace("_", " ").lower()
    args.output_dir.mkdir(parents=True)
    report = {"status": "RUNNING", "mode": args.mode, "hashes": hashes, "plan_sha256": sha(args.plan),
              "source_hashes": current_sources, "official_test_images_used": 0,
              "torch": torch.__version__, "transformers": transformers.__version__, "python": platform.python_version(),
              "gpu": torch.cuda.get_device_name(), "model_revision": runner.resolved_revision,
              "frozen_probe_sha256": sha(args.frozen_probes) if args.frozen_probes else None,
              "fixed_probe_status": "supplied_diagnostic_only" if probes else "N/A_no_compatible_frozen_Qwen_probe",
              "vision_cls_status": "N/A_Qwen_visual_tower_has_no_CLS_token",
              "compute_matching": "measured_token_sum_only; latency includes diagnostic collection; no FLOP claim"}
    report_path = args.output_dir / "report.json"
    write(report_path, report)
    expected_outcomes = 0
    try:
        with (args.output_dir / "outcomes.jsonl").open("x") as stream:
            for index, row in enumerate(rows):
                attribute = attributes[int(row["attribute_id"])]
                # Every selected path passed official metadata and development membership checks above.
                path = args.cub_root / "images" / row["relative_path"]
                with Image.open(path) as opened:
                    image = opened.convert("RGB")
                image_hash = sha(path)
                visible = [p for p in locations[int(row["image_id"])] if p.visible and
                           part_names[p.part_id] in attribute["relevant_parts"]]
                points = torch.tensor([[p.x / image.width, p.y / image.height] for p in visible],
                                      dtype=torch.float64).reshape(-1, 2)
                if len(points) and not bool(((points >= 0) & (points <= 1)).all()):
                    raise RuntimeError("visible landmark lies outside the image")
                canonical = prompt(attribute, ["yes", "no"])
                runner.set_answer_pair("yes", "no")
                inputs, grids, _ = prepare_views(runner, image, [FULL], [config["glance_tokens"]], canonical)
                native, aux = measure(runner, inputs, capture=True)
                null_prompt = "Use all supplied views of the same bird. " + config["null_question"] + " Answer only yes or no."
                null_inputs, null_grids, _ = prepare_views(runner, image, [FULL], [config["glance_tokens"]], null_prompt)
                null, null_aux = measure(runner, null_inputs, capture=True)
                if null_grids != grids:
                    raise RuntimeError("null question changed the image grid")
                maps, null_maps = aux["maps"], null_aux["maps"]
                proposals, selector_audit = {}, {}
                for name in ["spectral", *config["selector_baselines"]]:
                    proposal = select_depth_map(maps, null_maps, name)
                    proposals[name] = choose_crops(proposal.scores, grids[0], width=config["crop_width"])
                    selector_audit[name] = {"status": proposal.status, "top_rank": proposal.top_rank,
                                            "anchor_projection": proposal.anchor_projection}
                spectral = select_depth_map(maps, null_maps)
                proposals["spectral_multi"] = choose_crops(spectral.scores, grids[0], width=config["crop_width"], count=2)
                proposals["logit_concept"] = choose_crops(aux["logit_concept"], grids[0], width=config["crop_width"])
                for seed in config["random_seeds"]:
                    proposals[f"random_{seed}"] = random_crops(grids[0], width=config["crop_width"], count=1,
                        image_id=str(row["image_id"]), attribute_id=str(row["attribute_id"]), seed=seed)
                    proposals[f"random_multi_{seed}"] = random_crops(grids[0], width=config["crop_width"], count=2,
                        image_id=str(row["image_id"]), attribute_id=str(row["attribute_id"]), seed=seed)
                # Oracle is computed after all inference proposals are frozen, same rule for both labels.
                if len(points):
                    proposals["oracle_landmark"] = [crop_at(*points.mean(0).tolist(), config["crop_width"])]
                write(args.output_dir / f"{row['decision_id']}_proposals.json",
                      {"proposals": proposals, "selector_audit": selector_audit, "grid": grids[0]})
                variants = [(pair, reverse) for pair in config["answer_pairs"] for reverse in (False, True)] if args.mode == "robustness" else [(["yes", "no"], False)]
                for pair, reverse in variants:
                    runner.set_answer_pair(*pair)
                    text = prompt(attribute, pair, reverse)
                    arms = [("native", [FULL], [config["glance_tokens"]], None),
                            ("native_default", [FULL], None, None),
                            ("uniform_answer", [FULL], [config["answer_tokens"]], None),
                            ("uniform_total", [FULL], [config["total_uniform_tokens"]], None)]
                    for name, boxes in proposals.items():
                        remaining = config["answer_tokens"] - config["global_tokens"]
                        arms.append((name, [FULL, *boxes], [config["global_tokens"], *([remaining // len(boxes)] * len(boxes))], None))
                    for name in ("spectral", "random_0", "oracle_landmark"):
                        if name in proposals:
                            arms.extend([(f"attention_{name}", [FULL], [config["glance_tokens"]], ("attention", name)),
                                         (f"replace_{name}", [FULL], [config["glance_tokens"]], ("mean_replace", name))])
                    if args.mode == "robustness":
                        arms = [arm for arm in arms if arm[0] in ("native", "uniform_total", "spectral", "random_0", "random_1", "random_2")]
                    expected_outcomes += len(arms)
                    for name, boxes, budgets, internal in arms:
                        if name == "native_default":
                            arm_inputs, _, _ = runner._prepare(image, text)
                            merge_size = int(runner.model.config.vision_config.spatial_merge_size)
                            thw = arm_inputs["image_grid_thw"].tolist()
                            if len(thw) != 1 or thw[0][0] != 1:
                                raise RuntimeError("unsupported default-image grid")
                            arm_grids = [(thw[0][1] // merge_size, thw[0][2] // merge_size)]
                            actual_boxes = [FULL]
                        else:
                            arm_inputs, arm_grids, actual_boxes = prepare_views(runner, image, boxes, budgets, text)
                        selection = selected_native_tokens(grids[0], proposals[internal[1]]) if internal else None
                        if name == "native" and pair == ["yes", "no"] and not reverse:
                            result, measured = dict(native), aux
                        else:
                            result, measured = measure(runner, arm_inputs, capture=True,
                                cached_visual=aux["visual"] if internal else None, selected=selection,
                                intervention=internal[0] if internal else None, gain=config["internal_attention_gain"],
                                layer_fraction=config["internal_layer_fraction"])
                        margin = result["margin"]
                        target = int(row["target"])
                        result.update(decision_id=row["decision_id"], image_id=row["image_id"], attribute_id=row["attribute_id"],
                                      target=target, arm=name, pair=pair, reverse_order=reverse, image_sha256=image_hash,
                                      correct=int(margin != 0 and int(margin > 0) == target), correct_margin=margin * (2*target-1),
                                      boxes=actual_boxes, grids=arm_grids, visible_landmarks=len(points),
                                      evidence_scope="multiple_visible_landmarks" if len(points)>1 else "one_visible_landmark" if len(points)==1 else "not_evaluable",
                                      frozen_probe_scores=probe_scores(probes, int(row["attribute_id"]), measured["pooled_states"]))
                        result["encoding_policy"] = "processor_default" if name == "native_default" else "explicit_budget"
                        is_proposal = name in proposals and name != "oracle_landmark"
                        overhead = [native, null] if name in ("spectral", "spectral_multi", "contrast") else [native] if is_proposal and not name.startswith("random_") else []
                        if internal:
                            overhead = [native, null] if internal[1] == "spectral" else [native]
                        result["total_new_encoded_tokens"] = result["new_encoded_tokens"] + sum(x["new_encoded_tokens"] for x in overhead)
                        result["total_forward_seconds"] = result["forward_seconds"] + sum(x["forward_seconds"] for x in overhead)
                        result["charged_model_forwards"] = 1 + len(overhead)
                        result["proxy_tokens"] = {str(radius): proxy_token_coverage(points, actual_boxes, arm_grids, radius)
                                                  for radius in config["proxy_radii"]}
                        # Local crop coverage excludes the global view to keep this diagnostic informative.
                        local_boxes = actual_boxes[1:] if len(actual_boxes)>1 else []
                        covered = torch.zeros(len(points), dtype=torch.bool)
                        for box in local_boxes:
                            covered |= inside(points, box)
                        result["local_landmark_fraction"] = float(covered.double().mean()) if len(points) and local_boxes else None
                        result["local_complete_landmark_set"] = bool(covered.all()) if len(points) and local_boxes else None
                        # Same annotation-defined ROI regardless of target. Missing ROI stays N/A.
                        roi = torch.zeros(result["visual_tokens"], dtype=torch.bool)
                        offset = 0
                        for box, grid in zip(actual_boxes, arm_grids):
                            centers = grid_centers(grid)
                            centers[:, 0] = box[0] + centers[:, 0] * (box[2]-box[0])
                            centers[:, 1] = box[1] + centers[:, 1] * (box[3]-box[1])
                            for point in points:
                                roi[offset:offset+len(centers)] |= ((centers-point).abs().amax(1) <= 0.02)
                            offset += len(centers)
                        result["roi_patch_lens_by_layer"] = measured["patch_lens"][:, roi].mean(1).tolist() if bool(roi.any()) else None
                        stream.write(json.dumps(result, allow_nan=False) + "\n")
                        stream.flush()
                        if measured is not aux:
                            del measured
                del aux, null_aux
                print(f"Completed {index+1}/{len(rows)} decisions", flush=True)
        report.update(status="PASS", decisions=len(rows), expected_outcomes=expected_outcomes,
                      outcome_sha256=sha(args.output_dir / "outcomes.jsonl"))
    except Exception as error:
        report.update(status="FAIL", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write(report_path, report)


if __name__ == "__main__":
    main()
