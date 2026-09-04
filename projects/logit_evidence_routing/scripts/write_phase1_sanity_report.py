#!/usr/bin/env python3
"""Audit the Phase 01 Vision-CLS result and write its stage-gate report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lger.cub import (  # noqa: E402
    discover_cub_root,
    load_cub_part_locations,
    load_cub_records,
    load_pilot_manifest,
    map_point_to_center_crop,
)
from lger.localization import selection_part_metrics  # noqa: E402
from lger.phase1b import feature_key  # noqa: E402


TOKENIZATION_POLICY = "single_lexical_token_v1"
REFERENCE_VISION_ACCURACY = {16: 0.925, 32: 0.95}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _concept_cache_is_valid(config: dict[str, object]) -> bool:
    token_ids = config.get("concept_token_ids")
    tokens = config.get("concept_tokens")
    if config.get("concept_tokenization_policy") != TOKENIZATION_POLICY:
        return False
    if not isinstance(token_ids, list) or not isinstance(tokens, list):
        return False
    return bool(token_ids) and len(token_ids) == len(tokens) and all(
        str(token).replace("▁", "").replace("Ġ", "").strip()
        for token in tokens
    )


def _tensor_fingerprint(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--search-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--cub-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    cub_root = args.cub_root or discover_cub_root(args.search_root)
    output = args.output or args.results_dir / "phase1_sanity_report.md"
    manifest = load_pilot_manifest(args.manifest)
    official_by_id = {record.image_id: record for record in load_cub_records(cub_root)}
    manifest_by_id = {record.image_id: record for record in manifest}
    ids_unique = len(manifest_by_id) == len(manifest)
    paths = [record.relative_path for record in manifest]
    paths_unique = len(set(paths)) == len(paths)
    train_ids = {record.image_id for record in manifest if record.split == "train"}
    val_ids = {record.image_id for record in manifest if record.split == "val"}
    split_disjoint = train_ids.isdisjoint(val_ids)
    metadata_errors: list[str] = []
    for record in manifest:
        official = official_by_id.get(record.image_id)
        if official is None:
            metadata_errors.append(f"image {record.image_id}: absent from official metadata")
            continue
        expected = (official.relative_path, official.label, official.class_name)
        observed = (record.relative_path, record.label, record.class_name)
        if observed != expected:
            metadata_errors.append(f"image {record.image_id}: manifest metadata mismatch")
        if official.official_split != "train":
            metadata_errors.append(f"image {record.image_id}: uses official test split")

    extraction_config_path = args.cache_dir / "extraction_config.json"
    extraction_config = json.loads(extraction_config_path.read_text(encoding="utf-8"))
    concept_valid = _concept_cache_is_valid(extraction_config)
    cache_paths = sorted((args.cache_dir / "records").glob("*.pt"))
    cache_ids: set[int] = set()
    cache_errors: list[str] = []
    part_errors: list[str] = []
    fingerprints: dict[str, list[int]] = {}
    part_locations = load_cub_part_locations(cub_root)
    part_rows: list[dict[str, object]] = []
    evaluation_config = json.loads(
        (args.results_dir / "evaluation_config.json").read_text(encoding="utf-8")
    )
    k_values = [int(value) for value in evaluation_config["k_values"]]
    for path in cache_paths:
        record = torch.load(path, map_location="cpu", weights_only=False)
        image_id = int(record["image_id"])
        if image_id in cache_ids:
            cache_errors.append(f"duplicate cache record for image {image_id}")
            continue
        cache_ids.add(image_id)
        expected = manifest_by_id.get(image_id)
        if expected is None:
            cache_errors.append(f"unexpected cache image {image_id}")
            continue
        for field in ("relative_path", "label", "class_name", "split"):
            if record.get(field) != getattr(expected, field):
                cache_errors.append(f"image {image_id}: cache {field} mismatch")
        features = record.get("features", {})
        if "global_all" not in features:
            cache_errors.append(f"image {image_id}: missing global_all feature")
        else:
            digest = _tensor_fingerprint(features["global_all"])
            fingerprints.setdefault(digest, []).append(image_id)

        if expected.split == "val":
            original_size = tuple(int(value) for value in record["original_image_size"])
            image_size = tuple(int(value) for value in record["processed_image_size"])
            grid_size = tuple(int(value) for value in record["grid_size"])
            mapped_parts = []
            for part in part_locations.get(image_id, []):
                if not part.visible:
                    continue
                mapped = map_point_to_center_crop(
                    (part.x, part.y),
                    original_size=original_size,
                    output_size=(image_size[1], image_size[0]),
                )
                if mapped is not None:
                    mapped_parts.append(mapped)
            if not mapped_parts:
                part_errors.append(f"image {image_id}: no visible parts remain in crop")
                continue
            for k in k_values:
                key = feature_key("vision_cls_attention", k)
                selected = record.get("selections", {}).get(key)
                if not isinstance(selected, torch.Tensor):
                    part_errors.append(f"image {image_id}: missing {key}")
                    continue
                part_rows.append(
                    {
                        "image_id": image_id,
                        "K": k,
                        **selection_part_metrics(
                            selected,
                            mapped_parts,
                            grid_size=grid_size,
                            image_size=image_size,
                        ),
                    }
                )

    expected_ids = set(manifest_by_id)
    if cache_ids != expected_ids:
        missing = sorted(expected_ids - cache_ids)
        unexpected = sorted(cache_ids - expected_ids)
        cache_errors.append(
            f"cache/manifest ID mismatch: {len(missing)} missing, {len(unexpected)} unexpected"
        )
    duplicate_feature_groups = [ids for ids in fingerprints.values() if len(ids) > 1]
    if duplicate_feature_groups:
        cache_errors.append(
            f"{len(duplicate_feature_groups)} exact duplicate global-feature groups"
        )

    part_csv = args.results_dir / "vision_cls_part_localization.csv"
    part_csv.parent.mkdir(parents=True, exist_ok=True)
    with part_csv.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "image_id",
            "K",
            "part_patch_recall",
            "any_part_hit",
            "top1_part_hit",
            "top1_nearest_part_distance_patches",
            "visible_part_patches",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(part_rows)

    selector_summary = _read_csv(args.results_dir / "selector_summary.csv")
    selector_metrics = _read_csv(args.results_dir / "selector_metrics.csv")
    localization_summary = _read_csv(args.results_dir / "localization_summary.csv")
    vision_rows = [
        row for row in selector_summary if row["selector"] == "vision_cls_attention"
    ]
    vision_seed_rows = [
        row for row in selector_metrics if row["selector"] == "vision_cls_attention"
    ]
    probe_seeds = {int(row["probe_seed"]) for row in vision_seed_rows}
    three_seed_check = len(probe_seeds) >= 3
    accuracy_by_k = {int(row["K"]): float(row["accuracy_mean"]) for row in vision_rows}
    reproduced = all(
        k in accuracy_by_k and abs(accuracy_by_k[k] - expected) <= 0.0250001
        for k, expected in REFERENCE_VISION_ACCURACY.items()
    )
    bbox_by_k = {
        int(row["K"]): row
        for row in localization_summary
        if row["selector"] == "vision_cls_attention"
    }
    bbox_complete = all(k in bbox_by_k for k in REFERENCE_VISION_ACCURACY)
    part_by_k: dict[int, list[dict[str, object]]] = {}
    for row in part_rows:
        part_by_k.setdefault(int(row["K"]), []).append(row)
    qualitative_count = len(
        list((args.results_dir / "qualitative").glob("*.png"))
    )
    part_image_ids = {int(row["image_id"]) for row in part_rows}
    parts_complete = part_image_ids == val_ids and not part_errors

    checks = {
        "manifest IDs unique": ids_unique,
        "manifest paths unique": paths_unique,
        "pilot train/validation disjoint": split_disjoint,
        "all pilot images come from official CUB training split": not metadata_errors,
        "cache records and metadata match manifest": not cache_errors,
        "Vision-CLS evaluated with at least three probe seeds": three_seed_check,
        "Vision-CLS K=16/K=32 accuracy reproduces within two validation images": reproduced,
        "Vision-CLS broad-box metrics exist at K=16 and K=32": bbox_complete,
        "Vision-CLS qualitative figures exist": qualitative_count > 0,
        "Vision-CLS part metrics cover every validation image": parts_complete,
    }
    gate_passed = all(checks.values())

    lines = [
        "# Phase 01 sanity report",
        "",
        f"**Stage-gate status: {'PASS' if gate_passed else 'STOP / INVESTIGATE'}**",
        "",
        "This report audits the surprising Vision-CLS patch result before the project "
        "moves to representation tracing. It does not treat attention as explanation or "
        "species-probe accuracy as evidence of causal use.",
        "",
        "## Integrity checks",
        "",
        "| Check | Status |",
        "|---|---|",
    ]
    lines.extend(
        f"| {name} | {'PASS' if passed else 'FAIL'} |" for name, passed in checks.items()
    )
    lines.extend(
        [
            "",
            f"Pilot size: {len(train_ids)} train + {len(val_ids)} validation images. "
            "Official test images used: 0.",
            f"Exact duplicate cached global features: {len(duplicate_feature_groups)} groups.",
            f"Qualitative Vision-CLS figures found: {qualitative_count}.",
            "",
            "## Vision-CLS reproduction",
            "",
            "| K | Accuracy mean ± std | Macro-F1 mean ± std | Inside bird box | "
            "Part-patch recall | Any-part hit | Top-1 distance to nearest part |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(vision_rows, key=lambda item: int(item["K"])):
        k = int(row["K"])
        parts = part_by_k.get(k, [])
        bbox = bbox_by_k.get(k, {})
        lines.append(
            f"| {k} | {100 * float(row['accuracy_mean']):.1f}% ± "
            f"{100 * float(row['accuracy_std']):.1f} | "
            f"{100 * float(row['macro_f1_mean']):.1f}% ± "
            f"{100 * float(row['macro_f1_std']):.1f} | "
            f"{100 * float(bbox.get('inside_fraction_mean', 'nan')):.1f}% | "
            f"{100 * _mean([float(item['part_patch_recall']) for item in parts]):.1f}% | "
            f"{100 * _mean([float(item['any_part_hit']) for item in parts]):.1f}% | "
            f"{_mean([float(item['top1_nearest_part_distance_patches']) for item in parts]):.2f} patches |"
        )
    lines.extend(
        [
            "",
            f"Probe initialization seeds: {sorted(probe_seeds)}. These reruns test the "
            "linear readout's optimization stability; Vision-CLS selection itself is "
            "deterministic.",
            "",
            "## Concept-logit correction",
            "",
        ]
    )
    if concept_valid:
        lines.append(
            "The cache uses the audited single-lexical-token policy. Concept-logit and "
            "attention/concept-fusion rows may be evaluated as diagnostics."
        )
    else:
        lines.append(
            "**Excluded:** `logit_concept` and `attention_logit_fusion`. This cache does "
            "not use the audited single-lexical-token policy; the earlier run included "
            "a standalone whitespace token. Repair the cache and regenerate results "
            "before interpreting those two rows."
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "Observation: Vision-CLS attention selects late LLM patch states that support "
            "a strong species probe and concentrate inside the bird box. Interpretation: "
            "this is evidence for a strong vision-side routing signal, not 95% accuracy "
            "from raw vision features. It does not yet show which attributes are encoded, "
            "whether information is "
            "lost at the projector/LLM boundary, or whether the VLM uses that evidence "
            "when generating an answer.",
        ]
    )
    if metadata_errors or cache_errors or part_errors:
        lines.extend(["", "## Anomalies", ""])
        lines.extend(
            f"- {item}" for item in metadata_errors + cache_errors + part_errors
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Phase 01 sanity report written to {output}")
    print(f"Stage-gate status: {'PASS' if gate_passed else 'STOP / INVESTIGATE'}")
    if not gate_passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
