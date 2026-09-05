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
REQUIRED_CONCEPTS = ("bird", "birds")
REQUIRED_SELECTORS = (
    "vision_cls_attention",
    "logit_concept",
    "attention_logit_fusion",
)
SELECTOR_LABELS = {
    "vision_cls_attention": "Vision-CLS attention",
    "logit_concept": "Corrected concept logit",
    "attention_logit_fusion": "Attention/concept fusion",
}
REJECTED_CONCEPT_TOKEN_IDS = {29871}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _normalized_token(token: object) -> str:
    return str(token).replace("▁", "").replace("Ġ", "").strip().lower()


def _concept_cache_errors(config: dict[str, object]) -> list[str]:
    errors: list[str] = []
    token_ids = config.get("concept_token_ids")
    tokens = config.get("concept_tokens")
    if config.get("concept_tokenization_policy") != TOKENIZATION_POLICY:
        errors.append("concept tokenization policy is not single_lexical_token_v1")
    fixed_concepts = config.get("fixed_concepts")
    if fixed_concepts != list(REQUIRED_CONCEPTS):
        errors.append("fixed concepts are not exactly ['bird', 'birds']")
    if not isinstance(token_ids, list) or not isinstance(tokens, list):
        errors.append("concept token IDs/tokens are missing or malformed")
        return errors
    if not token_ids or len(token_ids) != len(tokens):
        errors.append("concept token IDs and decoded tokens do not align")
    else:
        try:
            parsed_ids = [int(token_id) for token_id in token_ids]
        except (TypeError, ValueError):
            errors.append("concept token IDs are not integers")
        else:
            rejected = sorted(REJECTED_CONCEPT_TOKEN_IDS.intersection(parsed_ids))
            if rejected:
                errors.append(f"concept token IDs include rejected IDs {rejected}")
        normalized = [_normalized_token(token) for token in tokens]
        if normalized != list(REQUIRED_CONCEPTS):
            errors.append(
                "decoded concept tokens do not resolve exactly to ['bird', 'birds']"
            )
    return errors


def _configured_path_matches(configured: object, expected: Path) -> bool:
    if not isinstance(configured, str) or not configured:
        return False
    return Path(configured).expanduser().resolve() == expected.expanduser().resolve()


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
    official_test_ids: list[int] = []
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
            official_test_ids.append(record.image_id)
            metadata_errors.append(f"image {record.image_id}: uses official test split")

    extraction_config_path = args.cache_dir / "extraction_config.json"
    extraction_config = json.loads(extraction_config_path.read_text(encoding="utf-8"))
    concept_errors = _concept_cache_errors(extraction_config)
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
    evaluation_errors: list[str] = []
    if int(evaluation_config.get("schema_version", 0)) != 2:
        evaluation_errors.append("evaluation schema version is not 2")
    if not _configured_path_matches(evaluation_config.get("manifest"), args.manifest):
        evaluation_errors.append("evaluation manifest does not match the audited manifest")
    if not _configured_path_matches(evaluation_config.get("cache_dir"), args.cache_dir):
        evaluation_errors.append("evaluation cache_dir does not match the audited cache")
    if evaluation_config.get("cache_config") != extraction_config:
        evaluation_errors.append(
            "evaluation's embedded cache config does not match extraction_config.json"
        )
    if int(evaluation_config.get("official_test_images_used", -1)) != 0:
        evaluation_errors.append("evaluation does not record zero official test images")
    if not set(REFERENCE_VISION_ACCURACY).issubset(k_values):
        evaluation_errors.append("evaluation is missing required K=16/K=32 settings")
    evaluated_selectors = set(evaluation_config.get("selectors", []))
    if not set(REQUIRED_SELECTORS).issubset(evaluated_selectors):
        evaluation_errors.append("evaluation is missing one or more required selectors")
    configured_probe_seeds = {
        int(value) for value in evaluation_config.get("probe_seeds", [])
    }
    if len(configured_probe_seeds) < 3:
        evaluation_errors.append("evaluation config records fewer than three probe seeds")

    correction_errors: list[str] = []
    correction_summary_path = args.cache_dir / "correction_summary.json"
    cache_was_repaired = isinstance(extraction_config.get("cache_correction"), dict)
    correction_summary: dict[str, object] | None = None
    if cache_was_repaired:
        if not correction_summary_path.is_file():
            correction_errors.append("repaired cache is missing correction_summary.json")
        else:
            correction_summary = json.loads(
                correction_summary_path.read_text(encoding="utf-8")
            )
            for field in (
                "fixed_concepts",
                "concept_token_ids",
                "concept_tokens",
                "concept_tokenization_policy",
            ):
                if correction_summary.get(field) != extraction_config.get(field):
                    correction_errors.append(
                        f"correction summary {field} does not match extraction config"
                    )
            if not _configured_path_matches(
                correction_summary.get("output_dir"), args.cache_dir
            ):
                correction_errors.append(
                    "correction summary output_dir does not match the audited cache"
                )
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
        if cache_was_repaired:
            for field in (
                "concept_token_ids",
                "concept_tokens",
                "concept_tokenization_policy",
            ):
                observed = record.get(field)
                expected_value = extraction_config.get(field)
                if isinstance(observed, tuple):
                    observed = list(observed)
                if observed != expected_value:
                    cache_errors.append(
                        f"image {image_id}: cache {field} does not match extraction config"
                    )
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
    if correction_summary is not None:
        requested = int(correction_summary.get("requested_records", -1))
        completed = int(correction_summary.get("new_records", -1)) + int(
            correction_summary.get("resumed_records", -1)
        )
        if requested != len(expected_ids) or completed != requested:
            correction_errors.append(
                "correction summary does not account for every manifest record"
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
    validation_predictions = _read_csv(
        args.results_dir / "validation_predictions.csv"
    )
    localization_metrics = _read_csv(args.results_dir / "localization_metrics.csv")
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
    required_pairs = {
        (selector, k)
        for selector in REQUIRED_SELECTORS
        for k in REFERENCE_VISION_ACCURACY
    }
    summary_pairs = {
        (row["selector"], int(row["K"]))
        for row in selector_summary
        if row.get("K") and row["selector"] in REQUIRED_SELECTORS
    }
    corrected_selector_rows_complete = required_pairs.issubset(summary_pairs)

    prediction_errors: list[str] = []
    prediction_coverage: dict[tuple[str, int, int], set[int]] = {}
    for row in validation_predictions:
        if row.get("selector") not in REQUIRED_SELECTORS:
            continue
        image_id = int(row["image_id"])
        expected = manifest_by_id.get(image_id)
        if expected is None or expected.split != "val":
            prediction_errors.append(
                f"validation prediction contains non-validation image {image_id}"
            )
            continue
        if int(row["target_label"]) != expected.label:
            prediction_errors.append(
                f"image {image_id}: prediction target label does not match manifest"
            )
        if row["target_class_name"] != expected.class_name:
            prediction_errors.append(
                f"image {image_id}: prediction target class does not match manifest"
            )
        key = (row["selector"], int(row["K"]), int(row["probe_seed"]))
        prediction_coverage.setdefault(key, set()).add(image_id)
    expected_prediction_groups = {
        (selector, k, seed)
        for selector, k in required_pairs
        for seed in configured_probe_seeds
    }
    for key in sorted(expected_prediction_groups):
        if prediction_coverage.get(key) != val_ids:
            prediction_errors.append(
                f"prediction coverage mismatch for {key[0]} K={key[1]} seed={key[2]}"
            )

    localization_errors: list[str] = []
    localization_coverage: dict[tuple[str, int], set[int]] = {}
    for row in localization_metrics:
        if row.get("selector") not in REQUIRED_SELECTORS or row.get("split") != "val":
            continue
        key = (row["selector"], int(row["K"]))
        localization_coverage.setdefault(key, set()).add(int(row["image_id"]))
    localization_summary_pairs = {
        (row["selector"], int(row["K"]))
        for row in localization_summary
        if row.get("K") and row["selector"] in REQUIRED_SELECTORS
    }
    for key in sorted(required_pairs):
        if localization_coverage.get(key) != val_ids:
            localization_errors.append(
                f"localization coverage mismatch for {key[0]} K={key[1]}"
            )
    if not required_pairs.issubset(localization_summary_pairs):
        localization_errors.append("localization summary is missing required rows")
    part_by_k: dict[int, list[dict[str, object]]] = {}
    for row in part_rows:
        part_by_k.setdefault(int(row["K"]), []).append(row)
    qualitative_dir = args.results_dir / "qualitative"
    qualitative_pngs = list(qualitative_dir.glob("*.png"))
    qualitative_count = len(qualitative_pngs)
    qualitative_errors: list[str] = []
    qualitative_index_path = qualitative_dir / "index.csv"
    if not qualitative_index_path.is_file():
        qualitative_errors.append("qualitative/index.csv is missing")
    else:
        qualitative_index = _read_csv(qualitative_index_path)
        indexed_ids: set[int] = set()
        for row in qualitative_index:
            image_id = int(row["image_id"])
            figure = Path(row["figure"])
            if image_id in indexed_ids:
                qualitative_errors.append(
                    f"qualitative index repeats image {image_id}"
                )
            indexed_ids.add(image_id)
            if image_id not in val_ids:
                qualitative_errors.append(
                    f"qualitative index contains non-validation image {image_id}"
                )
            if figure.name != str(figure) or not (qualitative_dir / figure).is_file():
                qualitative_errors.append(
                    f"qualitative figure is missing or unsafe: {figure}"
                )
        if not qualitative_index:
            qualitative_errors.append("qualitative index is empty")
    expected_part_pairs = {
        (image_id, k) for image_id in val_ids for k in REFERENCE_VISION_ACCURACY
    }
    observed_part_pairs = {(int(row["image_id"]), int(row["K"])) for row in part_rows}
    parts_complete = observed_part_pairs == expected_part_pairs and not part_errors

    anomaly_lines: list[str] = []
    localization_by_selector_k = {
        (row["selector"], int(row["K"])): row
        for row in localization_summary
        if row.get("K")
    }
    for k in sorted(REFERENCE_VISION_ACCURACY):
        vision = localization_by_selector_k.get(("vision_cls_attention", k))
        random = localization_by_selector_k.get(("random", k))
        if vision is None:
            continue
        inside = float(vision.get("inside_fraction_mean", "nan"))
        pointing = float(vision.get("pointing_game_mean", "nan"))
        random_pointing = (
            float(random.get("pointing_game_mean", "nan"))
            if random is not None
            else float("nan")
        )
        if inside >= 0.70 and pointing < 0.50:
            comparison = (
                f" versus {100 * random_pointing:.1f}% for Random"
                if random_pointing == random_pointing
                else ""
            )
            anomaly_lines.append(
                f"Vision-CLS K={k} puts {100 * inside:.1f}% of selected patches "
                f"inside the bird box, but its top-1 pointing rate is only "
                f"{100 * pointing:.1f}%{comparison}."
            )

    checks = {
        "manifest IDs unique": ids_unique,
        "manifest paths unique": paths_unique,
        "pilot train/validation disjoint": split_disjoint,
        "all pilot images come from official CUB training split": not metadata_errors,
        "cache records and metadata match manifest": not cache_errors,
        "concept cache is the audited bird/birds correction": not concept_errors,
        "repaired-cache provenance is complete": not correction_errors,
        "evaluation config matches manifest and corrected cache": not evaluation_errors,
        "Vision-CLS evaluated with at least three probe seeds": three_seed_check,
        "Vision-CLS K=16/K=32 accuracy reproduces within two validation images": reproduced,
        "corrected selector rows exist at K=16 and K=32": corrected_selector_rows_complete,
        "required validation predictions are complete and aligned": not prediction_errors,
        "Vision-CLS broad-box metrics exist at K=16 and K=32": bbox_complete,
        "corrected localization rows cover every validation image": not localization_errors,
        "qualitative index and figures are consistent": (
            qualitative_count > 0 and not qualitative_errors
        ),
        "Vision-CLS part metrics cover every validation image": parts_complete,
    }
    gate_passed = all(checks.values())
    gate_status = (
        "STOP / INVESTIGATE"
        if not gate_passed
        else "PASS WITH ANOMALY"
        if anomaly_lines
        else "PASS"
    )

    lines = [
        "# Phase 01 sanity report",
        "",
        f"**Stage-gate status: {gate_status}**",
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
            f"Official test images used: {len(official_test_ids)}.",
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
    if not concept_errors:
        lines.append(
            "The cache uses the audited single-lexical-token policy with exactly "
            f"`bird`/`birds` (IDs {extraction_config['concept_token_ids']}); rejected "
            "whitespace token ID 29871 is absent. Concept-logit and "
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
            "## Corrected selector audit",
            "",
            "All rows below use the same late-LLM visual-token states and matched "
            "linear-probe protocol. They measure species accessibility after routing, "
            "not raw vision classification or causal utilization.",
            "",
            "| Selector | K | Accuracy mean | Macro-F1 mean | Inside bird box | "
            "Pointing game |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    selector_rows_by_key = {
        (row["selector"], int(row["K"])): row
        for row in selector_summary
        if row.get("K") and row["selector"] in REQUIRED_SELECTORS
    }
    for selector in REQUIRED_SELECTORS:
        for k in sorted(REFERENCE_VISION_ACCURACY):
            row = selector_rows_by_key.get((selector, k), {})
            localization = localization_by_selector_k.get((selector, k), {})
            lines.append(
                f"| {SELECTOR_LABELS[selector]} | {k} | "
                f"{100 * float(row.get('accuracy_mean', 'nan')):.1f}% | "
                f"{100 * float(row.get('macro_f1_mean', 'nan')):.1f}% | "
                f"{100 * float(localization.get('inside_fraction_mean', 'nan')):.1f}% | "
                f"{100 * float(localization.get('pointing_game_mean', 'nan')):.1f}% |"
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
    blocking_findings = (
        metadata_errors
        + cache_errors
        + concept_errors
        + correction_errors
        + evaluation_errors
        + prediction_errors
        + localization_errors
        + qualitative_errors
        + part_errors
    )
    if blocking_findings:
        lines.extend(["", "## Blocking findings", ""])
        lines.extend(
            f"- {item}" for item in blocking_findings
        )
    if anomaly_lines:
        lines.extend(["", "## Nonfatal anomalies", ""])
        lines.extend(f"- {item}" for item in anomaly_lines)
        lines.append(
            "- Preserve this Top-K concentration/top-1 pointing mismatch in later "
            "interpretation; it is not evidence of precise part localization."
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    gate_record = {
        "schema_version": 1,
        "status": gate_status,
        "passed": gate_passed,
        "checks": checks,
        "blocking_findings": blocking_findings,
        "nonfatal_anomalies": anomaly_lines,
        "manifest": str(args.manifest.expanduser().resolve()),
        "cache_dir": str(args.cache_dir.expanduser().resolve()),
        "results_dir": str(args.results_dir.expanduser().resolve()),
        "concept_tokenization_policy": extraction_config.get(
            "concept_tokenization_policy"
        ),
        "concept_token_ids": extraction_config.get("concept_token_ids"),
        "concept_tokens": extraction_config.get("concept_tokens"),
        "probe_seeds": sorted(configured_probe_seeds),
        "k_values": sorted(k_values),
    }
    gate_path = args.results_dir / "phase1_gate.json"
    temporary_gate_path = gate_path.with_suffix(".json.tmp")
    temporary_gate_path.write_text(
        json.dumps(gate_record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_gate_path.replace(gate_path)
    print(f"Phase 01 sanity report written to {output}")
    print(f"Machine-readable gate written to {gate_path}")
    print(f"Stage-gate status: {gate_status}")
    if not gate_passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
