"""Full CUB annotation audit and deterministic split/cohort manifests."""

from __future__ import annotations

import csv
import random
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from lger.cub import (
    count_known_attribute_extra_zero_rows,
    discover_cub_root,
    load_cub_attributes,
    load_cub_bounding_boxes,
    load_cub_certainties,
    load_cub_image_attribute_labels,
    load_cub_part_locations,
    load_cub_records,
)

from .core import SCHEMA_VERSION, atomic_write_json, atomic_write_jsonl, canonical_hash


REQUIRED_ANNOTATIONS = (
    "images.txt",
    "image_class_labels.txt",
    "train_test_split.txt",
    "classes.txt",
    "bounding_boxes.txt",
    "parts/parts.txt",
    "parts/part_locs.txt",
    "attributes/attributes.txt",
    "attributes/certainties.txt",
    "attributes/image_attribute_labels.txt",
)

# Recovered from retained per-decision artifacts, not inferred from outcomes.
HISTORICAL_CLASS_IDS = (
    11, 25, 36, 56, 65, 67, 73, 78, 92, 99,
    104, 108, 123, 125, 130, 131, 150, 159, 194, 195,
)
HISTORICAL_ATTRIBUTE_IDS = (
    2, 5, 7, 8, 96, 100, 101, 102, 103, 104, 105, 107, 111,
    117, 118, 120, 294, 295, 299, 305, 306, 308, 309, 310, 311, 312,
)


def _read_names(path: Path) -> dict[int, str]:
    values: dict[int, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            parts = raw.strip().split(maxsplit=1)
            if len(parts) != 2:
                raise ValueError(f"malformed name row at {path}:{line_number}")
            values[int(parts[0])] = parts[1]
    return values


def audit_cub(cub_root: Path, *, verify_all_images: bool = True) -> dict[str, Any]:
    cub_root = cub_root.expanduser().resolve()
    missing = [relative for relative in REQUIRED_ANNOTATIONS if not (cub_root / relative).is_file()]
    if missing:
        raise FileNotFoundError(
            "CUB is incomplete; images-only datasets are not sufficient. Missing: "
            + ", ".join(missing)
        )
    records = load_cub_records(cub_root)
    boxes = load_cub_bounding_boxes(cub_root)
    parts = load_cub_part_locations(cub_root)
    attributes = load_cub_attributes(cub_root)
    certainties = load_cub_certainties(cub_root)
    labels = load_cub_image_attribute_labels(cub_root)
    part_names = _read_names(cub_root / "parts" / "parts.txt")
    normalized_extra_zero_rows = count_known_attribute_extra_zero_rows(cub_root)

    image_ids = {record.image_id for record in records}
    failures: list[str] = []
    if len(records) != 11788:
        failures.append(f"expected 11788 images, found {len(records)}")
    if len({record.label for record in records}) != 200:
        failures.append("class count is not 200")
    if image_ids != set(boxes):
        failures.append("bounding-box image IDs do not exactly match images.txt")
    if image_ids != set(parts):
        failures.append("part-location image IDs do not exactly match images.txt")
    if image_ids != set(labels):
        failures.append("attribute-label image IDs do not exactly match images.txt")
    if len(attributes) != 312:
        failures.append(f"expected 312 attributes, found {len(attributes)}")
    if len(part_names) != 15:
        failures.append(f"expected 15 named parts, found {len(part_names)}")
    if verify_all_images:
        absent = [
            record.relative_path
            for record in records
            if not (cub_root / "images" / record.relative_path).is_file()
        ]
        if absent:
            failures.append(f"{len(absent)} image files are missing")
    if failures:
        raise ValueError("CUB integrity audit failed: " + "; ".join(failures))
    split_counts = Counter(record.official_split for record in records)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "cub_root": str(cub_root),
        "image_count": len(records),
        "class_count": len({record.label for record in records}),
        "attribute_count": len(attributes),
        "part_count": len(part_names),
        "official_split_counts": dict(sorted(split_counts.items())),
        "bounding_box_count": len(boxes),
        "part_row_count": sum(len(values) for values in parts.values()),
        "attribute_label_row_count": sum(len(values) for values in labels.values()),
        "known_six_column_attribute_rows_normalized": normalized_extra_zero_rows,
        "certainty_names": [value.name for value in certainties],
        "historical_class_ids_recovered": list(HISTORICAL_CLASS_IDS),
        "historical_attribute_ids_recovered": list(HISTORICAL_ATTRIBUTE_IDS),
    }


def stratified_training_partition(
    records: Iterable[Any], *, dev_fraction: float, seed: int
) -> dict[int, str]:
    if not 0 < dev_fraction < 1:
        raise ValueError("dev_fraction must be between zero and one")
    by_class: dict[int, list[int]] = defaultdict(list)
    for record in records:
        if record.official_split == "train":
            by_class[int(record.label)].append(int(record.image_id))
    rng = random.Random(seed)
    output: dict[int, str] = {}
    for label in sorted(by_class):
        image_ids = sorted(by_class[label])
        rng.shuffle(image_ids)
        count = min(len(image_ids) - 1, max(1, round(len(image_ids) * dev_fraction)))
        development = set(image_ids[:count])
        for image_id in image_ids:
            output[image_id] = "development" if image_id in development else "fit"
    return output


def _eligible_questions(
    cub_root: Path,
    *,
    image_ids: set[int],
    attribute_ids: set[int],
    allowed_certainty: set[str],
) -> list[dict[str, Any]]:
    attributes = {value.attribute_id: value for value in load_cub_attributes(cub_root)}
    certainty = {
        value.certainty_id: value.name.strip().casefold()
        for value in load_cub_certainties(cub_root)
    }
    labels = load_cub_image_attribute_labels(cub_root, image_ids=image_ids)
    output: list[dict[str, Any]] = []
    for image_id in sorted(image_ids):
        for label in labels[image_id]:
            if label.attribute_id not in attribute_ids:
                continue
            certainty_name = certainty[label.certainty_id]
            if certainty_name not in allowed_certainty:
                continue
            attribute = attributes[label.attribute_id]
            output.append(
                {
                    "question_id": f"cub-{image_id:05d}-a{label.attribute_id:03d}",
                    "image_id": image_id,
                    "attribute_id": label.attribute_id,
                    "attribute_name": attribute.name,
                    "attribute_group": attribute.group,
                    "target": int(label.is_present),
                    "certainty_id": label.certainty_id,
                    "certainty_name": certainty_name,
                    "prompt": f"Is the bird's {attribute.name.replace('has_', '').replace('::', ' ')} present? Answer yes or no.",
                }
            )
    return output


def build_manifests(
    cub_root: Path,
    output_dir: Path,
    *,
    seed: int = 20260916,
    dev_fraction: float = 1 / 3,
    max_expensive_images: int = 256,
    max_decisions_per_image: int = 4,
) -> dict[str, Any]:
    """Write cohort manifests without inspecting any model output."""

    audit = audit_cub(cub_root)
    records = load_cub_records(cub_root)
    partitions = stratified_training_partition(records, dev_fraction=dev_fraction, seed=seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    record_rows: list[dict[str, Any]] = []
    for record in records:
        if record.label not in HISTORICAL_CLASS_IDS:
            continue
        role = record.official_split
        if role == "train":
            role = partitions[record.image_id]
        record_rows.append({**asdict(record), "analysis_role": role})

    test_ids = {row["image_id"] for row in record_rows if row["analysis_role"] == "test"}
    questions = _eligible_questions(
        cub_root,
        image_ids=test_ids,
        attribute_ids=set(HISTORICAL_ATTRIBUTE_IDS),
        allowed_certainty={"probably", "definitely"},
    )
    # Select images before any model outputs. Alternate polarity while keeping
    # image diversity; each image contributes at most the declared cap.
    by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in questions:
        by_image[int(row["image_id"])].append(row)
    rng = random.Random(seed)
    candidate_images = sorted(by_image)
    rng.shuffle(candidate_images)
    selected_images = set(candidate_images[:max_expensive_images])
    expensive: list[dict[str, Any]] = []
    for image_id in sorted(selected_images):
        rows = sorted(
            by_image[image_id],
            key=lambda row: (row["target"], row["attribute_group"], row["attribute_id"]),
        )
        positives = [row for row in rows if row["target"] == 1]
        negatives = [row for row in rows if row["target"] == 0]
        rng.shuffle(positives)
        rng.shuffle(negatives)
        interleaved: list[dict[str, Any]] = []
        for index in range(max(len(positives), len(negatives))):
            if index < len(positives):
                interleaved.append(positives[index])
            if index < len(negatives):
                interleaved.append(negatives[index])
        expensive.extend(interleaved[:max_decisions_per_image])

    class_names = _read_names(cub_root / "classes.txt")
    attribute_names = _read_names(cub_root / "attributes" / "attributes.txt")
    atomic_write_json(output_dir / "cub_integrity_audit.json", audit)
    atomic_write_jsonl(output_dir / "image_manifest.jsonl", record_rows)
    atomic_write_jsonl(
        output_dir / "class_manifest.jsonl",
        [
            {"class_id": value, "class_name": class_names[value], "source": "historical_recovered"}
            for value in HISTORICAL_CLASS_IDS
        ],
    )
    atomic_write_jsonl(
        output_dir / "attribute_manifest.jsonl",
        [
            {
                "attribute_id": value,
                "attribute_name": attribute_names[value],
                "source": "historical_recovered",
            }
            for value in HISTORICAL_ATTRIBUTE_IDS
        ],
    )
    atomic_write_jsonl(output_dir / "question_manifest_all_test.jsonl", questions)
    atomic_write_jsonl(output_dir / "question_manifest_expensive.jsonl", expensive)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "manifest_hash": canonical_hash(
            {"images": record_rows, "questions": questions, "expensive": expensive}
        ),
        "population": "20 recovered historical CUB species; not all 200 species",
        "official_test_image_count": len(test_ids),
        "eligible_test_decision_count": len(questions),
        "expensive_image_count": len({row["image_id"] for row in expensive}),
        "expensive_decision_count": len(expensive),
        "expensive_positive_count": sum(row["target"] for row in expensive),
        "expensive_negative_count": sum(1 - row["target"] for row in expensive),
        "max_decisions_per_image": max_decisions_per_image,
    }
    atomic_write_json(output_dir / "manifest_summary.json", summary)
    return summary


def discover_under_kaggle(search_root: Path = Path("/kaggle/input")) -> Path:
    return discover_cub_root(search_root)
