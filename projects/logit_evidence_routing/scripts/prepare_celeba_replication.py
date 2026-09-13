#!/usr/bin/env python3
"""Build identity-disjoint CelebA development manifests without test-image access."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

CELEBA_IMAGE_COUNT = 202_599
CELEBA_TRAIN_END = 162_770
CELEBA_VALID_END = 182_637

from lger.stage_cache import atomic_json_write, config_digest  # noqa: E402


def read_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "DEVELOPMENT_PROTOCOL" or payload.get("official_test_images_used") != 0:
        raise RuntimeError("CelebA preparation requires an untouched development protocol")
    return payload


def first_existing(root: Path, relative_paths: list[str]) -> Path:
    for relative in relative_paths:
        path = root / relative
        if path.is_file():
            return path
    raise RuntimeError(f"CelebA file is missing; tried: {relative_paths}")


def first_existing_or_none(root: Path, relative_paths: list[str]) -> Path | None:
    for relative in relative_paths:
        path = root / relative
        if path.is_file():
            return path
    return None


def resolve_image_subdirectory(root: Path, configured: str) -> str:
    candidates = (configured.strip("/"), "img_celeba")
    for relative in candidates:
        if (root / relative).is_dir():
            return relative
    raise RuntimeError(
        "CelebA in-the-wild image directory is missing; do not substitute aligned crops "
        "because the configured boxes and landmarks use original-image coordinates"
    )


def official_partition_for_filename(filename: str) -> int:
    stem, suffix = Path(filename).stem, Path(filename).suffix.lower()
    if suffix != ".jpg" or len(stem) != 6 or not stem.isdigit():
        raise RuntimeError(f"unexpected CelebA image filename: {filename}")
    image_number = int(stem)
    if not 1 <= image_number <= CELEBA_IMAGE_COUNT:
        raise RuntimeError(f"CelebA image number is outside the official range: {filename}")
    if image_number <= CELEBA_TRAIN_END:
        return 0
    if image_number <= CELEBA_VALID_END:
        return 1
    return 2


def official_partition_from_image_directory(image_directory: Path) -> dict[str, int]:
    """Reconstruct CelebA's published contiguous official filename partitions."""
    image_names = [path.name for path in image_directory.iterdir() if path.is_file()]
    if len(image_names) != CELEBA_IMAGE_COUNT:
        raise RuntimeError(
            "CelebA package lacks list_eval_partition.txt and does not contain exactly "
            f"{CELEBA_IMAGE_COUNT} in-the-wild images"
        )
    partition = {name: official_partition_for_filename(name) for name in image_names}
    if len(partition) != CELEBA_IMAGE_COUNT:
        raise RuntimeError("CelebA in-the-wild image filenames are not unique")
    return partition


def simple_map(path: Path) -> dict[str, int]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        name, value = line.split()
        result[name] = int(value)
    return result


def annotation_table(path: Path, allowed_files: set[str]) -> tuple[list[str], dict[str, list[int]]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 3:
        raise RuntimeError(f"malformed CelebA annotation table: {path}")
    count = int(lines[0].strip())
    names = lines[1].split()
    rows = {}
    for line in lines[2:]:
        parts = line.split()
        if not parts or parts[0] not in allowed_files:
            continue
        values = [int(value) for value in parts[1:]]
        if len(values) != len(names):
            raise RuntimeError(f"annotation width differs: {parts[0]}")
        rows[parts[0]] = values
    if count < len(rows):
        raise RuntimeError("CelebA annotation count is inconsistent")
    return names, rows


def _rank(seed: int, split: str, filename: str) -> str:
    return hashlib.sha256(f"{seed}\0{split}\0{filename}".encode()).hexdigest()


def balanced_identity_sample(
    candidates: list[dict[str, Any]],
    *,
    count: int,
    attributes: list[str],
    seed: int,
    split: str,
) -> list[dict[str, Any]]:
    by_identity: dict[int, list[dict[str, Any]]] = {}
    for row in candidates:
        by_identity.setdefault(int(row["identity_id"]), []).append(row)
    representatives = [
        min(rows, key=lambda row: _rank(seed, split, row["filename"]))
        for rows in by_identity.values()
    ]
    if len(representatives) < count:
        raise RuntimeError(f"{split} lacks {count} distinct identities")
    rng = random.Random(seed + sum(ord(char) for char in split))
    selected: list[dict[str, Any]] = []
    remaining = list(representatives)
    label_counts = {(attribute, target): 0 for attribute in attributes for target in (0, 1)}
    target_quota = count / 2
    while len(selected) < count:
        best_score = None
        best: list[dict[str, Any]] = []
        for row in remaining:
            score = sum(max(0.0, target_quota - label_counts[(attribute, row["attributes"][attribute])])
                        for attribute in attributes)
            if best_score is None or score > best_score:
                best_score, best = score, [row]
            elif score == best_score:
                best.append(row)
        chosen = best[rng.randrange(len(best))]
        selected.append(chosen)
        remaining.remove(chosen)
        for attribute in attributes:
            label_counts[(attribute, chosen["attributes"][attribute])] += 1
    for attribute in attributes:
        positives = label_counts[(attribute, 1)]
        if not (0.35 * count <= positives <= 0.65 * count):
            raise RuntimeError(f"balanced sampling failed for {split}/{attribute}: {positives}/{count}")
    return sorted(selected, key=lambda row: row["filename"])


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--celeba-root", type=Path, required=True)
    parser.add_argument("--config", type=Path,
                        default=PROJECT / "configs/celeba_replication.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = read_config(args.config)
    image_subdirectory = resolve_image_subdirectory(
        args.celeba_root, str(config["image_subdirectory"])
    )
    image_directory = args.celeba_root / image_subdirectory
    partition_path = first_existing_or_none(args.celeba_root, [
        "Eval/list_eval_partition.txt", "list_eval_partition.txt"
    ])
    identity_path = first_existing(args.celeba_root, [
        "Anno/identity_CelebA.txt", "identity_CelebA.txt"
    ])
    attribute_path = first_existing(args.celeba_root, [
        "Anno/list_attr_celeba.txt", "list_attr_celeba.txt"
    ])
    landmark_path = first_existing(args.celeba_root, [
        "Anno/list_landmarks_celeba.txt", "list_landmarks_celeba.txt"
    ])
    bbox_path = first_existing(args.celeba_root, [
        "Anno/list_bbox_celeba.txt", "list_bbox_celeba.txt"
    ])
    if partition_path is None:
        partition = official_partition_from_image_directory(image_directory)
        partition_source = "published_official_filename_ranges"
    else:
        partition = simple_map(partition_path)
        partition_source = str(partition_path.relative_to(args.celeba_root))
    identity = simple_map(identity_path)
    # Official test rows are excluded before attribute or landmark values are parsed.
    development_files = {name for name, value in partition.items() if value in (0, 1)}
    attr_names, attrs = annotation_table(
        attribute_path, development_files
    )
    landmark_names, landmarks = annotation_table(
        landmark_path, development_files
    )
    bbox_names, boxes = annotation_table(
        bbox_path, development_files
    )
    selected = [row["name"] for row in config["selected_attributes"]]
    if not set(selected) <= set(attr_names):
        raise RuntimeError(f"CelebA lacks selected attributes: {set(selected) - set(attr_names)}")
    attr_index = {name: index for index, name in enumerate(attr_names)}
    candidates: dict[int, list[dict[str, Any]]] = {0: [], 1: []}
    for filename in sorted(development_files):
        if filename not in identity or filename not in attrs or filename not in landmarks or filename not in boxes:
            raise RuntimeError(f"CelebA annotations are incomplete for {filename}")
        candidates[partition[filename]].append({
            "filename": filename,
            "identity_id": identity[filename],
            "attributes": {
                name: int(attrs[filename][attr_index[name]] > 0) for name in selected
            },
            "bbox": dict(zip(bbox_names, boxes[filename])),
            "landmarks": dict(zip(landmark_names, landmarks[filename])),
        })

    train = balanced_identity_sample(
        candidates[0], count=int(config["development_train_images"]), attributes=selected,
        seed=int(config["sampling_seed"]), split="train",
    )
    train_ids = {row["identity_id"] for row in train}
    valid_candidates = [row for row in candidates[1] if row["identity_id"] not in train_ids]
    valid = balanced_identity_sample(
        valid_candidates, count=int(config["development_validation_images"]), attributes=selected,
        seed=int(config["sampling_seed"]), split="val",
    )
    if train_ids & {row["identity_id"] for row in valid}:
        raise RuntimeError("CelebA development splits are not identity-disjoint")

    image_rows = []
    decision_rows = []
    attributes = {row["name"]: row for row in config["selected_attributes"]}
    prompt_template = str(config["prompt_template"])
    for split, rows in (("train", train), ("val", valid)):
        for image_index, row in enumerate(rows):
            image_id = f"celeba-{split}-{image_index:05d}"
            image_rows.append({
                "image_id": image_id,
                "filename": row["filename"],
                "relative_path": f"{image_subdirectory}/{row['filename']}",
                "identity_id": row["identity_id"],
                "development_split": split,
                "official_partition": 0 if split == "train" else 1,
                "bbox_json": json.dumps(row["bbox"], sort_keys=True),
                "landmarks_json": json.dumps(row["landmarks"], sort_keys=True),
            })
            for attribute_id, name in enumerate(selected, 1):
                metadata = attributes[name]
                target = int(row["attributes"][name])
                decision_rows.append({
                    "decision_id": f"{image_id}::{attribute_id}::celeba_attribute_yes_no_v1",
                    "image_id": image_id,
                    "attribute_id": attribute_id,
                    "attribute_name": name,
                    "attribute_phrase": metadata["phrase"],
                    "target": target,
                    "prompt_text": prompt_template.format(attribute_phrase=metadata["phrase"]),
                    "relative_path": f"{image_subdirectory}/{row['filename']}",
                    "development_split": split,
                    "identity_id": row["identity_id"],
                    "relevant_landmarks_json": json.dumps(metadata["relevant_landmarks"]),
                    "official_test_image": 0,
                })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_path = args.output_dir / "celeba_development_images.csv"
    decision_path = args.output_dir / "celeba_development_decisions.csv"
    write_csv(image_path, image_rows)
    write_csv(decision_path, decision_rows)
    report = {
        "schema_version": 1,
        "status": "PASS",
        "purpose": config["purpose"],
        "protocol_digest": config_digest(config),
        "development_train_images": len(train),
        "development_validation_images": len(valid),
        "development_decisions": len(decision_rows),
        "selected_attributes": selected,
        "identity_disjoint": True,
        "max_images_per_identity": 1,
        "image_subdirectory": image_subdirectory,
        "official_partition_source": partition_source,
        "official_test_images_used": 0,
        "artifacts": {
            image_path.name: sha256(image_path),
            decision_path.name: sha256(decision_path),
        },
    }
    atomic_json_write(report, args.output_dir / "celeba_development_report.json")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
