"""CUB-200-2011 discovery, metadata parsing, and deterministic pilot splits."""

from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path


REQUIRED_CUB_FILES = (
    "images.txt",
    "image_class_labels.txt",
    "train_test_split.txt",
)


@dataclass(frozen=True)
class CubRecord:
    image_id: int
    relative_path: str
    label: int
    class_name: str
    official_split: str


@dataclass(frozen=True)
class PilotRecord:
    image_id: int
    relative_path: str
    label: int
    class_name: str
    split: str


def discover_cub_root(search_root: Path) -> Path:
    """Find one official CUB metadata root below a Kaggle input directory."""

    search_root = search_root.expanduser().resolve()
    if not search_root.exists():
        raise FileNotFoundError(f"CUB search root does not exist: {search_root}")

    candidates: list[Path] = []
    roots = [search_root]
    roots.extend(path.parent for path in search_root.rglob("train_test_split.txt"))
    for root in roots:
        if all((root / filename).is_file() for filename in REQUIRED_CUB_FILES):
            if (root / "images").is_dir():
                candidates.append(root)
    candidates = sorted(set(candidates))
    if not candidates:
        raise FileNotFoundError(
            "Could not find the official CUB files below "
            f"{search_root}. Add a Kaggle CUB-200-2011 dataset containing "
            "images/, images.txt, image_class_labels.txt, and train_test_split.txt."
        )
    if len(candidates) > 1:
        rendered = "\n".join(f"- {path}" for path in candidates)
        raise RuntimeError(
            "Multiple CUB roots were found. Pass the intended root explicitly:\n"
            f"{rendered}"
        )
    return candidates[0]


def _read_two_column_file(path: Path) -> dict[int, str]:
    values: dict[int, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            parts = stripped.split(maxsplit=1)
            if len(parts) != 2:
                raise ValueError(f"Malformed row at {path}:{line_number}")
            key = int(parts[0])
            if key in values:
                raise ValueError(f"Duplicate image ID {key} in {path}")
            values[key] = parts[1]
    return values


def load_cub_records(cub_root: Path) -> list[CubRecord]:
    """Load and cross-check the official image, label, and split metadata."""

    cub_root = cub_root.expanduser().resolve()
    missing = [name for name in REQUIRED_CUB_FILES if not (cub_root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing CUB metadata files: {missing}")

    images = _read_two_column_file(cub_root / "images.txt")
    labels_raw = _read_two_column_file(cub_root / "image_class_labels.txt")
    split_raw = _read_two_column_file(cub_root / "train_test_split.txt")
    classes_path = cub_root / "classes.txt"
    classes = _read_two_column_file(classes_path) if classes_path.is_file() else {}

    if images.keys() != labels_raw.keys() or images.keys() != split_raw.keys():
        raise ValueError("CUB metadata files do not contain the same image IDs")

    records: list[CubRecord] = []
    for image_id in sorted(images):
        label = int(labels_raw[image_id])
        relative_path = images[image_id]
        image_path = cub_root / "images" / relative_path
        if not image_path.is_file():
            raise FileNotFoundError(f"CUB image is missing: {image_path}")
        records.append(
            CubRecord(
                image_id=image_id,
                relative_path=relative_path,
                label=label,
                class_name=classes.get(label, relative_path.split("/", 1)[0]),
                official_split="train" if int(split_raw[image_id]) == 1 else "test",
            )
        )
    return records


def make_balanced_pilot_split(
    records: list[CubRecord],
    *,
    num_classes: int,
    train_per_class: int,
    val_per_class: int,
    seed: int,
) -> list[PilotRecord]:
    """Sample a balanced pilot from official training images only."""

    if num_classes <= 1 or train_per_class <= 0 or val_per_class <= 0:
        raise ValueError("pilot sizes must be positive and include at least two classes")

    by_class: dict[int, list[CubRecord]] = {}
    for record in records:
        if record.official_split == "train":
            by_class.setdefault(record.label, []).append(record)
    eligible = sorted(
        label
        for label, examples in by_class.items()
        if len(examples) >= train_per_class + val_per_class
    )
    if len(eligible) < num_classes:
        raise ValueError(
            f"Only {len(eligible)} classes have enough official training images; "
            f"{num_classes} requested"
        )

    rng = random.Random(seed)
    chosen_labels = sorted(rng.sample(eligible, num_classes))
    pilot: list[PilotRecord] = []
    for label in chosen_labels:
        examples = sorted(by_class[label], key=lambda record: record.image_id)
        rng.shuffle(examples)
        for record in examples[:train_per_class]:
            pilot.append(
                PilotRecord(
                    image_id=record.image_id,
                    relative_path=record.relative_path,
                    label=record.label,
                    class_name=record.class_name,
                    split="train",
                )
            )
        start = train_per_class
        for record in examples[start : start + val_per_class]:
            pilot.append(
                PilotRecord(
                    image_id=record.image_id,
                    relative_path=record.relative_path,
                    label=record.label,
                    class_name=record.class_name,
                    split="val",
                )
            )
    return sorted(pilot, key=lambda record: (record.split, record.label, record.image_id))


def save_pilot_manifest(
    records: list[PilotRecord], destination: Path, metadata: dict[str, object]
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["image_id", "relative_path", "label", "class_name", "split"]
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)
    sidecar = destination.with_suffix(".json")
    sidecar.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def load_pilot_manifest(path: Path) -> list[PilotRecord]:
    records: list[PilotRecord] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            records.append(
                PilotRecord(
                    image_id=int(row["image_id"]),
                    relative_path=row["relative_path"],
                    label=int(row["label"]),
                    class_name=row["class_name"],
                    split=row["split"],
                )
            )
    if not records:
        raise ValueError(f"Pilot manifest is empty: {path}")
    return records
