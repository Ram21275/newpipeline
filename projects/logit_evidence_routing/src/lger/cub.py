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


@dataclass(frozen=True)
class CubBoundingBox:
    """One CUB bounding box in original-image pixel coordinates."""

    x: float
    y: float
    width: float
    height: float


@dataclass(frozen=True)
class CubPartLocation:
    """One CUB part annotation in original-image pixel coordinates."""

    part_id: int
    x: float
    y: float
    visible: bool


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


def load_cub_bounding_boxes(cub_root: Path) -> dict[int, CubBoundingBox]:
    """Load CUB's ``image_id x y width height`` bounding-box annotations."""

    path = cub_root.expanduser().resolve() / "bounding_boxes.txt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing CUB bounding boxes: {path}")
    boxes: dict[int, CubBoundingBox] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            parts = raw_line.split()
            if not parts:
                continue
            if len(parts) != 5:
                raise ValueError(f"Malformed bounding box at {path}:{line_number}")
            image_id = int(parts[0])
            if image_id in boxes:
                raise ValueError(f"Duplicate image ID {image_id} in {path}")
            box = CubBoundingBox(*(float(value) for value in parts[1:]))
            if box.width <= 0 or box.height <= 0:
                raise ValueError(f"Non-positive bounding box at {path}:{line_number}")
            boxes[image_id] = box
    if not boxes:
        raise ValueError(f"CUB bounding-box file is empty: {path}")
    return boxes


def load_cub_part_locations(cub_root: Path) -> dict[int, list[CubPartLocation]]:
    """Load CUB's ``image_id part_id x y visible`` annotations."""

    path = cub_root.expanduser().resolve() / "parts" / "part_locs.txt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing CUB part locations: {path}")
    locations: dict[int, list[CubPartLocation]] = {}
    seen: set[tuple[int, int]] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            parts = raw_line.split()
            if not parts:
                continue
            if len(parts) != 5:
                raise ValueError(f"Malformed part location at {path}:{line_number}")
            image_id, part_id = int(parts[0]), int(parts[1])
            key = (image_id, part_id)
            if key in seen:
                raise ValueError(f"Duplicate image/part pair {key} in {path}")
            seen.add(key)
            locations.setdefault(image_id, []).append(
                CubPartLocation(
                    part_id=part_id,
                    x=float(parts[2]),
                    y=float(parts[3]),
                    visible=bool(int(parts[4])),
                )
            )
    if not locations:
        raise ValueError(f"CUB part-location file is empty: {path}")
    return locations


def map_bbox_to_center_crop(
    box: CubBoundingBox,
    *,
    original_size: tuple[int, int],
    output_size: tuple[int, int],
) -> tuple[float, float, float, float]:
    """Map a box through Transformers' shortest-edge resize and center crop."""

    original_width, original_height = original_size
    output_width, output_height = output_size
    if min(original_width, original_height, output_width, output_height) <= 0:
        raise ValueError("image dimensions must be positive")
    if output_width != output_height:
        raise ValueError("classic LLaVA center-crop output must be square")
    if original_width <= original_height:
        resized_width = output_width
        resized_height = int(output_width * original_height / original_width)
    else:
        resized_height = output_height
        resized_width = int(output_height * original_width / original_height)
    scale_x = resized_width / original_width
    scale_y = resized_height / original_height
    crop_left = (resized_width - output_width) // 2
    crop_top = (resized_height - output_height) // 2
    x1 = box.x * scale_x - crop_left
    y1 = box.y * scale_y - crop_top
    x2 = (box.x + box.width) * scale_x - crop_left
    y2 = (box.y + box.height) * scale_y - crop_top
    clipped = (
        min(max(x1, 0.0), float(output_width)),
        min(max(y1, 0.0), float(output_height)),
        min(max(x2, 0.0), float(output_width)),
        min(max(y2, 0.0), float(output_height)),
    )
    if clipped[0] >= clipped[2] or clipped[1] >= clipped[3]:
        raise ValueError("bounding box falls outside the model's center crop")
    return clipped


def map_point_to_center_crop(
    point: tuple[float, float],
    *,
    original_size: tuple[int, int],
    output_size: tuple[int, int],
) -> tuple[float, float] | None:
    """Map a point through the same shortest-edge resize and center crop."""

    original_width, original_height = original_size
    output_width, output_height = output_size
    if min(original_width, original_height, output_width, output_height) <= 0:
        raise ValueError("image dimensions must be positive")
    if output_width != output_height:
        raise ValueError("classic LLaVA center-crop output must be square")
    if original_width <= original_height:
        resized_width = output_width
        resized_height = int(output_width * original_height / original_width)
    else:
        resized_height = output_height
        resized_width = int(output_height * original_width / original_height)
    x = point[0] * resized_width / original_width - (resized_width - output_width) // 2
    y = point[1] * resized_height / original_height - (resized_height - output_height) // 2
    if not (0 <= x < output_width and 0 <= y < output_height):
        return None
    return x, y


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
