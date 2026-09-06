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


@dataclass(frozen=True)
class CubAttribute:
    """One entry from CUB's explicit image-attribute vocabulary."""

    attribute_id: int
    name: str

    @property
    def group(self) -> str:
        return self.name.split("::", 1)[0]


@dataclass(frozen=True)
class CubCertainty:
    """One MTurk certainty level defined by the CUB distribution."""

    certainty_id: int
    name: str


@dataclass(frozen=True)
class CubImageAttributeLabel:
    """Raw per-image CUB attribute label with its annotation certainty."""

    image_id: int
    attribute_id: int
    is_present: bool
    certainty_id: int
    worker_time_seconds: float


@dataclass(frozen=True)
class CenterCropTransform:
    """Exact shortest-edge resize and center-crop geometry for classic LLaVA."""

    original_size: tuple[int, int]
    resized_size: tuple[int, int]
    output_size: tuple[int, int]
    scale_xy: tuple[float, float]
    crop_left: int
    crop_top: int


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


def load_cub_attributes(cub_root: Path) -> list[CubAttribute]:
    """Load the official CUB attribute vocabulary without inferring new labels."""

    path = cub_root.expanduser().resolve() / "attributes" / "attributes.txt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing CUB attribute vocabulary: {path}")
    values = _read_two_column_file(path)
    attributes = [CubAttribute(key, values[key]) for key in sorted(values)]
    if not attributes:
        raise ValueError(f"CUB attribute vocabulary is empty: {path}")
    return attributes


def load_cub_certainties(cub_root: Path) -> list[CubCertainty]:
    """Load CUB's named certainty scale used by image attribute labels."""

    path = cub_root.expanduser().resolve() / "attributes" / "certainties.txt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing CUB certainty vocabulary: {path}")
    values = _read_two_column_file(path)
    certainties = [CubCertainty(key, values[key]) for key in sorted(values)]
    if not certainties:
        raise ValueError(f"CUB certainty vocabulary is empty: {path}")
    return certainties


def load_cub_image_attribute_labels(
    cub_root: Path,
    *,
    image_ids: set[int] | None = None,
) -> dict[int, list[CubImageAttributeLabel]]:
    """Load raw image attributes, optionally retaining only requested images.

    The loader preserves presence and certainty separately. It deliberately does
    not turn ``not visible`` or ``guess`` rows into training targets; that policy
    is applied explicitly by :func:`materialize_attribute_targets`.
    """

    cub_root = cub_root.expanduser().resolve()
    path = cub_root / "attributes" / "image_attribute_labels.txt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing CUB image attribute labels: {path}")
    attribute_ids = {item.attribute_id for item in load_cub_attributes(cub_root)}
    certainty_ids = {item.certainty_id for item in load_cub_certainties(cub_root)}
    requested = set(image_ids) if image_ids is not None else None
    labels: dict[int, list[CubImageAttributeLabel]] = {}
    seen: set[tuple[int, int]] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            parts = raw_line.split()
            if not parts:
                continue
            if len(parts) != 5:
                raise ValueError(f"Malformed attribute label at {path}:{line_number}")
            image_id, attribute_id = int(parts[0]), int(parts[1])
            if requested is not None and image_id not in requested:
                continue
            key = (image_id, attribute_id)
            if key in seen:
                raise ValueError(f"Duplicate image/attribute pair {key} in {path}")
            seen.add(key)
            is_present = int(parts[2])
            certainty_id = int(parts[3])
            if attribute_id not in attribute_ids:
                raise ValueError(
                    f"Unknown attribute ID {attribute_id} at {path}:{line_number}"
                )
            if is_present not in (0, 1):
                raise ValueError(
                    f"Attribute presence must be 0/1 at {path}:{line_number}"
                )
            if certainty_id not in certainty_ids:
                raise ValueError(
                    f"Unknown certainty ID {certainty_id} at {path}:{line_number}"
                )
            worker_time = float(parts[4])
            if worker_time < 0:
                raise ValueError(f"Negative annotation time at {path}:{line_number}")
            labels.setdefault(image_id, []).append(
                CubImageAttributeLabel(
                    image_id=image_id,
                    attribute_id=attribute_id,
                    is_present=bool(is_present),
                    certainty_id=certainty_id,
                    worker_time_seconds=worker_time,
                )
            )
    if requested is not None:
        for image_id in requested:
            labels.setdefault(image_id, [])
    if not labels:
        raise ValueError(f"No requested CUB attribute labels found in {path}")
    for rows in labels.values():
        rows.sort(key=lambda item: item.attribute_id)
    return labels


def materialize_attribute_targets(
    attributes: list[CubAttribute],
    certainties: list[CubCertainty],
    labels: list[CubImageAttributeLabel],
) -> list[dict[str, object]]:
    """Create explicit raw and primary-target states for one image.

    The primary policy treats ``probably`` and ``definitely`` as observed binary
    targets, while ``guess`` and ``not visible`` remain stored but masked. A
    genuinely absent annotation is represented as ``missing``. This keeps label
    uncertainty separate from feature extraction and permits later sensitivity
    analyses without changing the cache.
    """

    certainty_by_id = {item.certainty_id: item.name for item in certainties}
    label_by_attribute = {item.attribute_id: item for item in labels}
    if len(label_by_attribute) != len(labels):
        raise ValueError("duplicate attribute labels cannot be materialized")
    output: list[dict[str, object]] = []
    for attribute in sorted(attributes, key=lambda item: item.attribute_id):
        label = label_by_attribute.get(attribute.attribute_id)
        if label is None:
            output.append(
                {
                    "attribute_id": attribute.attribute_id,
                    "name": attribute.name,
                    "group": attribute.group,
                    "is_present": None,
                    "certainty_id": None,
                    "certainty_name": None,
                    "worker_time_seconds": None,
                    "state": "missing",
                    "primary_target": None,
                }
            )
            continue
        certainty_name = certainty_by_id[label.certainty_id]
        normalized_certainty = certainty_name.strip().lower()
        if normalized_certainty == "not visible":
            state = "not_visible"
            primary_target: bool | None = None
        elif normalized_certainty == "guess":
            state = "uncertain"
            primary_target = None
        else:
            state = "present" if label.is_present else "absent"
            primary_target = label.is_present
        output.append(
            {
                "attribute_id": attribute.attribute_id,
                "name": attribute.name,
                "group": attribute.group,
                "is_present": label.is_present,
                "certainty_id": label.certainty_id,
                "certainty_name": certainty_name,
                "worker_time_seconds": label.worker_time_seconds,
                "state": state,
                "primary_target": primary_target,
            }
        )
    return output


def select_training_attribute_subset(
    attributes: list[CubAttribute],
    certainties: list[CubCertainty],
    labels_by_image: dict[int, list[CubImageAttributeLabel]],
    *,
    train_image_ids: set[int],
    groups: set[str],
    allowed_certainty_names: set[str],
    min_positive: int,
    min_negative: int,
    max_missing_fraction: float,
) -> list[dict[str, object]]:
    """Select probe attributes using development-training annotations only."""

    if not train_image_ids:
        raise ValueError("attribute selection requires development-training images")
    if not groups:
        raise ValueError("at least one attribute group is required")
    if min(min_positive, min_negative) < 0:
        raise ValueError("minimum positive/negative counts cannot be negative")
    if not 0 <= max_missing_fraction < 1:
        raise ValueError("max_missing_fraction must be in [0, 1)")
    certainty_by_id = {
        item.certainty_id: item.name.strip().lower() for item in certainties
    }
    allowed = {name.strip().lower() for name in allowed_certainty_names}
    labels_by_key = {
        (row.image_id, row.attribute_id): row
        for image_id, rows in labels_by_image.items()
        if image_id in train_image_ids
        for row in rows
    }
    selected: list[dict[str, object]] = []
    total = len(train_image_ids)
    for attribute in attributes:
        if attribute.group not in groups:
            continue
        positive = 0
        negative = 0
        for image_id in train_image_ids:
            label = labels_by_key.get((image_id, attribute.attribute_id))
            if label is None or certainty_by_id[label.certainty_id] not in allowed:
                continue
            if label.is_present:
                positive += 1
            else:
                negative += 1
        observed = positive + negative
        missing_fraction = 1.0 - observed / total
        if (
            positive >= min_positive
            and negative >= min_negative
            and missing_fraction <= max_missing_fraction
        ):
            selected.append(
                {
                    "attribute_id": attribute.attribute_id,
                    "name": attribute.name,
                    "group": attribute.group,
                    "positive_train": positive,
                    "negative_train": negative,
                    "observed_train": observed,
                    "missing_fraction_train": missing_fraction,
                }
            )
    return selected


def center_crop_transform(
    *,
    original_size: tuple[int, int],
    output_size: tuple[int, int],
) -> CenterCropTransform:
    """Resolve the geometry shared by box, part, and patch-coordinate mapping."""

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
    return CenterCropTransform(
        original_size=original_size,
        resized_size=(resized_width, resized_height),
        output_size=output_size,
        scale_xy=(resized_width / original_width, resized_height / original_height),
        crop_left=(resized_width - output_width) // 2,
        crop_top=(resized_height - output_height) // 2,
    )


def map_bbox_to_center_crop(
    box: CubBoundingBox,
    *,
    original_size: tuple[int, int],
    output_size: tuple[int, int],
) -> tuple[float, float, float, float]:
    """Map a box through Transformers' shortest-edge resize and center crop."""

    output_width, output_height = output_size
    transform = center_crop_transform(
        original_size=original_size, output_size=output_size
    )
    scale_x, scale_y = transform.scale_xy
    x1 = box.x * scale_x - transform.crop_left
    y1 = box.y * scale_y - transform.crop_top
    x2 = (box.x + box.width) * scale_x - transform.crop_left
    y2 = (box.y + box.height) * scale_y - transform.crop_top
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

    output_width, output_height = output_size
    transform = center_crop_transform(
        original_size=original_size, output_size=output_size
    )
    scale_x, scale_y = transform.scale_xy
    x = point[0] * scale_x - transform.crop_left
    y = point[1] * scale_y - transform.crop_top
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
