"""Metadata-only preparation and fail-closed image access for the bridge."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


def table(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def official_metadata(cub_root: Path) -> tuple[dict[int, str], dict[int, int]]:
    def read(name: str) -> dict[int, str]:
        result = {}
        for line in (cub_root / name).read_text().splitlines():
            key, value = line.split(maxsplit=1)
            if int(key) in result:
                raise ValueError(f"duplicate CUB metadata ID in {name}")
            result[int(key)] = value
        return result
    return read("images.txt"), {k: int(v) for k, v in read("train_test_split.txt").items()}


def validate_decisions(rows: list[dict], development: list[dict], cub_root: Path,
                       allowed_attributes: set[int]) -> list[dict]:
    """Reads metadata only. Call this before image hashes, dimensions, or pixels."""
    paths, splits = official_metadata(cub_root)
    dev = {int(row["image_id"]): row for row in development}
    if len(dev) != len(development):
        raise ValueError("duplicate development IDs")
    if not rows or len({str(r["decision_id"]) for r in rows}) != len(rows):
        raise ValueError("empty or duplicate decision IDs")
    seen = set()
    for row in rows:
        image_id, attribute_id = int(row["image_id"]), int(row["attribute_id"])
        if (image_id, attribute_id) in seen:
            raise ValueError("duplicate image-attribute decision")
        seen.add((image_id, attribute_id))
        if splits.get(image_id) != 1 or image_id not in dev:
            raise ValueError("official-test or non-development image rejected before pixel access")
        if dev[image_id]["split"] != "val":
            raise ValueError("bridge evaluation must use the fixed development val subset")
        if row["relative_path"] != paths[image_id] or dev[image_id]["relative_path"] != paths[image_id]:
            raise ValueError("path differs from official/development metadata")
        relative = Path(paths[image_id])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("unsafe image path")
        root = (cub_root / "images").resolve()
        if root not in (root / relative).resolve().parents:
            raise ValueError("image path escapes the CUB image directory")
        if attribute_id not in allowed_attributes or int(row["target"]) not in (0, 1):
            raise ValueError("attribute or target outside the locked policy")
    return rows


def select_pilot(rows: list[dict], allowed_attributes: set[int]) -> list[dict]:
    """One decision per attribute/target, picked by ID hash, never model outcome.

    Missing strata are recorded by the caller, not replaced with new attributes.
    This is a falsification pilot, not a powered significance study.
    """
    selected = []
    for attribute in sorted(allowed_attributes):
        for target in (0, 1):
            eligible = [r for r in rows if int(r["attribute_id"]) == attribute and int(r["target"]) == target]
            if eligible:
                selected.append(min(eligible, key=lambda r: hashlib.sha256(
                    f"1609:{r['image_id']}:{attribute}".encode()).hexdigest()))
    return selected
