#!/usr/bin/env python3
"""Add deterministic opposite-label image donors to a binary decision manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"decision manifest is empty: {path}")
    return rows


def rank(seed: int, decision_id: str, donor_id: str) -> str:
    return hashlib.sha256(f"{seed}\0{decision_id}\0{donor_id}".encode()).hexdigest()


def add_opposite_label_donors(
    rows: list[dict[str, Any]], *, seed: int
) -> list[dict[str, Any]]:
    groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    seen: set[str] = set()
    for source in rows:
        row = dict(source)
        decision_id = str(row.get("decision_id", "")).strip()
        if not decision_id or decision_id in seen:
            raise RuntimeError("decision IDs must be unique and nonempty")
        seen.add(decision_id)
        attribute_id, target = int(row["attribute_id"]), int(row["target"])
        if target not in (0, 1):
            raise RuntimeError("target must be binary")
        groups[(attribute_id, target)].append(row)

    by_attribute: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for values in groups.values():
        for value in values:
            by_attribute[int(value["attribute_id"])].append(value)
    output = []
    for source in rows:
        row = dict(source)
        attribute_id, target = int(row["attribute_id"]), int(row["target"])
        shuffle_fields = ("shuffled_image_id", "shuffled_relative_path")
        present = [field in row and row[field] not in (None, "") for field in shuffle_fields]
        if any(present) and not all(present):
            raise RuntimeError("shuffled image identity is partially present")
        if not any(present):
            shuffle_candidates = [candidate for candidate in by_attribute[attribute_id]
                                  if str(candidate["image_id"]) != str(row["image_id"])]
            if not shuffle_candidates:
                raise RuntimeError(f"attribute {attribute_id} cannot be image-shuffled")
            shuffled = min(shuffle_candidates, key=lambda candidate: rank(
                seed + 1, str(row["decision_id"]), str(candidate["decision_id"])
            ))
            row["shuffled_image_id"] = shuffled["image_id"]
            row["shuffled_relative_path"] = shuffled["relative_path"]
        candidates = groups.get((attribute_id, 1 - target), [])
        if not candidates:
            raise RuntimeError(f"attribute {attribute_id} lacks an opposite-label donor")
        same_class = [candidate for candidate in candidates
                      if candidate.get("class_id") == row.get("class_id")]
        pool = same_class or candidates
        donor = min(pool, key=lambda candidate: rank(
            seed, str(row["decision_id"]), str(candidate["decision_id"])
        ))
        if str(donor["image_id"]) == str(row["image_id"]):
            raise RuntimeError("opposite-label donor reused the evaluated image")
        row.update({
            "opposite_label_image_id": donor["image_id"],
            "opposite_label_relative_path": donor["relative_path"],
            "opposite_label_target": int(donor["target"]),
            "opposite_label_same_class": int(bool(same_class)),
            "opposite_label_seed": seed,
        })
        output.append(row)
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=31415)
    args = parser.parse_args()
    rows = add_opposite_label_donors(read_csv(args.input), seed=args.seed)
    write_csv(args.output, rows)
    print(json.dumps({
        "status": "PASS",
        "output": str(args.output.resolve()),
        "decisions": len(rows),
        "same_class_donors": sum(int(row["opposite_label_same_class"]) for row in rows),
        "official_test_images_used": 0,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
