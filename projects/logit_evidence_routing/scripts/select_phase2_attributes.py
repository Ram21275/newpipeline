#!/usr/bin/env python3
"""Freeze a CUB attribute subset using development-training labels only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lger.cub import (  # noqa: E402
    discover_cub_root,
    load_cub_attributes,
    load_cub_certainties,
    load_cub_image_attribute_labels,
    load_cub_records,
    load_pilot_manifest,
    select_training_attribute_subset,
)
from lger.reproducibility import current_git_commit  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--search-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--cub-root", type=Path)
    args = parser.parse_args()

    selection_config = json.loads(args.config.read_text(encoding="utf-8"))
    if int(selection_config.get("schema_version", 0)) != 1:
        raise RuntimeError("unsupported Phase 2 attribute-selection config")
    if selection_config.get("selection_split") != "development_train_only":
        raise RuntimeError("attribute selection must use development training only")

    manifest = load_pilot_manifest(args.manifest)
    train_ids = {record.image_id for record in manifest if record.split == "train"}
    validation_ids = {record.image_id for record in manifest if record.split == "val"}
    if not train_ids or not train_ids.isdisjoint(validation_ids):
        raise RuntimeError("pilot train/validation identities are invalid")
    cub_root = args.cub_root or discover_cub_root(args.search_root)
    official = {record.image_id: record for record in load_cub_records(cub_root)}
    nontraining = sorted(
        image_id
        for image_id in train_ids
        if image_id not in official or official[image_id].official_split != "train"
    )
    if nontraining:
        raise RuntimeError(
            f"development-training set contains official test/unknown images: {nontraining[:10]}"
        )

    attributes = load_cub_attributes(cub_root)
    certainties = load_cub_certainties(cub_root)
    configured_groups = set(selection_config["groups"])
    available_groups = {attribute.group for attribute in attributes}
    missing_groups = sorted(configured_groups - available_groups)
    if missing_groups:
        raise RuntimeError(
            f"configured attribute groups are absent from CUB: {missing_groups}"
        )
    labels = load_cub_image_attribute_labels(cub_root, image_ids=train_ids)
    selected = select_training_attribute_subset(
        attributes,
        certainties,
        labels,
        train_image_ids=train_ids,
        groups=configured_groups,
        allowed_certainty_names=set(selection_config["allowed_certainty_names"]),
        min_positive=int(selection_config["min_positive_train"]),
        min_negative=int(selection_config["min_negative_train"]),
        max_missing_fraction=float(selection_config["max_missing_fraction"]),
    )
    if not selected:
        raise RuntimeError(
            "No attributes passed the predeclared development-training thresholds"
        )
    payload = {
        "schema_version": 1,
        "git_commit": current_git_commit(REPO_ROOT),
        "manifest": str(args.manifest.expanduser().resolve()),
        "cub_root": str(cub_root.expanduser().resolve()),
        "selection_split": "development_train_only",
        "train_images_used": len(train_ids),
        "validation_images_inspected": 0,
        "config": selection_config,
        "attribute_vocabulary_size": len(attributes),
        "selected_attribute_count": len(selected),
        "selected_attributes": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(args.output)
    print(
        f"Selected {len(selected)} attributes from {len(configured_groups)} "
        f"predeclared groups using {len(train_ids)} development-training images"
    )
    print(f"Attribute subset written to {args.output}")


if __name__ == "__main__":
    main()
