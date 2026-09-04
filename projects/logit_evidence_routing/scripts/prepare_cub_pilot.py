#!/usr/bin/env python3
"""Create a deterministic class-balanced Phase 01 split from official CUB data."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lger.cub import (  # noqa: E402
    discover_cub_root,
    load_cub_records,
    make_balanced_pilot_split,
    save_pilot_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--cub-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-classes", type=int, default=20)
    parser.add_argument("--train-per-class", type=int, default=8)
    parser.add_argument("--val-per-class", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cub_root = args.cub_root or discover_cub_root(args.search_root)
    all_records = load_cub_records(cub_root)
    pilot = make_balanced_pilot_split(
        all_records,
        num_classes=args.num_classes,
        train_per_class=args.train_per_class,
        val_per_class=args.val_per_class,
        seed=args.seed,
    )
    metadata = {
        "cub_root": str(cub_root.resolve()),
        "official_test_images_used": 0,
        "num_classes": args.num_classes,
        "train_per_class": args.train_per_class,
        "val_per_class": args.val_per_class,
        "seed": args.seed,
        "train_images": sum(record.split == "train" for record in pilot),
        "val_images": sum(record.split == "val" for record in pilot),
    }
    save_pilot_manifest(pilot, args.output, metadata)
    print(f"CUB root: {cub_root}")
    print(
        f"Pilot: {metadata['num_classes']} classes, "
        f"{metadata['train_images']} train, {metadata['val_images']} validation"
    )
    print(f"Manifest: {args.output}")


if __name__ == "__main__":
    main()
