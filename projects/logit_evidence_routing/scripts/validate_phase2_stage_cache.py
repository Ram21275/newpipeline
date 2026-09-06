#!/usr/bin/env python3
"""Hash-check and fully reload a completed Phase 2 development cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lger.stage_cache import (  # noqa: E402
    SCHEMA_VERSION,
    atomic_json_write,
    config_digest,
    validate_safetensors_shard,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    run_config_path = args.cache_dir / "run_config.json"
    index_path = args.cache_dir / "index.json"
    summary_path = args.cache_dir / "extraction_summary.json"
    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    index = json.loads(index_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    digest = config_digest(run_config)
    if (
        int(run_config.get("schema_version", 0)) != SCHEMA_VERSION
        or run_config.get("purpose")
        != "full_240_image_development_pilot_stage_cache"
        or run_config.get("storage_format") != "safetensors_sharded_v1"
        or int(run_config.get("official_test_images", -1)) != 0
    ):
        raise RuntimeError("run configuration is not a Phase 2 development cache")
    if (
        index.get("config_digest") != digest
        or index.get("complete") is not True
        or summary.get("config_digest") != digest
        or summary.get("status") != "PASS"
    ):
        raise RuntimeError("cache config, index, or summary is incomplete/inconsistent")
    records = index.get("records")
    shards = index.get("shards")
    if not isinstance(records, list) or not isinstance(shards, list):
        raise RuntimeError("cache index lacks records or shards")
    expected_ids = [int(value) for value in run_config.get("manifest_image_ids", [])]
    indexed_ids = [int(item["image_id"]) for item in records]
    if indexed_ids != expected_ids or len(indexed_ids) != 240:
        raise RuntimeError("cache image IDs do not exactly match the frozen manifest")
    split_counts = {
        split: sum(item.get("development_split") == split for item in records)
        for split in ("train", "val")
    }
    if split_counts != {"train": 160, "val": 80}:
        raise RuntimeError(f"cache development splits differ: {split_counts}")

    validation_rows: list[dict[str, object]] = []
    validated_ids: list[int] = []
    total_bytes = 0
    for expected_shard_index, shard in enumerate(shards):
        if int(shard.get("shard_index", -1)) != expected_shard_index:
            raise RuntimeError("cache shard indices are not contiguous")
        tensor_path = args.cache_dir / str(shard["tensor_path"])
        metadata_path = args.cache_dir / str(shard["metadata_path"])
        if (
            _sha256(tensor_path) != shard.get("tensor_sha256")
            or _sha256(metadata_path) != shard.get("metadata_sha256")
        ):
            raise RuntimeError(f"cache shard hash differs: {tensor_path}")
        validations = validate_safetensors_shard(
            tensor_path,
            metadata_path,
            expected_config_digest=digest,
        )
        shard_ids = [int(value) for value in shard.get("image_ids", [])]
        if len(validations) != len(shard_ids):
            raise RuntimeError("shard validation count differs from its image index")
        validated_ids.extend(shard_ids)
        total_bytes += tensor_path.stat().st_size + metadata_path.stat().st_size
        validation_rows.extend(validations)
        print(
            f"Validated shard {expected_shard_index + 1}/{len(shards)} "
            f"({len(shard_ids)} images)"
        )
    if validated_ids != expected_ids:
        raise RuntimeError("validated shard order differs from the frozen manifest")
    if any(int(row.get("stage_count", 0)) != 9 for row in validation_rows):
        raise RuntimeError("one or more records do not contain all nine stages")

    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "config_digest": digest,
        "images": len(validated_ids),
        "development_split_counts": split_counts,
        "official_test_images": 0,
        "shards": len(shards),
        "reload_validated_records": len(validation_rows),
        "stage_count_per_record": 9,
        "cache_bytes": total_bytes,
        "vision_hidden_sizes": sorted(
            {int(row["vision_hidden_size"]) for row in validation_rows}
        ),
        "language_hidden_sizes": sorted(
            {int(row["language_hidden_size"]) for row in validation_rows}
        ),
        "patch_counts": sorted({int(row["patch_count"]) for row in validation_rows}),
        "official_test_split_untouched": True,
    }
    output = args.output or args.cache_dir / "validation_report.json"
    atomic_json_write(report, output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
