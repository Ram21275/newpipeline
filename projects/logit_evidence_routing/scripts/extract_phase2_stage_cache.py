#!/usr/bin/env python3
"""Extract the gated 240-image Phase 2 development cache into safe shards."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import platform
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lger.cub import (  # noqa: E402
    center_crop_transform,
    discover_cub_root,
    load_cub_attributes,
    load_cub_bounding_boxes,
    load_cub_certainties,
    load_cub_image_attribute_labels,
    load_cub_part_locations,
    load_cub_records,
    load_pilot_manifest,
    map_bbox_to_center_crop,
    map_point_to_center_crop,
    materialize_attribute_targets,
)
from lger.hf_stage_cache import HfLlavaStageExtractor, StageExtraction  # noqa: E402
from lger.reproducibility import current_git_commit  # noqa: E402
from lger.stage_cache import (  # noqa: E402
    SCHEMA_VERSION,
    atomic_json_write,
    config_digest,
    patch_centers,
    plan_pending_shards,
    stage_metadata,
    validate_safetensors_shard,
    validate_stage_record,
    write_or_validate_config,
    write_safetensors_shard,
)


DEFAULT_MODEL = "llava-hf/llava-1.5-7b-hf"
DEFAULT_REVISION = "b234b804b114d9e37bb655e11cbbb5f5e971b7a9"
EXPECTED_PILOT_IMAGES = 240
EXPECTED_TRAIN_IMAGES = 160
EXPECTED_VALIDATION_IMAGES = 80


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: Path) -> dict[str, object]:
    return {"bytes": path.stat().st_size, "sha256": _sha256(path)}


def _load_gate(path: Path, manifest: Path) -> dict[str, object]:
    gate = json.loads(path.read_text(encoding="utf-8"))
    if gate.get("passed") is not True or gate.get("status") not in {
        "PASS",
        "PASS WITH ANOMALY",
    }:
        raise RuntimeError("Phase 1 gate did not pass; stage extraction is blocked")
    configured_manifest = Path(str(gate.get("manifest", ""))).expanduser().resolve()
    if configured_manifest != manifest.expanduser().resolve():
        raise RuntimeError("Phase 1 gate and Phase 2 manifest paths differ")
    return gate


def _load_attribute_subset(
    path: Path, manifest_path: Path, train_image_count: int
) -> tuple[dict[str, object], list[int]]:
    subset = json.loads(path.read_text(encoding="utf-8"))
    if subset.get("selection_split") != "development_train_only":
        raise RuntimeError("attribute subset was not selected from development training")
    if int(subset.get("validation_images_inspected", -1)) != 0:
        raise RuntimeError("attribute subset inspected validation labels")
    if int(subset.get("train_images_used", -1)) != train_image_count:
        raise RuntimeError("attribute subset used a different development-training set")
    subset_manifest = Path(str(subset.get("manifest", ""))).expanduser().resolve()
    if subset_manifest != manifest_path.expanduser().resolve():
        raise RuntimeError("attribute subset and extraction use different manifests")
    policy = subset.get("config")
    allowed = (
        {
            str(value).strip().lower()
            for value in policy.get("allowed_certainty_names", [])
        }
        if isinstance(policy, dict)
        else set()
    )
    if allowed != {"probably", "definitely"}:
        raise RuntimeError("attribute subset uses an incompatible certainty policy")
    selected = subset.get("selected_attributes")
    if not isinstance(selected, list) or not selected:
        raise RuntimeError("attribute subset is empty or malformed")
    selected_ids = [int(item["attribute_id"]) for item in selected]
    if (
        len(set(selected_ids)) != len(selected_ids)
        or int(subset.get("selected_attribute_count", -1)) != len(selected_ids)
    ):
        raise RuntimeError("attribute subset count or identity is inconsistent")
    return subset, selected_ids


def _review_smoke(
    smoke_dir: Path,
    *,
    phase1_gate: Path,
    manifest: Path,
    attribute_subset: Path,
    model: str,
    revision: str,
    quantization: str,
    prompt: str,
    max_new_tokens: int,
    stage_plan: dict[str, int | None],
    gpu_total_bytes: int,
) -> dict[str, object]:
    smoke_config_path = smoke_dir / "run_config.json"
    smoke_summary_path = smoke_dir / "smoke_summary.json"
    smoke_config = json.loads(smoke_config_path.read_text(encoding="utf-8"))
    summary = json.loads(smoke_summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "PASS":
        raise RuntimeError("the one-image Phase 2 smoke did not pass")
    if summary.get("config_digest") != config_digest(smoke_config):
        raise RuntimeError("smoke summary and run configuration differ")
    reload_validation = summary.get("reload_validation")
    if (
        not isinstance(reload_validation, dict)
        or int(reload_validation.get("stage_count", 0)) != 9
        or int(reload_validation.get("patch_count", 0)) <= 0
        or int(reload_validation.get("tensor_bytes", 0)) <= 0
    ):
        raise RuntimeError("smoke reload validation is incomplete")
    expected_values = {
        "manifest": str(manifest.expanduser().resolve()),
        "model": model,
        "requested_revision": revision,
        "resolved_revision": revision,
        "quantization": quantization,
        "prompt": prompt,
        "stage_plan": stage_plan,
    }
    for key, expected in expected_values.items():
        if smoke_config.get(key) != expected:
            raise RuntimeError(f"production argument differs from smoke setting: {key}")
    generation = smoke_config.get("generation")
    if not isinstance(generation, dict) or int(
        generation.get("max_new_tokens", 0)
    ) != max_new_tokens:
        raise RuntimeError("production max-new-tokens differs from the smoke")
    gate_record = smoke_config.get("phase1_gate")
    subset_record = smoke_config.get("attribute_subset")
    if (
        not isinstance(gate_record, dict)
        or gate_record.get("sha256") != _sha256(phase1_gate)
        or not isinstance(subset_record, dict)
        or subset_record.get("sha256") != _sha256(attribute_subset)
    ):
        raise RuntimeError("smoke gate/subset identity differs from production inputs")

    projected_bytes = int(summary.get("projected_record_bytes", 0))
    peak_gpu_bytes = int(summary.get("peak_gpu_memory_bytes", 0))
    projected_seconds = float(summary.get("projected_extraction_seconds", 0.0))
    if min(projected_bytes, peak_gpu_bytes) <= 0 or projected_seconds <= 0:
        raise RuntimeError("smoke storage, memory, or runtime projection is missing")
    required_disk_bytes = math.ceil(projected_bytes * 1.15)
    if peak_gpu_bytes > int(gpu_total_bytes * 0.8):
        raise RuntimeError(
            f"smoke peak GPU allocation {peak_gpu_bytes} exceeds the 80% safety limit"
        )
    return {
        "smoke_config": str(smoke_config_path.resolve()),
        "smoke_config_sha256": _sha256(smoke_config_path),
        "smoke_summary": str(smoke_summary_path.resolve()),
        "smoke_summary_sha256": _sha256(smoke_summary_path),
        "projected_record_bytes": projected_bytes,
        "projected_extraction_seconds": projected_seconds,
        "peak_gpu_memory_bytes": peak_gpu_bytes,
        "required_disk_bytes_with_15_percent_margin": required_disk_bytes,
        "gpu_total_bytes": gpu_total_bytes,
        "decision": "PASS_FOR_240_IMAGE_DEVELOPMENT_PILOT",
    }


def _build_record(
    *,
    pilot: object,
    official: object,
    attributes: list[object],
    certainties: list[object],
    labels: list[object],
    box: object,
    parts: list[object],
    original_size: tuple[int, int],
    extraction: StageExtraction,
    prompt: str,
    selected_attribute_ids: list[int],
    expected_config_digest: str,
    extraction_seconds: float,
    peak_gpu_memory_bytes: int,
) -> dict[str, object]:
    processed_height, processed_width = extraction.processed_image_size
    crop = center_crop_transform(
        original_size=original_size,
        output_size=(processed_width, processed_height),
    )
    centers_model, centers_normalized = patch_centers(
        extraction.grid_size, extraction.processed_image_size
    )
    mapped_box = map_bbox_to_center_crop(
        box,
        original_size=original_size,
        output_size=(processed_width, processed_height),
    )
    part_rows: list[dict[str, object]] = []
    for part in parts:
        mapped = (
            map_point_to_center_crop(
                (part.x, part.y),
                original_size=original_size,
                output_size=(processed_width, processed_height),
            )
            if part.visible
            else None
        )
        part_rows.append(
            {
                "part_id": part.part_id,
                "visible": part.visible,
                "original_xy": [part.x, part.y],
                "model_xy": list(mapped) if mapped is not None else None,
            }
        )
    attribute_targets = materialize_attribute_targets(
        attributes, certainties, labels
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "complete": True,
        "config_digest": expected_config_digest,
        "image": {
            "image_id": pilot.image_id,
            "relative_path": pilot.relative_path,
            "class_id": pilot.label,
            "class_name": pilot.class_name,
            "official_split": official.official_split,
            "development_split": pilot.split,
            "attributes": attribute_targets,
            "selected_attribute_ids": selected_attribute_ids,
            "parts": part_rows,
            "bbox_original_xywh": [box.x, box.y, box.width, box.height],
            "bbox_model_xyxy": list(mapped_box),
        },
        "spatial": {
            "original_image_size_wh": list(original_size),
            "processed_image_size_hw": list(extraction.processed_image_size),
            "crop_transform": asdict(crop),
            "patch_count": centers_model.shape[0],
            "grid_size": list(extraction.grid_size),
            "patch_centers_model_xy": centers_model,
            "patch_centers_normalized_xy": centers_normalized,
            "visual_token_positions": extraction.visual_token_positions,
        },
        "stages": extraction.stages,
        "stage_metadata": stage_metadata(
            extraction.stages, extraction.stage_plan
        ),
        "generation": {
            "prompt": prompt,
            "rendered_prompt": extraction.rendered_prompt,
            "prompt_token_count": extraction.prompt_token_count,
            "answer_text": extraction.answer_text,
            "answer_token_ids": extraction.answer_token_ids,
            "answer_token_strings": list(extraction.answer_token_strings),
            "answer_token_logits": extraction.answer_token_logits,
        },
        "runtime": {
            "extraction_seconds": extraction_seconds,
            "peak_gpu_memory_bytes": peak_gpu_memory_bytes,
        },
    }


def _load_index(path: Path, digest: str) -> dict[str, object]:
    if not path.is_file():
        return {
            "schema_version": SCHEMA_VERSION,
            "storage_format": "safetensors_sharded_v1",
            "config_digest": digest,
            "complete": False,
            "shards": [],
            "records": [],
        }
    index = json.loads(path.read_text(encoding="utf-8"))
    if (
        int(index.get("schema_version", 0)) != SCHEMA_VERSION
        or index.get("storage_format") != "safetensors_sharded_v1"
        or index.get("config_digest") != digest
        or not isinstance(index.get("shards"), list)
        or not isinstance(index.get("records"), list)
    ):
        raise RuntimeError("existing production index is malformed or incompatible")
    record_ids = [int(item["image_id"]) for item in index["records"]]
    if len(record_ids) != len(set(record_ids)):
        raise RuntimeError("production index contains duplicate image IDs")
    return index


def _validate_index_files(index: dict[str, object], output_dir: Path) -> None:
    for shard in index["shards"]:
        tensor_path = output_dir / str(shard["tensor_path"])
        metadata_path = output_dir / str(shard["metadata_path"])
        if not tensor_path.is_file() or not metadata_path.is_file():
            raise RuntimeError(f"indexed shard files are missing: {tensor_path}")
        if (
            _sha256(tensor_path) != shard["tensor_sha256"]
            or _sha256(metadata_path) != shard["metadata_sha256"]
        ):
            raise RuntimeError(f"indexed shard identity changed: {tensor_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1-gate", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--attribute-subset", type=Path, required=True)
    parser.add_argument("--smoke-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--search-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--cub-root", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--quantization", choices=("4bit", "none"), default="4bit")
    parser.add_argument("--prompt", default="Describe the image briefly.")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--shard-size", type=int, default=20)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Enable a Kaggle GPU accelerator for Phase 2 extraction")
    if args.max_new_tokens <= 0:
        raise ValueError("max-new-tokens must be positive")
    if not 1 <= args.shard_size <= 40:
        raise ValueError("shard-size must be between 1 and 40 images")
    if len(args.revision) != 40 or any(
        character not in "0123456789abcdef" for character in args.revision.lower()
    ):
        raise ValueError("Phase 2 requires an immutable 40-character model revision")
    if args.smoke_dir.expanduser().resolve() == args.output_dir.expanduser().resolve():
        raise ValueError("production output must not overwrite the smoke directory")

    gate = _load_gate(args.phase1_gate, args.manifest)
    manifest = sorted(load_pilot_manifest(args.manifest), key=lambda item: item.image_id)
    if len(manifest) != EXPECTED_PILOT_IMAGES:
        raise RuntimeError(f"expected exactly {EXPECTED_PILOT_IMAGES} pilot images")
    if len({record.image_id for record in manifest}) != len(manifest):
        raise RuntimeError("pilot manifest contains duplicate image IDs")
    train_count = sum(record.split == "train" for record in manifest)
    validation_count = sum(record.split == "val" for record in manifest)
    if (train_count, validation_count) != (
        EXPECTED_TRAIN_IMAGES,
        EXPECTED_VALIDATION_IMAGES,
    ):
        raise RuntimeError("pilot must contain the frozen 160/80 development split")
    subset, selected_ids = _load_attribute_subset(
        args.attribute_subset, args.manifest, train_count
    )

    cub_root = args.cub_root or discover_cub_root(args.search_root)
    subset_cub_root = Path(str(subset.get("cub_root", ""))).expanduser().resolve()
    if subset_cub_root != cub_root.expanduser().resolve():
        raise RuntimeError("attribute subset and extraction use different CUB roots")
    official_by_id = {record.image_id: record for record in load_cub_records(cub_root)}
    for pilot in manifest:
        official = official_by_id.get(pilot.image_id)
        if official is None or official.official_split != "train":
            raise RuntimeError(
                f"pilot image {pilot.image_id} is absent from official CUB training"
            )
        if (
            official.relative_path,
            official.label,
            official.class_name,
        ) != (pilot.relative_path, pilot.label, pilot.class_name):
            raise RuntimeError(f"official metadata differs for image {pilot.image_id}")

    attributes = load_cub_attributes(cub_root)
    if not set(selected_ids).issubset(
        {attribute.attribute_id for attribute in attributes}
    ):
        raise RuntimeError("attribute subset contains IDs absent from CUB")
    certainties = load_cub_certainties(cub_root)
    pilot_ids = {record.image_id for record in manifest}
    print("Auditing attribute annotations for all 240 pilot images...")
    labels_by_image = load_cub_image_attribute_labels(cub_root, image_ids=pilot_ids)
    boxes = load_cub_bounding_boxes(cub_root)
    missing_boxes = sorted(pilot_ids - set(boxes))
    if missing_boxes:
        raise RuntimeError(f"pilot images lack bounding boxes: {missing_boxes[:10]}")
    parts_by_image = load_cub_part_locations(cub_root)

    identity_files = (
        "images.txt",
        "image_class_labels.txt",
        "train_test_split.txt",
        "bounding_boxes.txt",
        "parts/part_locs.txt",
        "attributes/attributes.txt",
        "attributes/certainties.txt",
        "attributes/image_attribute_labels.txt",
    )
    dataset_identity = {
        relative: _file_identity(cub_root / relative) for relative in identity_files
    }
    import transformers  # type: ignore[import-not-found]

    print(f"Loading frozen {args.model}@{args.revision} for the 240-image pilot...")
    model_load_started = time.perf_counter()
    extractor = HfLlavaStageExtractor.from_pretrained(
        args.model,
        revision=args.revision,
        quantization=args.quantization,
    )
    model_load_seconds = time.perf_counter() - model_load_started
    if extractor.resolved_revision != args.revision:
        raise RuntimeError("resolved model revision differs from the immutable revision")
    if any(parameter.requires_grad for parameter in extractor.model.parameters()):
        raise RuntimeError("one or more VLM parameters remain trainable")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    gpu_total = int(torch.cuda.get_device_properties(0).total_memory)
    smoke_review = _review_smoke(
        args.smoke_dir,
        phase1_gate=args.phase1_gate,
        manifest=args.manifest,
        attribute_subset=args.attribute_subset,
        model=args.model,
        revision=args.revision,
        quantization=args.quantization,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        stage_plan=extractor.stage_plan,
        gpu_total_bytes=gpu_total,
    )
    run_config = {
        "schema_version": SCHEMA_VERSION,
        "purpose": "full_240_image_development_pilot_stage_cache",
        "storage_format": "safetensors_sharded_v1",
        "shard_size": args.shard_size,
        "git_commit": current_git_commit(REPO_ROOT),
        "phase1_gate": {
            "path": str(args.phase1_gate.expanduser().resolve()),
            "sha256": _sha256(args.phase1_gate),
            "status": gate["status"],
        },
        "smoke_review": smoke_review,
        "manifest": str(args.manifest.expanduser().resolve()),
        "manifest_sha256": _sha256(args.manifest),
        "manifest_image_ids": [record.image_id for record in manifest],
        "development_split_counts": {"train": train_count, "val": validation_count},
        "official_test_images": 0,
        "attribute_subset": {
            "path": str(args.attribute_subset.expanduser().resolve()),
            "sha256": _sha256(args.attribute_subset),
            "selected_attribute_count": subset["selected_attribute_count"],
        },
        "dataset": {
            "name": "CUB-200-2011",
            "root": str(cub_root.expanduser().resolve()),
            "identity_files": dataset_identity,
        },
        "model": args.model,
        "requested_revision": args.revision,
        "resolved_revision": extractor.resolved_revision,
        "quantization": args.quantization,
        "prompt": args.prompt,
        "generation": {
            "do_sample": False,
            "max_new_tokens": args.max_new_tokens,
            "use_cache": True,
        },
        "stage_plan": extractor.stage_plan,
        "vision_feature_select_strategy": extractor.vision_feature_select_strategy,
        "spatial_preprocessing": extractor.spatial_preprocessing,
        "stored_dtype": "float16",
        "attribute_primary_policy": {
            "included_certainties": ["probably", "definitely"],
            "masked_states": ["guess", "not visible", "missing"],
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "torch_cuda": str(torch.version.cuda),
            "gpu": torch.cuda.get_device_name(0),
        },
    }
    digest = write_or_validate_config(
        args.output_dir / "run_config.json", run_config
    )
    index_path = args.output_dir / "index.json"
    index = _load_index(index_path, digest)
    _validate_index_files(index, args.output_dir)
    ordered_ids = [record.image_id for record in manifest]
    completed_ids = {int(item["image_id"]) for item in index["records"]}
    pending_shards = plan_pending_shards(
        ordered_ids, completed_ids, shard_size=args.shard_size
    )
    remaining_images = sum(len(image_ids) for image_ids in pending_shards)
    required_remaining_disk = math.ceil(
        int(smoke_review["required_disk_bytes_with_15_percent_margin"])
        * remaining_images
        / len(ordered_ids)
    )
    disk_free = shutil.disk_usage(args.output_dir).free
    if required_remaining_disk > disk_free:
        raise RuntimeError(
            f"insufficient free disk for {remaining_images} remaining images: "
            f"need {required_remaining_disk} bytes including margin, have {disk_free}"
        )
    if not pending_shards:
        if index.get("complete") is not True:
            raise RuntimeError("all records exist but the index is not marked complete")
        print("Phase 2 production cache already complete and hash-validated")
        return

    by_id = {record.image_id: record for record in manifest}
    total_shards = math.ceil(len(ordered_ids) / args.shard_size)
    all_started = time.perf_counter()
    for image_ids in pending_shards:
        shard_number = len(index["shards"])
        records: list[tuple[int, dict[str, object]]] = []
        for image_id in image_ids:
            pilot = by_id[image_id]
            official = official_by_id[image_id]
            image_path = cub_root / "images" / pilot.relative_path
            torch.cuda.reset_peak_memory_stats()
            extraction_started = time.perf_counter()
            with Image.open(image_path) as image_file:
                image = image_file.convert("RGB")
                original_size = image.size
                extraction = extractor.extract(
                    image,
                    args.prompt,
                    max_new_tokens=args.max_new_tokens,
                )
            extraction_seconds = time.perf_counter() - extraction_started
            peak_gpu_bytes = int(torch.cuda.max_memory_allocated())
            record = _build_record(
                pilot=pilot,
                official=official,
                attributes=attributes,
                certainties=certainties,
                labels=labels_by_image[image_id],
                box=boxes[image_id],
                parts=parts_by_image.get(image_id, []),
                original_size=original_size,
                extraction=extraction,
                prompt=args.prompt,
                selected_attribute_ids=selected_ids,
                expected_config_digest=digest,
                extraction_seconds=extraction_seconds,
                peak_gpu_memory_bytes=peak_gpu_bytes,
            )
            validation = validate_stage_record(
                record, expected_config_digest=digest
            )
            records.append((image_id, record))
            print(
                f"Extracted {len(completed_ids) + len(records)}/{len(ordered_ids)} "
                f"image={image_id} seconds={extraction_seconds:.2f} "
                f"tokens={validation['answer_tokens']}"
            )

        stem = f"stage-{shard_number:05d}-of-{total_shards:05d}"
        tensor_path = args.output_dir / "shards" / f"{stem}.safetensors"
        metadata_path = args.output_dir / "shards" / f"{stem}.json"
        write_safetensors_shard(
            records,
            tensor_path=tensor_path,
            metadata_path=metadata_path,
            expected_config_digest=digest,
        )
        validations = validate_safetensors_shard(
            tensor_path,
            metadata_path,
            expected_config_digest=digest,
        )
        shard_entry = {
            "shard_index": shard_number,
            "image_ids": image_ids,
            "tensor_path": str(tensor_path.relative_to(args.output_dir)),
            "metadata_path": str(metadata_path.relative_to(args.output_dir)),
            "tensor_bytes": tensor_path.stat().st_size,
            "metadata_bytes": metadata_path.stat().st_size,
            "tensor_sha256": _sha256(tensor_path),
            "metadata_sha256": _sha256(metadata_path),
            "reload_validated": True,
        }
        index["shards"].append(shard_entry)
        for record_index, ((image_id, record), validation) in enumerate(
            zip(records, validations)
        ):
            runtime = record["runtime"]
            index["records"].append(
                {
                    "image_id": image_id,
                    "development_split": record["image"]["development_split"],
                    "shard_index": shard_number,
                    "record_index": record_index,
                    "tensor_prefix": f"image/{image_id:05d}",
                    "extraction_seconds": runtime["extraction_seconds"],
                    "peak_gpu_memory_bytes": runtime["peak_gpu_memory_bytes"],
                    "tensor_bytes": validation["tensor_bytes"],
                    "complete": True,
                }
            )
        completed_ids.update(image_ids)
        index["complete"] = len(completed_ids) == len(ordered_ids)
        atomic_json_write(index, index_path)
        total_bytes = sum(
            int(item["tensor_bytes"]) + int(item["metadata_bytes"])
            for item in index["shards"]
        )
        progress = {
            "schema_version": SCHEMA_VERSION,
            "status": "PASS" if index["complete"] else "IN_PROGRESS",
            "config_digest": digest,
            "completed_images": len(completed_ids),
            "total_images": len(ordered_ids),
            "completed_shards": len(index["shards"]),
            "total_shards": total_shards,
            "cache_bytes": total_bytes,
            "model_load_seconds": model_load_seconds,
            "session_elapsed_seconds": time.perf_counter() - all_started,
            "official_test_images": 0,
        }
        if index["complete"]:
            durations = [
                float(item["extraction_seconds"]) for item in index["records"]
            ]
            peaks = [int(item["peak_gpu_memory_bytes"]) for item in index["records"]]
            progress.update(
                {
                    "mean_extraction_seconds": sum(durations) / len(durations),
                    "total_extraction_seconds": sum(durations),
                    "peak_gpu_memory_bytes": max(peaks),
                    "stage_plan": extractor.stage_plan,
                    "storage_format": "safetensors_sharded_v1",
                    "reload_validated_shards": len(index["shards"]),
                    "full_240_image_development_cache_complete": True,
                    "official_test_split_untouched": True,
                }
            )
        atomic_json_write(progress, args.output_dir / "extraction_summary.json")
        print(
            f"Committed shard {shard_number + 1}/{total_shards}; "
            f"{len(completed_ids)}/{len(ordered_ids)} images indexed"
        )
        del records
        gc.collect()
        torch.cuda.empty_cache()

    print("Phase 2 240-image development cache PASS")
    print(f"Index: {index_path}")
    print("STOP: inspect extraction_summary.json before training Phase 3 probes")


if __name__ == "__main__":
    main()
