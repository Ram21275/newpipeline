#!/usr/bin/env python3
"""Run exactly one gated, stage-aligned LLaVA cache extraction on Kaggle."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
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
from lger.hf_stage_cache import HfLlavaStageExtractor  # noqa: E402
from lger.reproducibility import current_git_commit  # noqa: E402
from lger.stage_cache import (  # noqa: E402
    SCHEMA_VERSION,
    atomic_json_write,
    atomic_torch_save,
    config_digest,
    patch_centers,
    stage_metadata,
    validate_stage_record,
    write_or_validate_config,
)


DEFAULT_MODEL = "llava-hf/llava-1.5-7b-hf"
DEFAULT_REVISION = "b234b804b114d9e37bb655e11cbbb5f5e971b7a9"


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1-gate", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--attribute-subset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--search-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--cub-root", type=Path)
    parser.add_argument("--image-id", type=int)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--quantization", choices=("4bit", "none"), default="4bit")
    parser.add_argument("--prompt", default="Describe the image briefly.")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Enable a Kaggle GPU accelerator for the Phase 2 smoke test")
    if args.max_new_tokens <= 0:
        raise ValueError("max-new-tokens must be positive")
    if len(args.revision) != 40 or any(
        character not in "0123456789abcdef" for character in args.revision.lower()
    ):
        raise ValueError("Phase 2 requires an immutable 40-character model revision")

    gate = _load_gate(args.phase1_gate, args.manifest)
    subset = json.loads(args.attribute_subset.read_text(encoding="utf-8"))
    if subset.get("selection_split") != "development_train_only":
        raise RuntimeError("attribute subset was not selected from development training")
    if int(subset.get("validation_images_inspected", -1)) != 0:
        raise RuntimeError("attribute subset inspected validation labels")
    subset_manifest = Path(str(subset.get("manifest", ""))).expanduser().resolve()
    if subset_manifest != args.manifest.expanduser().resolve():
        raise RuntimeError("attribute subset and smoke run use different manifests")
    subset_policy = subset.get("config")
    if not isinstance(subset_policy, dict) or {
        str(value).strip().lower()
        for value in subset_policy.get("allowed_certainty_names", [])
    } != {"probably", "definitely"}:
        raise RuntimeError("attribute subset uses an incompatible certainty policy")
    selected_attributes = subset.get("selected_attributes")
    if not isinstance(selected_attributes, list) or not selected_attributes:
        raise RuntimeError("attribute subset is empty or malformed")
    selected_ids = [int(item["attribute_id"]) for item in selected_attributes]
    if (
        len(set(selected_ids)) != len(selected_ids)
        or int(subset.get("selected_attribute_count", -1)) != len(selected_ids)
    ):
        raise RuntimeError("attribute subset count or identity is inconsistent")

    manifest = load_pilot_manifest(args.manifest)
    by_id = {record.image_id: record for record in manifest}
    if len(by_id) != len(manifest):
        raise RuntimeError("pilot manifest contains duplicate image IDs")
    train_image_count = sum(record.split == "train" for record in manifest)
    if int(subset.get("train_images_used", -1)) != train_image_count:
        raise RuntimeError("attribute subset used a different development-training set")
    if args.image_id is None:
        pilot_record = sorted(manifest, key=lambda item: item.image_id)[0]
    else:
        try:
            pilot_record = by_id[args.image_id]
        except KeyError as error:
            raise ValueError(f"image {args.image_id} is not in the pilot manifest") from error

    cub_root = args.cub_root or discover_cub_root(args.search_root)
    subset_cub_root = Path(str(subset.get("cub_root", ""))).expanduser().resolve()
    if subset_cub_root != cub_root.expanduser().resolve():
        raise RuntimeError("attribute subset and smoke run use different CUB roots")
    official_by_id = {record.image_id: record for record in load_cub_records(cub_root)}
    official = official_by_id.get(pilot_record.image_id)
    if official is None or official.official_split != "train":
        raise RuntimeError("smoke image is absent from the official CUB training split")
    if (
        official.relative_path,
        official.label,
        official.class_name,
    ) != (
        pilot_record.relative_path,
        pilot_record.label,
        pilot_record.class_name,
    ):
        raise RuntimeError("official CUB and pilot metadata differ for the smoke image")

    attributes = load_cub_attributes(cub_root)
    available_attribute_ids = {attribute.attribute_id for attribute in attributes}
    if not set(selected_ids).issubset(available_attribute_ids):
        raise RuntimeError("attribute subset contains IDs absent from CUB")
    certainties = load_cub_certainties(cub_root)
    labels = load_cub_image_attribute_labels(
        cub_root, image_ids={pilot_record.image_id}
    )[pilot_record.image_id]
    attribute_targets = materialize_attribute_targets(
        attributes, certainties, labels
    )
    boxes = load_cub_bounding_boxes(cub_root)
    parts = load_cub_part_locations(cub_root).get(pilot_record.image_id, [])
    if pilot_record.image_id not in boxes:
        raise RuntimeError("smoke image has no CUB bounding box")

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

    print(
        f"Loading frozen {args.model}@{args.revision} for one-image stage smoke..."
    )
    model_load_started = time.perf_counter()
    extractor = HfLlavaStageExtractor.from_pretrained(
        args.model,
        revision=args.revision,
        quantization=args.quantization,
    )
    model_load_seconds = time.perf_counter() - model_load_started
    if extractor.resolved_revision != args.revision:
        raise RuntimeError(
            "resolved model revision differs from the requested immutable revision: "
            f"{extractor.resolved_revision} != {args.revision}"
        )
    if any(parameter.requires_grad for parameter in extractor.model.parameters()):
        raise RuntimeError("one or more VLM parameters remain trainable")

    run_config = {
        "schema_version": SCHEMA_VERSION,
        "purpose": "one_image_storage_alignment_smoke_only",
        "storage_format": "torch_pt_smoke_v1_not_for_full_extraction",
        "git_commit": current_git_commit(REPO_ROOT),
        "phase1_gate": {
            "path": str(args.phase1_gate.expanduser().resolve()),
            "sha256": _sha256(args.phase1_gate),
            "status": gate["status"],
        },
        "manifest": str(args.manifest.expanduser().resolve()),
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
        "image_id": pilot_record.image_id,
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
    args.output_dir.mkdir(parents=True, exist_ok=True)
    digest = write_or_validate_config(
        args.output_dir / "run_config.json", run_config
    )
    record_path = args.output_dir / "records" / f"{pilot_record.image_id:05d}.pt"
    if record_path.is_file() and not args.overwrite:
        reloaded = torch.load(record_path, map_location="cpu", weights_only=False)
        validation = validate_stage_record(
            reloaded, expected_config_digest=digest
        )
        if not (args.output_dir / "index.json").is_file() or not (
            args.output_dir / "smoke_summary.json"
        ).is_file():
            raise RuntimeError(
                "A valid smoke record exists but its index/summary is incomplete. "
                "Rerun with --overwrite to rebuild the one-image smoke atomically."
            )
        print(f"Validated existing complete smoke record: {record_path}")
        print(json.dumps(validation, indent=2, sort_keys=True))
        return

    image_path = cub_root / "images" / pilot_record.relative_path
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
    peak_gpu_memory_bytes = int(torch.cuda.max_memory_allocated())

    processed_height, processed_width = extraction.processed_image_size
    crop = center_crop_transform(
        original_size=original_size,
        output_size=(processed_width, processed_height),
    )
    centers_model, centers_normalized = patch_centers(
        extraction.grid_size, extraction.processed_image_size
    )
    box = boxes[pilot_record.image_id]
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

    record = {
        "schema_version": SCHEMA_VERSION,
        "complete": True,
        "config_digest": digest,
        "image": {
            "image_id": pilot_record.image_id,
            "relative_path": pilot_record.relative_path,
            "class_id": pilot_record.label,
            "class_name": pilot_record.class_name,
            "official_split": official.official_split,
            "development_split": pilot_record.split,
            "attributes": attribute_targets,
            "selected_attribute_ids": [
                int(item["attribute_id"]) for item in selected_attributes
            ],
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
            "prompt": args.prompt,
            "rendered_prompt": extraction.rendered_prompt,
            "prompt_token_count": extraction.prompt_token_count,
            "answer_text": extraction.answer_text,
            "answer_token_ids": extraction.answer_token_ids,
            "answer_token_strings": list(extraction.answer_token_strings),
            "answer_token_logits": extraction.answer_token_logits,
        },
    }
    in_memory_validation = validate_stage_record(
        record, expected_config_digest=digest
    )
    atomic_torch_save(record, record_path)
    reloaded = torch.load(record_path, map_location="cpu", weights_only=False)
    reload_validation = validate_stage_record(
        reloaded, expected_config_digest=digest
    )
    if reload_validation != in_memory_validation:
        raise RuntimeError("stage record changed across save/reload")

    record_bytes = record_path.stat().st_size
    projected_images = len(manifest)
    index = {
        "schema_version": SCHEMA_VERSION,
        "config_digest": digest,
        "records": [
            {
                "image_id": pilot_record.image_id,
                "path": str(record_path.relative_to(args.output_dir)),
                "bytes": record_bytes,
                "sha256": _sha256(record_path),
                "complete": True,
            }
        ],
    }
    atomic_json_write(index, args.output_dir / "index.json")
    state_counts: dict[str, int] = {}
    for target in attribute_targets:
        state = str(target["state"])
        state_counts[state] = state_counts.get(state, 0) + 1
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "config_digest": digest,
        "image_id": pilot_record.image_id,
        "model_load_seconds": model_load_seconds,
        "extraction_seconds_per_image": extraction_seconds,
        "projected_images": projected_images,
        "projected_extraction_seconds": extraction_seconds * projected_images,
        "peak_gpu_memory_bytes": peak_gpu_memory_bytes,
        "record_bytes_per_image": record_bytes,
        "projected_record_bytes": record_bytes * projected_images,
        "raw_tensor_bytes_per_image": reload_validation["tensor_bytes"],
        "reload_validation": reload_validation,
        "stage_metadata": record["stage_metadata"],
        "attribute_state_counts": state_counts,
        "answer_text": extraction.answer_text,
        "answer_tokens": extraction.answer_token_ids.numel(),
        "production_storage_decision": "PENDING_REVIEW_OF_THIS_SMOKE_RESULT",
        "full_240_image_extraction_authorized": False,
    }
    atomic_json_write(summary, args.output_dir / "smoke_summary.json")
    print(f"Phase 2 one-image smoke PASS: {record_path}")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("STOP: review storage/alignment/memory before any 240-image extraction")


if __name__ == "__main__":
    main()
