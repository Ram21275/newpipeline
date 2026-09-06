"""Schema, alignment, atomicity, and sharding helpers for the Phase 2 cache."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch


SCHEMA_VERSION = 1
REQUIRED_STAGE_NAMES = (
    "vision.early",
    "vision.middle",
    "vision.late",
    "vision.final",
    "projector.output",
    "llm.early",
    "llm.middle",
    "llm.late",
    "llm.final",
)


def normalize_hidden_state_index(index: int, hidden_state_count: int) -> int:
    """Resolve a possibly negative index into a hidden-state tuple."""

    if hidden_state_count <= 0:
        raise ValueError("hidden_state_count must be positive")
    resolved = index + hidden_state_count if index < 0 else index
    if not 0 <= resolved < hidden_state_count:
        raise ValueError(
            f"hidden-state index {index} is outside a tuple of length "
            f"{hidden_state_count}"
        )
    return resolved


def resolve_stage_indices(
    num_hidden_layers: int,
    *,
    late_hidden_state_index: int = -2,
) -> dict[str, int]:
    """Resolve early/middle/late/final hidden-state tuple indices.

    Index 0 is the embedding output and index ``num_hidden_layers`` is the final
    model output. Early and middle use the 25% and 50% depth boundaries. Late is
    explicit and defaults to ``-2``, matching the existing Phase 01 late-state
    contract; final is the last hidden state.
    """

    if num_hidden_layers < 4:
        raise ValueError("stage tracing requires at least four transformer layers")
    count = num_hidden_layers + 1
    indices = {
        "early": max(1, num_hidden_layers // 4),
        "middle": max(2, num_hidden_layers // 2),
        "late": normalize_hidden_state_index(late_hidden_state_index, count),
        "final": num_hidden_layers,
    }
    if list(indices.values()) != sorted(set(indices.values())):
        raise ValueError(
            f"resolved stages must be unique and ordered, found {indices}"
        )
    return indices


def resolve_stage_plan(
    *,
    vision_num_hidden_layers: int,
    llm_num_hidden_layers: int,
    vision_feature_layer: int,
) -> dict[str, int | None]:
    """Resolve the exact stage names used by the one-image extractor."""

    vision = resolve_stage_indices(
        vision_num_hidden_layers,
        late_hidden_state_index=vision_feature_layer,
    )
    llm = resolve_stage_indices(llm_num_hidden_layers)
    plan: dict[str, int | None] = {
        f"vision.{name}": index for name, index in vision.items()
    }
    plan["projector.output"] = None
    plan.update({f"llm.{name}": index for name, index in llm.items()})
    return plan


def patch_centers(
    grid_size: tuple[int, int],
    image_size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return row-major patch centers in model pixels and normalized x/y."""

    rows, columns = grid_size
    height, width = image_size
    if min(rows, columns, height, width) <= 0:
        raise ValueError("grid and image dimensions must be positive")
    row_ids, column_ids = torch.meshgrid(
        torch.arange(rows, dtype=torch.float32),
        torch.arange(columns, dtype=torch.float32),
        indexing="ij",
    )
    normalized = torch.stack(
        ((column_ids + 0.5) / columns, (row_ids + 0.5) / rows), dim=-1
    ).reshape(-1, 2)
    pixels = normalized * torch.tensor([width, height], dtype=torch.float32)
    return pixels, normalized


def config_digest(config: dict[str, Any]) -> str:
    """Hash a JSON-compatible run configuration deterministically."""

    serialized = json.dumps(
        config, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def write_or_validate_config(path: Path, config: dict[str, Any]) -> str:
    """Atomically create a config or reject incompatible resumption."""

    expected_digest = config_digest(config)
    if path.is_file():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous != config:
            raise RuntimeError(
                f"Existing stage-cache configuration differs: {path}. "
                "Use the original arguments or a new output directory."
            )
        return expected_digest
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(config, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return expected_digest


def atomic_json_write(payload: dict[str, Any], destination: Path) -> None:
    """Write JSON through an adjacent temporary file and atomic rename."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def atomic_torch_save(payload: dict[str, Any], destination: Path) -> None:
    """Write a smoke record so interrupted files never appear complete."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)


def tensor_storage_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def stage_metadata(
    tensors: dict[str, torch.Tensor],
    stage_plan: dict[str, int | None],
) -> dict[str, dict[str, object]]:
    """Describe tensor shapes/dtypes and resolved layer indices."""

    return {
        name: {
            "hidden_state_index": stage_plan[name],
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "bytes": tensor_storage_bytes(tensor),
        }
        for name, tensor in tensors.items()
    }


def validate_stage_record(
    record: dict[str, Any],
    *,
    expected_config_digest: str | None = None,
) -> dict[str, object]:
    """Validate completeness, finite tensors, and cross-stage patch alignment."""

    if int(record.get("schema_version", 0)) != SCHEMA_VERSION:
        raise RuntimeError("unsupported or missing stage-cache schema version")
    if record.get("complete") is not True:
        raise RuntimeError("stage-cache record is incomplete")
    if (
        expected_config_digest is not None
        and record.get("config_digest") != expected_config_digest
    ):
        raise RuntimeError("stage-cache record configuration does not match this run")
    spatial = record.get("spatial")
    if not isinstance(spatial, dict):
        raise RuntimeError("stage-cache record is missing spatial metadata")
    patch_count = int(spatial.get("patch_count", 0))
    grid_size = tuple(int(value) for value in spatial.get("grid_size", ()))
    if len(grid_size) != 2 or grid_size[0] * grid_size[1] != patch_count:
        raise RuntimeError("patch count and grid size are inconsistent")
    processed_size = tuple(
        int(value) for value in spatial.get("processed_image_size_hw", ())
    )
    if len(processed_size) != 2 or min(processed_size) <= 0:
        raise RuntimeError("processed image size is missing or invalid")
    expected_centers, expected_normalized = patch_centers(grid_size, processed_size)
    for key in ("patch_centers_model_xy", "patch_centers_normalized_xy"):
        coordinates = spatial.get(key)
        if not isinstance(coordinates, torch.Tensor):
            raise RuntimeError(f"spatial metadata is missing {key}")
        if coordinates.shape != (patch_count, 2):
            raise RuntimeError(f"{key} does not align with the patch grid")
        if not bool(torch.isfinite(coordinates).all()):
            raise RuntimeError(f"{key} contains non-finite values")
    if not torch.allclose(
        spatial["patch_centers_model_xy"].float(), expected_centers
    ) or not torch.allclose(
        spatial["patch_centers_normalized_xy"].float(), expected_normalized
    ):
        raise RuntimeError("stored patch centers are not the row-major grid centers")
    visual_positions = spatial.get("visual_token_positions")
    if (
        not isinstance(visual_positions, torch.Tensor)
        or visual_positions.ndim != 1
        or visual_positions.numel() != patch_count
        or visual_positions.unique().numel() != patch_count
        or visual_positions.dtype
        not in {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}
    ):
        raise RuntimeError("expanded visual token positions are not one-to-one")
    if patch_count > 1 and not bool(
        torch.all(visual_positions[1:] > visual_positions[:-1])
    ):
        raise RuntimeError("expanded visual token positions are not ordered")

    image = record.get("image")
    if not isinstance(image, dict) or image.get("official_split") != "train":
        raise RuntimeError("record is not tied to an official CUB training image")
    attributes = image.get("attributes")
    if not isinstance(attributes, list) or not attributes:
        raise RuntimeError("record is missing CUB attribute annotations")
    valid_attribute_states = {
        "present",
        "absent",
        "uncertain",
        "not_visible",
        "missing",
    }
    if any(
        not isinstance(item, dict) or item.get("state") not in valid_attribute_states
        for item in attributes
    ):
        raise RuntimeError("record contains an invalid attribute-observation state")
    attribute_ids = [int(item.get("attribute_id", 0)) for item in attributes]
    if min(attribute_ids) <= 0 or len(set(attribute_ids)) != len(attribute_ids):
        raise RuntimeError("attribute IDs must be positive and unique")
    expected_targets = {"present": True, "absent": False}
    for item in attributes:
        state = str(item["state"])
        target = item.get("primary_target")
        if state in expected_targets and target is not expected_targets[state]:
            raise RuntimeError("observed attribute state and primary target disagree")
        if state not in expected_targets and target is not None:
            raise RuntimeError("uncertain or missing attribute has a primary target")
    selected_attribute_ids = image.get("selected_attribute_ids")
    if (
        not isinstance(selected_attribute_ids, list)
        or len(set(int(value) for value in selected_attribute_ids))
        != len(selected_attribute_ids)
        or not set(int(value) for value in selected_attribute_ids).issubset(
            attribute_ids
        )
    ):
        raise RuntimeError("selected attribute IDs do not match cached annotations")

    stages = record.get("stages")
    if not isinstance(stages, dict) or set(stages) != set(REQUIRED_STAGE_NAMES):
        raise RuntimeError("stage-cache record does not contain the required stage set")
    dimensions: dict[str, int] = {}
    for name in REQUIRED_STAGE_NAMES:
        tensor = stages[name]
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2:
            raise RuntimeError(f"stage {name} must be a rank-2 tensor")
        if tensor.shape[0] != patch_count:
            raise RuntimeError(f"stage {name} lost patch correspondence")
        if tensor.requires_grad:
            raise RuntimeError(f"stage {name} was not detached")
        if not bool(torch.isfinite(tensor).all()):
            raise RuntimeError(f"stage {name} contains non-finite values")
        dimensions[name] = int(tensor.shape[1])
    if len({dimensions[name] for name in REQUIRED_STAGE_NAMES[:4]}) != 1:
        raise RuntimeError("vision-stage hidden dimensions do not match")
    language_names = REQUIRED_STAGE_NAMES[4:]
    if len({dimensions[name] for name in language_names}) != 1:
        raise RuntimeError("projector and LLM hidden dimensions do not match")
    metadata = record.get("stage_metadata")
    if not isinstance(metadata, dict) or set(metadata) != set(REQUIRED_STAGE_NAMES):
        raise RuntimeError("stage metadata does not describe the required stage set")
    for name, tensor in stages.items():
        description = metadata[name]
        if (
            description.get("shape") != list(tensor.shape)
            or description.get("dtype") != str(tensor.dtype).removeprefix("torch.")
            or int(description.get("bytes", -1)) != tensor_storage_bytes(tensor)
        ):
            raise RuntimeError(f"stage metadata does not match tensor {name}")

    generation = record.get("generation")
    if not isinstance(generation, dict):
        raise RuntimeError("stage-cache record is missing generation output")
    answer_ids = generation.get("answer_token_ids")
    answer_logits = generation.get("answer_token_logits")
    if not isinstance(answer_ids, torch.Tensor) or answer_ids.ndim != 1:
        raise RuntimeError("answer token IDs must be a rank-1 tensor")
    if answer_ids.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise RuntimeError("answer token IDs must use an integer dtype")
    if not isinstance(answer_logits, torch.Tensor) or answer_logits.ndim != 2:
        raise RuntimeError("answer token logits must be a rank-2 tensor")
    token_strings = generation.get("answer_token_strings")
    if (
        answer_logits.shape[0] != answer_ids.numel()
        or not isinstance(token_strings, list)
        or len(token_strings) != answer_ids.numel()
    ):
        raise RuntimeError("answer logits do not align with generated tokens")
    if answer_logits.shape[1] <= 1 or not answer_logits.is_floating_point():
        raise RuntimeError("answer token logits do not span a floating vocabulary")
    if answer_logits.requires_grad:
        raise RuntimeError("answer token logits were not detached")
    if not bool(torch.isfinite(answer_logits).all()):
        raise RuntimeError("answer token logits contain non-finite values")

    tensor_bytes = sum(tensor_storage_bytes(value) for value in stages.values())
    tensor_bytes += tensor_storage_bytes(answer_ids)
    tensor_bytes += tensor_storage_bytes(answer_logits)
    return {
        "patch_count": patch_count,
        "stage_count": len(stages),
        "answer_tokens": answer_ids.numel(),
        "tensor_bytes": tensor_bytes,
        "vision_hidden_size": dimensions["vision.final"],
        "language_hidden_size": dimensions["llm.final"],
    }


_TENSOR_REFERENCE_KEY = "__stage_cache_tensor__"


def pack_stage_record(
    record: dict[str, Any], *, image_id: int
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Split one nested record into JSON metadata and safetensors-ready tensors.

    Tensor names are namespaced by image ID and structural path, which keeps the
    production shards self-describing without relying on insertion order.
    """

    if image_id <= 0:
        raise ValueError("image_id must be positive")
    tensors: dict[str, torch.Tensor] = {}
    prefix = f"image/{image_id:05d}"

    def visit(value: Any, path: tuple[str, ...]) -> Any:
        if isinstance(value, torch.Tensor):
            if not path:
                raise ValueError("a stage record cannot be a bare tensor")
            tensor_key = "/".join((prefix, *path))
            if tensor_key in tensors:
                raise ValueError(f"duplicate packed tensor key: {tensor_key}")
            tensors[tensor_key] = value.detach().cpu().contiguous()
            return {_TENSOR_REFERENCE_KEY: tensor_key}
        if isinstance(value, dict):
            if any(not isinstance(key, str) for key in value):
                raise TypeError("stage-record dictionaries require string keys")
            return {key: visit(item, (*path, key)) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [visit(item, (*path, str(index))) for index, item in enumerate(value)]
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        raise TypeError(f"unsupported stage-record value at {'/'.join(path)}: {type(value)}")

    packed = visit(record, ())
    if not isinstance(packed, dict) or not tensors:
        raise ValueError("packed stage record is empty or contains no tensors")
    return packed, tensors


def packed_tensor_keys(packed_record: dict[str, Any]) -> set[str]:
    """Return every tensor reference embedded in packed record metadata."""

    keys: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            if set(value) == {_TENSOR_REFERENCE_KEY}:
                key = value[_TENSOR_REFERENCE_KEY]
                if not isinstance(key, str) or not key:
                    raise RuntimeError("packed record contains an invalid tensor reference")
                if key in keys:
                    raise RuntimeError(f"packed record repeats tensor reference {key}")
                keys.add(key)
                return
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(packed_record)
    if not keys:
        raise RuntimeError("packed record does not reference any tensors")
    return keys


def unpack_stage_record(
    packed_record: dict[str, Any], tensors: dict[str, torch.Tensor]
) -> dict[str, Any]:
    """Reconstruct a stage record and reject missing or surplus tensors."""

    expected = packed_tensor_keys(packed_record)
    actual = set(tensors)
    if actual != expected:
        missing = sorted(expected - actual)
        surplus = sorted(actual - expected)
        raise RuntimeError(
            f"packed tensor set differs (missing={missing[:3]}, surplus={surplus[:3]})"
        )

    def visit(value: Any) -> Any:
        if isinstance(value, dict):
            if set(value) == {_TENSOR_REFERENCE_KEY}:
                return tensors[value[_TENSOR_REFERENCE_KEY]]
            return {key: visit(item) for key, item in value.items()}
        if isinstance(value, list):
            return [visit(item) for item in value]
        return value

    unpacked = visit(packed_record)
    if not isinstance(unpacked, dict):
        raise RuntimeError("unpacked stage record is not a dictionary")
    return unpacked


def plan_pending_shards(
    ordered_image_ids: list[int],
    completed_image_ids: set[int],
    *,
    shard_size: int,
) -> list[list[int]]:
    """Plan deterministic remaining shards for prefix-only resumable extraction."""

    if shard_size <= 0:
        raise ValueError("shard_size must be positive")
    if len(set(ordered_image_ids)) != len(ordered_image_ids):
        raise ValueError("ordered image IDs contain duplicates")
    if not completed_image_ids.issubset(ordered_image_ids):
        raise RuntimeError("cache index contains image IDs outside the manifest")
    prefix = set(ordered_image_ids[: len(completed_image_ids)])
    if completed_image_ids != prefix:
        raise RuntimeError("completed images are not a deterministic manifest prefix")
    remaining = ordered_image_ids[len(completed_image_ids) :]
    return [
        remaining[offset : offset + shard_size]
        for offset in range(0, len(remaining), shard_size)
    ]


def write_safetensors_shard(
    records: list[tuple[int, dict[str, Any]]],
    *,
    tensor_path: Path,
    metadata_path: Path,
    expected_config_digest: str,
) -> dict[str, Any]:
    """Atomically write one production shard and its JSON reconstruction map."""

    if not records:
        raise ValueError("cannot write an empty stage-cache shard")
    try:
        from safetensors.torch import save_file
    except ImportError as error:  # pragma: no cover - Kaggle dependency guard
        raise RuntimeError(
            "Production stage shards require safetensors; install requirements-kaggle.txt"
        ) from error

    image_ids = [image_id for image_id, _ in records]
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("a stage-cache shard cannot contain duplicate image IDs")
    packed_records: list[dict[str, Any]] = []
    tensors: dict[str, torch.Tensor] = {}
    for image_id, record in records:
        image = record.get("image")
        if (
            not isinstance(image, dict)
            or int(image.get("image_id", 0)) != image_id
        ):
            raise RuntimeError("record image identity differs from its shard key")
        validation = validate_stage_record(
            record, expected_config_digest=expected_config_digest
        )
        packed, record_tensors = pack_stage_record(record, image_id=image_id)
        overlap = set(tensors).intersection(record_tensors)
        if overlap:
            raise RuntimeError(f"duplicate tensor names across records: {sorted(overlap)[:3]}")
        tensors.update(record_tensors)
        packed_records.append(
            {
                "image_id": image_id,
                "validation": validation,
                "packed_record": packed,
            }
        )

    tensor_path.parent.mkdir(parents=True, exist_ok=True)
    tensor_temporary = tensor_path.with_suffix(tensor_path.suffix + ".tmp")
    save_file(
        tensors,
        str(tensor_temporary),
        metadata={
            "schema_version": str(SCHEMA_VERSION),
            "config_digest": expected_config_digest,
            "image_ids": ",".join(str(value) for value in image_ids),
        },
    )
    tensor_temporary.replace(tensor_path)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "complete": True,
        "config_digest": expected_config_digest,
        "tensor_file": tensor_path.name,
        "image_ids": image_ids,
        "records": packed_records,
    }
    atomic_json_write(payload, metadata_path)
    return payload


def validate_safetensors_shard(
    tensor_path: Path,
    metadata_path: Path,
    *,
    expected_config_digest: str,
) -> list[dict[str, object]]:
    """Reload every record in one shard and re-run the full schema validator."""

    try:
        from safetensors import safe_open
    except ImportError as error:  # pragma: no cover - Kaggle dependency guard
        raise RuntimeError(
            "Production stage shards require safetensors; install requirements-kaggle.txt"
        ) from error
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (
        int(payload.get("schema_version", 0)) != SCHEMA_VERSION
        or payload.get("complete") is not True
        or payload.get("config_digest") != expected_config_digest
        or payload.get("tensor_file") != tensor_path.name
    ):
        raise RuntimeError(f"stage shard metadata is incomplete or incompatible: {metadata_path}")
    packed_records = payload.get("records")
    image_ids = payload.get("image_ids")
    if (
        not isinstance(packed_records, list)
        or not packed_records
        or not isinstance(image_ids, list)
        or [int(item.get("image_id", 0)) for item in packed_records]
        != [int(value) for value in image_ids]
    ):
        raise RuntimeError(f"stage shard record index is malformed: {metadata_path}")

    validations: list[dict[str, object]] = []
    with safe_open(tensor_path, framework="pt", device="cpu") as handle:
        header = handle.metadata() or {}
        if (
            header.get("schema_version") != str(SCHEMA_VERSION)
            or header.get("config_digest") != expected_config_digest
            or header.get("image_ids")
            != ",".join(str(int(value)) for value in image_ids)
        ):
            raise RuntimeError(f"safetensors header is incompatible: {tensor_path}")
        expected_keys: set[str] = set()
        for item in packed_records:
            packed = item.get("packed_record")
            if not isinstance(packed, dict):
                raise RuntimeError("stage shard contains malformed packed metadata")
            record_keys = packed_tensor_keys(packed)
            if expected_keys.intersection(record_keys):
                raise RuntimeError("stage shard repeats tensor keys across records")
            expected_keys.update(record_keys)
        if set(handle.keys()) != expected_keys:
            raise RuntimeError("safetensors keys do not match shard metadata")
        for item in packed_records:
            packed = item["packed_record"]
            record_keys = packed_tensor_keys(packed)
            tensors = {key: handle.get_tensor(key) for key in record_keys}
            record = unpack_stage_record(packed, tensors)
            image = record.get("image")
            if (
                not isinstance(image, dict)
                or int(image.get("image_id", 0)) != int(item["image_id"])
            ):
                raise RuntimeError("stage shard image identity does not round-trip")
            validation = validate_stage_record(
                record, expected_config_digest=expected_config_digest
            )
            if validation != item.get("validation"):
                raise RuntimeError("stage shard validation changed after reload")
            validations.append(validation)
    return validations
