"""Shared full/crop/random/oracle/DoLa experiment on locked questions."""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

from PIL import Image

from lger.cub import load_cub_part_locations

from .core import ShardWriter, canonical_hash, file_sha256, read_jsonl
from .dola import restricted_binary_dola
from .geometry import padded_union_crop
from .models import FrozenVlmRunner, load_runner, unload_runner
from .selectors import predicted_square_crop


PARTS_BY_ATTRIBUTE_GROUP = {
    "has_bill_shape": (2,),
    "has_bill_color": (2,),
    "has_head_pattern": (5, 6, 7, 10, 11),
    "has_crown_color": (5,),
    "has_forehead_color": (6,),
    "has_eye_color": (7, 11),
    "has_nape_color": (10,),
    "has_throat_color": (15,),
    "has_breast_color": (4,),
    "has_belly_color": (3,),
    "has_underparts_color": (3, 4, 15),
    "has_upperparts_color": (1, 10),
    "has_back_color": (1,),
    "has_wing_color": (9, 13),
    "has_wing_pattern": (9, 13),
    "has_tail_shape": (14,),
    "has_tail_pattern": (14,),
    "has_tail_color": (14,),
    "has_leg_color": (8, 12),
}


def _implementation_hash() -> str:
    """Invalidate resumable shards when measurement semantics change."""

    root = Path(__file__).resolve().parent
    names = ("shared.py", "models.py", "scoring.py", "selectors.py", "dola.py", "geometry.py")
    return canonical_hash({name: file_sha256(root / name) for name in names})


def _finished_condition(writer: ShardWriter, key: str, condition: str) -> bool:
    value = writer.read(key)
    if value is None:
        return False
    return value.get("status") == "complete" or (
        condition == "oracle_part_crop" and value.get("status") == "excluded"
    )


def _stable_seed(seed: int, text: str) -> int:
    digest = hashlib.sha256(f"{seed}:{text}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _random_crop(
    image_size: tuple[int, int], *, crop_fraction: float, seed: int
) -> tuple[int, int, int, int]:
    width, height = image_size
    side = min(width, height) * crop_fraction
    rng = random.Random(seed)
    x1 = rng.uniform(0, max(0.0, width - side))
    y1 = rng.uniform(0, max(0.0, height - side))
    return round(x1), round(y1), round(x1 + side), round(y1 + side)


def _record(
    question: dict[str, Any],
    *,
    architecture: str,
    condition: str,
    result: dict[str, Any],
    crop_box: tuple[int, int, int, int] | None,
) -> dict[str, Any]:
    return {
        **question,
        "model": architecture,
        "condition": condition,
        "repeat": 0,
        "crop_box_original_pixels": list(crop_box) if crop_box else None,
        **result,
    }


def _dola_record(
    ordinary: dict[str, Any], *, condition: str, candidate_layers: list[int], alpha: float
) -> dict[str, Any]:
    layer_logits = ordinary.get("answer_first_token_layer_logits")
    if not layer_logits:
        raise RuntimeError("DoLa requires captured answer-position layer logits")
    dola = restricted_binary_dola(
        layer_logits, candidate_layers=candidate_layers, plausibility_alpha=alpha
    )
    return {
        **ordinary,
        "condition": condition,
        "ordinary_positive_log_probability": ordinary["positive_log_probability"],
        "ordinary_negative_log_probability": ordinary["negative_log_probability"],
        # Contrasted DoLa values are scores, not calibrated log probabilities.
        "positive_log_probability": dola["positive_score"],
        "negative_log_probability": dola["negative_score"],
        "dola": dola,
        "score_semantics": "contrastive_score_not_log_probability",
    }


def run_shared_experiment(
    *,
    architecture: str,
    cub_root: Path,
    protocol_path: Path,
    question_manifest: Path,
    output_dir: Path,
    quantization: str = "none",
    local_snapshot: str | None = None,
) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "LOCKED":
        raise RuntimeError("final runs require a LOCKED protocol")
    questions = read_jsonl(question_manifest)
    if not questions:
        raise ValueError("shared experiment question manifest is empty")
    config = {
        "architecture": architecture,
        "protocol_hash": protocol["protocol_hash"],
        "question_manifest_hash": canonical_hash(questions),
        "quantization": quantization,
        "implementation_hash": _implementation_hash(),
    }
    writer = ShardWriter(output_dir / architecture / "shared_shards", config_hash=canonical_hash(config))
    records_by_id = {
        int(parts[0]): parts[1]
        for parts in (
            line.strip().split(maxsplit=1)
            for line in (cub_root / "images.txt").read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    parts = load_cub_part_locations(cub_root)
    crop_fraction = float(protocol["shared_reencoding"].get("crop_fraction", 0.35))
    minimum_fraction = float(protocol["shared_reencoding"].get("oracle_minimum_fraction", 0.12))
    alpha = float(protocol["shared_reencoding"]["plausibility_alpha"])
    runner = load_runner(
        architecture, quantization=quantization, local_snapshot=local_snapshot
    )
    completed = 0
    try:
        for question in questions:
            question_id = str(question["question_id"])
            expected = [
                "full",
                "predicted_crop",
                "random_crop",
                "oracle_part_crop",
                "full_dola",
                "predicted_crop_dola",
            ]
            if architecture == "qwen3":
                expected.append("full_double_pixel_budget")
            if all(
                _finished_condition(
                    writer, f"{question_id}::{condition}", condition
                )
                for condition in expected
            ):
                continue
            image = Image.open(cub_root / "images" / records_by_id[int(question["image_id"])]).convert("RGB")
            full_key = f"{question_id}::full"
            cached_full = writer.read(full_key)
            if cached_full is None:
                full = runner.evaluate(image, str(question["prompt"]), capture_layers=True, capture_attention=True)
                full_record = _record(
                    question, architecture=architecture, condition="full", result=full, crop_box=None
                )
                writer.write(full_key, full_record)
            else:
                full_record = dict(cached_full["payload"])
                full = full_record

            scores = full.get("decoder_attention_scores")
            grid = full.get("merged_grid_size")
            if scores is None or grid is None or len(scores) != int(grid[0]) * int(grid[1]):
                reason = {"reason": "decoder attention/grid unavailable", "question": question}
                for failed_condition in ("predicted_crop", "predicted_crop_dola"):
                    failed_key = f"{question_id}::{failed_condition}"
                    writer.write(failed_key, reason, status="failed")
                predicted_box = None
            else:
                selected = max(range(len(scores)), key=lambda index: (float(scores[index]), -index))
                predicted_box = predicted_square_crop(
                    selected,
                    grid_size=(int(grid[0]), int(grid[1])),
                    image_size=image.size,
                    crop_fraction=crop_fraction,
                )
            random_box = _random_crop(
                image.size,
                crop_fraction=crop_fraction,
                seed=_stable_seed(protocol["seeds"]["cohort"], question_id),
            )
            group_parts = PARTS_BY_ATTRIBUTE_GROUP.get(str(question["attribute_group"]), ())
            visible_points = [
                (value.x, value.y)
                for value in parts[int(question["image_id"])]
                if value.visible and value.part_id in group_parts
            ]
            oracle_box = (
                padded_union_crop(
                    visible_points,
                    image_size=image.size,
                    padding_fraction=float(protocol["shared_reencoding"].get("crop_padding_fraction", 0.25)),
                    minimum_fraction=minimum_fraction,
                )
                if visible_points
                else None
            )
            ordinary_by_condition: dict[str, dict[str, Any]] = {"full": full_record}
            for condition, box in (
                ("predicted_crop", predicted_box),
                ("random_crop", random_box),
                ("oracle_part_crop", oracle_box),
            ):
                key = f"{question_id}::{condition}"
                cached = writer.read(key)
                if cached is not None and cached["status"] == "complete":
                    ordinary_by_condition[condition] = dict(cached["payload"])
                    continue
                if box is None:
                    if not _finished_condition(writer, key, condition):
                        writer.write(
                            key,
                            {"reason": "no visible part for attribute-group oracle", "question": question},
                            status="excluded",
                        )
                    continue
                evaluated = runner.evaluate(
                    image.crop(box),
                    str(question["prompt"]),
                    capture_layers=True,
                    capture_attention=False,
                )
                record = _record(
                    question,
                    architecture=architecture,
                    condition=condition,
                    result=evaluated,
                    crop_box=box,
                )
                ordinary_by_condition[condition] = record
                if not _finished_condition(writer, key, condition):
                    writer.write(key, record)

            layer_count = len(full["answer_first_token_layer_logits"])
            candidates = sorted({
                max(0, min(layer_count - 2, round((layer_count - 1) * fraction)))
                for fraction in (0.25, 0.5, 0.75)
            })
            for source_condition, dola_condition in (
                ("full", "full_dola"),
                ("predicted_crop", "predicted_crop_dola"),
            ):
                key = f"{question_id}::{dola_condition}"
                if source_condition not in ordinary_by_condition:
                    continue
                record = _dola_record(
                    ordinary_by_condition[source_condition],
                    condition=dola_condition,
                    candidate_layers=candidates,
                    alpha=alpha,
                )
                if not _finished_condition(writer, key, dola_condition):
                    writer.write(key, record)

            if architecture == "qwen3":
                # Honest two-pass compute bar: same full image upscaled so the
                # processor realizes roughly twice as many visual tokens.
                scale = math.sqrt(2.0)
                enlarged = image.resize(
                    (round(image.width * scale), round(image.height * scale)),
                    resample=Image.Resampling.BICUBIC,
                )
                condition = "full_double_pixel_budget"
                key = f"{question_id}::{condition}"
                if not _finished_condition(writer, key, condition):
                    evaluated = runner.evaluate(
                        enlarged,
                        str(question["prompt"]),
                        capture_layers=False,
                        capture_attention=False,
                    )
                    record = _record(
                        question,
                        architecture=architecture,
                        condition=condition,
                        result=evaluated,
                        crop_box=None,
                    )
                    record["compute_bar_scope"] = (
                        "pixel-area doubled; realized tokens and wall time are authoritative"
                    )
                    writer.write(key, record)
            completed += 1
    finally:
        unload_runner(runner)
    consolidated = output_dir / architecture / "shared_per_example_shards.jsonl"
    shard_count = writer.consolidate(consolidated)
    consolidated_rows = read_jsonl(consolidated)
    condition_count = 7 if architecture == "qwen3" else 6
    expected_shards = len(questions) * condition_count
    if shard_count != expected_shards:
        raise RuntimeError(
            f"{architecture} shared run has {shard_count}/{expected_shards} terminal shards"
        )
    failures = [row for row in consolidated_rows if row.get("status") == "failed"]
    if failures:
        sample_keys = [str(row.get("key")) for row in failures[:5]]
        raise RuntimeError(
            f"{architecture} shared run has {len(failures)} failed required shards; "
            f"examples={sample_keys}"
        )
    excluded_count = sum(row.get("status") == "excluded" for row in consolidated_rows)
    return {
        "status": "COMPLETE",
        "architecture": architecture,
        "questions_processed_this_invocation": completed,
        "terminal_shard_count": shard_count,
        "expected_terminal_shard_count": expected_shards,
        "excluded_shard_count": excluded_count,
        "failed_shard_count": 0,
        "consolidated": str(consolidated),
        "config_hash": canonical_hash(config),
    }
