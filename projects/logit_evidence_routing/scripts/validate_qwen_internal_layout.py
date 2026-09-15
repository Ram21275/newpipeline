#!/usr/bin/env python3
"""Run the mandatory one-image Qwen architecture/token-layout gate on Kaggle."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch
from PIL import Image


PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from lger.hf_llava import _resolve_language_projection  # noqa: E402
from lger.hf_qwen import HfQwenDecisionRunner  # noqa: E402
from lger.multimodal_dola import stack_projected_layers  # noqa: E402
from lger.qwen_layout import expected_merged_visual_tokens, qwen_layout_assertions  # noqa: E402
from lger.reproducibility import current_git_commit  # noqa: E402
from lger.stage_cache import atomic_json_write  # noqa: E402


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def first_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            try:
                return first_tensor(item)
            except RuntimeError:
                pass
    raise RuntimeError("hook output did not contain a tensor")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=PROJECT / "configs/qwen25vl_7b_replication.json"
    )
    parser.add_argument(
        "--protocol", type=Path, default=PROJECT / "configs/qwen_internal_stages.json"
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument(
        "--prompt", default="Is the bird's wing predominantly black? Answer yes or no."
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = read_json(args.config)
    protocol = read_json(args.protocol)
    if not (
        config.get("adapter") == "qwen2_5_vl"
        and protocol.get("development_only") is True
        and protocol.get("official_test_images_used") == 0
    ):
        raise RuntimeError("Qwen internal validation requires the frozen development protocol")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "FAIL",
        "git_commit": current_git_commit(REPO),
        "config_sha256": sha256(args.config),
        "protocol_sha256": sha256(args.protocol),
        "image_sha256": sha256(args.image),
        "model": config["model"],
        "requested_revision": config["revision"],
        "official_test_images_used": 0,
    }
    try:
        runner = HfQwenDecisionRunner.from_pretrained(
            str(config["model"]),
            revision=str(config["revision"]),
            quantization=str(config["quantization"]),
            positive_answer=str(config.get("positive_answer", "yes")),
            negative_answer=str(config.get("negative_answer", "no")),
            attention_layer_offset=int(config.get("attention_layer_offset", -2)),
        )
        if runner.resolved_revision != config["revision"]:
            raise RuntimeError("resolved Qwen revision differs from the frozen config")
        with Image.open(args.image) as opened:
            image = opened.convert("RGB")
        inputs, visual_positions, query_position = runner._prepare(image, args.prompt)
        if visual_positions is None or query_position is None:
            raise RuntimeError("Qwen processor did not expose visual/query positions")
        grid = inputs.get("image_grid_thw")
        if grid is None:
            raise RuntimeError("Qwen processor did not return image_grid_thw")
        visual = getattr(runner.model, "visual", None)
        merger = getattr(visual, "merger", None)
        if visual is None or merger is None:
            raise RuntimeError("could not locate Qwen visual tower and merger modules")
        merge_size = int(
            getattr(getattr(runner.model.config, "vision_config", None), "spatial_merge_size", 0)
            or getattr(visual, "spatial_merge_size", 0)
        )
        if merge_size <= 0:
            raise RuntimeError("could not resolve Qwen spatial_merge_size")
        captured: dict[str, tuple[int, ...]] = {}

        def visual_hook(_: Any, __: tuple[Any, ...], output: Any) -> None:
            captured["visual_output"] = tuple(int(value) for value in first_tensor(output).shape)

        def merger_pre_hook(_: Any, values: tuple[Any, ...]) -> None:
            captured["merger_input"] = tuple(int(value) for value in first_tensor(values).shape)

        def merger_hook(_: Any, __: tuple[Any, ...], output: Any) -> None:
            captured["merger_output"] = tuple(int(value) for value in first_tensor(output).shape)

        handles = [
            visual.register_forward_hook(visual_hook),
            merger.register_forward_pre_hook(merger_pre_hook),
            merger.register_forward_hook(merger_hook),
        ]
        try:
            with torch.inference_mode():
                outputs = runner.model(
                    **inputs,
                    output_hidden_states=True,
                    output_attentions=False,
                    use_cache=False,
                    return_dict=True,
                )
        finally:
            for handle in handles:
                handle.remove()
        hidden_states = tuple(outputs.hidden_states)
        if "visual_output" not in captured or "merger_output" not in captured:
            raise RuntimeError("Qwen visual hooks did not run")
        visual_output_tokens = int(captured["visual_output"][0])
        expected_by_image = expected_merged_visual_tokens(
            grid.detach().cpu(), spatial_merge_size=merge_size
        )
        assertions = qwen_layout_assertions(
            input_image_tokens=int(visual_positions.numel()),
            expected_image_tokens=expected_by_image,
            visual_output_tokens=visual_output_tokens,
            input_sequence_tokens=int(inputs["input_ids"].shape[1]),
            hidden_sequence_tokens=int(hidden_states[-1].shape[1]),
        )
        language_layers = len(hidden_states) - 1
        candidate_indices = tuple(sorted({max(1, language_layers // 2), max(1, language_layers - 2)}))
        final_norm, lm_head = _resolve_language_projection(runner.model)
        with torch.inference_mode():
            projected = stack_projected_layers(
                [hidden_states[index] for index in candidate_indices],
                final_norm=final_norm,
                lm_head=lm_head,
                query_positions=-1,
            )
        if not bool(torch.isfinite(projected).all()):
            raise RuntimeError("intermediate query states did not project to finite logits")
        report.update(
            {
                "status": "PASS",
                "resolved_revision": runner.resolved_revision,
                "image_grid_thw": grid.detach().cpu().tolist(),
                "spatial_merge_size": merge_size,
                "expected_merged_tokens_by_image": list(expected_by_image),
                "visual_positions": int(visual_positions.numel()),
                "query_position": int(query_position),
                "input_sequence_tokens": int(inputs["input_ids"].shape[1]),
                "language_layers": language_layers,
                "hidden_state_count": len(hidden_states),
                "hidden_width": int(hidden_states[-1].shape[-1]),
                "vocabulary_size": int(outputs.logits.shape[-1]),
                "projection_candidate_indices": list(candidate_indices),
                "projection_shape": list(projected.shape),
                "captured_shapes": {key: list(value) for key, value in captured.items()},
                "assertions": assertions,
            }
        )
    except Exception as error:
        report.update(
            {
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
        )
        raise
    finally:
        report["runtime_seconds"] = time.monotonic() - started
        atomic_json_write(report, args.output)
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
