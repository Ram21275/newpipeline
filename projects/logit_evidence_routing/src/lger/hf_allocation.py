"""Qwen2.5-VL 4.51.3 bridge adapter. Real forwards are Kaggle-only.

The internal arms replay cached merger tokens, bypassing the vision tower.
All token counts are checked against processor grid and merger output.
"""

from __future__ import annotations

import math
import time
from contextlib import contextmanager
from typing import Any

import torch
from PIL import Image

from .depth_allocation import grid_centers, inside, pixel_box, realized_box
from .hf_llava import _resolve_language_projection
from .qwen_layout import expected_merged_visual_tokens, qwen_layout_assertions


FULL = (0.0, 0.0, 1.0, 1.0)


def budget_grid(size: tuple[int, int], budget: int) -> tuple[int, int]:
    """Closest aspect ratio on the merged grid, within 2% of requested tokens."""
    if budget < 4 or min(size) <= 0:
        raise ValueError("invalid image size or token budget")
    candidates = []
    for n in range(math.ceil(0.98 * budget), math.floor(1.02 * budget) + 1):
        for h in range(1, n + 1):
            if n % h == 0:
                w = n // h
                candidates.append((abs(math.log((w / h) / (size[0] / size[1]))), abs(n-budget), h, w))
    _, _, h, w = min(candidates)
    return h, w


def prepare_views(runner: Any, image: Image.Image, boxes: list[tuple], budgets: list[int],
                  prompt: str) -> tuple[dict, list[tuple], list[tuple]]:
    if len(boxes) != len(budgets):
        raise ValueError("view/budget mismatch")
    visual_config = runner.model.config.vision_config
    merge, patch = int(visual_config.spatial_merge_size), int(visual_config.patch_size)
    views, grids, actual_boxes = [], [], []
    for box, budget in zip(boxes, budgets):
        region = image.crop(pixel_box(box, image.size))
        h, w = budget_grid(region.size, budget)
        views.append(region.resize((w * patch * merge, h * patch * merge), Image.Resampling.BICUBIC))
        grids.append((h, w))
        actual_boxes.append(realized_box(box, image.size))
    content = [{"type": "image"} for _ in views] + [{"type": "text", "text": prompt}]
    rendered = runner.processor.apply_chat_template([{"role": "user", "content": content}],
                                                   add_generation_prompt=True, tokenize=False)
    raw = runner.processor(text=[rendered], images=views, return_tensors="pt", padding=True)
    actual = expected_merged_visual_tokens(raw["image_grid_thw"], spatial_merge_size=merge)
    if actual != tuple(h*w for h, w in grids):
        raise RuntimeError("processor resized a budgeted view; token contract failed")
    for row, (h, w) in zip(raw["image_grid_thw"].tolist(), grids):
        if row != [1, h * merge, w * merge]:
            raise RuntimeError("unsupported temporal or spatial Qwen layout")
    inputs = {k: v.to(runner.input_device, dtype=runner.compute_dtype if v.is_floating_point() else v.dtype)
              for k, v in raw.items() if isinstance(v, torch.Tensor)}
    return inputs, grids, actual_boxes


def attention_bias_hook(visual_positions: torch.Tensor, selected: torch.Tensor,
                        query_position: int, gain: float, audit: dict):
    """Apply additive log(gain) only to selected visual keys at the answer query."""
    if gain <= 0:
        raise ValueError("gain must be positive")
    def hook(module, args, kwargs):
        mask = kwargs.get("attention_mask")
        if not isinstance(mask, torch.Tensor) or mask.ndim != 4 or mask.shape[-2] <= query_position:
            raise RuntimeError("internal arm requires an explicit 4D eager causal mask")
        keys = visual_positions[selected].to(mask.device)
        updated = mask.clone()
        updated[:, :, query_position, keys] += math.log(gain)
        audit["bias_calls"] = audit.get("bias_calls", 0) + 1
        return args, {**kwargs, "attention_mask": updated}
    return hook


@contextmanager
def cached_visual_intervention(model: Any, cached: torch.Tensor | None, *,
                               selected: torch.Tensor | None, replace: bool):
    """No new visual encoding: temporarily replace the visual forward callable."""
    if cached is None:
        yield
        return
    visual = model.visual
    had_instance_forward = "forward" in visual.__dict__
    old = visual.__dict__.get("forward")
    def forward(*args, **kwargs):
        result = cached.clone()
        if replace:
            if selected is None or not bool(selected.any()) or bool(selected.all()):
                raise RuntimeError("mean replacement requires a nonempty proper token subset")
            indices = selected.to(result.device)
            result[indices] = result[~indices].mean(0, keepdim=True)
        return result
    visual.forward = forward
    try:
        yield
    finally:
        if had_instance_forward:
            visual.forward = old
        else:
            del visual.forward


def measure(runner: Any, inputs: dict, *, capture: bool = False,
            cached_visual: torch.Tensor | None = None, selected: torch.Tensor | None = None,
            intervention: str | None = None, gain: float = 2.0,
            layer_fraction: float = 2/3) -> tuple[dict, dict]:
    """Return scores and audited diagnostics; ties count as incorrect downstream.

    Single-token answer alternatives are a deliberate architecture gate. The
    final lens uses model logits. Its independently projected final state must
    match; the double-normalized path is logged as a diagnostic, never used.
    """
    if len(runner.positive_token_ids) != 1 or len(runner.negative_token_ids) != 1:
        raise RuntimeError("bridge requires audited single-token answer alternatives")
    if intervention not in (None, "attention", "mean_replace"):
        raise ValueError("unknown intervention")
    if intervention and cached_visual is None:
        raise ValueError("internal arms must replay cached visual tokens")
    model = runner.model
    positions = (inputs["input_ids"][0] == runner.image_token_id).nonzero().flatten()
    merge = int(model.config.vision_config.spatial_merge_size)
    expected = expected_merged_visual_tokens(inputs["image_grid_thw"], spatial_merge_size=merge)
    auxiliary, audit = {}, {"bias_calls": 0}
    handles = []
    def visual_hook(module, args, output):
        if not isinstance(output, torch.Tensor) or output.ndim != 2:
            raise RuntimeError("unexpected Qwen merger output")
        auxiliary["visual"] = output.detach().clone()
    handles.append(model.visual.register_forward_hook(visual_hook))
    if intervention == "attention":
        layers = model.model.layers
        if not 0 <= layer_fraction < 1:
            raise ValueError("internal layer fraction must lie in [0, 1)")
        layer = int(len(layers) * layer_fraction)
        handles.append(layers[layer].self_attn.register_forward_pre_hook(
            attention_bias_hook(positions, selected, inputs["input_ids"].shape[1]-1, gain, audit),
            with_kwargs=True))
        audit["intervention_layer_zero_based"] = layer
    torch.cuda.synchronize()
    start = time.perf_counter()
    try:
        with torch.inference_mode(), cached_visual_intervention(
                model, cached_visual, selected=selected, replace=intervention == "mean_replace"):
            outputs = model(**inputs, output_attentions=capture, output_hidden_states=capture,
                            return_dict=True, use_cache=False)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    finally:
        for handle in handles:
            handle.remove()
    if intervention == "attention" and audit["bias_calls"] != 1:
        raise RuntimeError("guided-attention hook did not run exactly once")
    visual = auxiliary["visual"]
    qwen_layout_assertions(input_image_tokens=len(positions), expected_image_tokens=expected,
                           visual_output_tokens=len(visual), input_sequence_tokens=inputs["input_ids"].shape[1],
                           hidden_sequence_tokens=outputs.logits.shape[1])
    logits = outputs.logits[0, -1].float()
    if not bool(torch.isfinite(logits).all()):
        raise RuntimeError("nonfinite output logits")
    pos, neg = runner.positive_token_ids[0], runner.negative_token_ids[0]
    margin = float((logits[pos] - logits[neg]).cpu())
    record = {"margin": margin, "positive_prediction": int(margin > 0), "tie": margin == 0,
              "visual_tokens": len(positions), "view_tokens": list(expected),
              "new_encoded_tokens": len(positions) if cached_visual is None else 0,
              "sequence_tokens": int(inputs["input_ids"].shape[1]), "forward_seconds": elapsed,
              "model_forwards": 1, "vision_forwards": int(cached_visual is None), **audit}
    if capture:
        if outputs.attentions is None or any(a is None for a in outputs.attentions):
            raise RuntimeError("eager per-layer attentions unavailable")
        auxiliary["maps"] = torch.stack([a[0, :, -1, positions.to(a.device)].float().mean(0).cpu()
                                         for a in outputs.attentions])
        norm, head = _resolve_language_projection(model)
        states = outputs.hidden_states
        with torch.inference_mode():
            final = head(states[-1][:, -1]).float()[0]
            max_error = float((final - logits.to(final.device)).abs().max().cpu())
            tolerance = 0.1 + 0.002 * float(logits.abs().max().cpu())
            if max_error > tolerance:
                raise RuntimeError(f"final projection audit failed: {max_error} > {tolerance}")
            double_error = float((head(norm(states[-1][:, -1])).float()[0] - logits.to(final.device)).abs().max().cpu())
            lens, patch_lens, pooled_states = [], [], []
            for index, state in enumerate(states[1:], 1):
                # Final HF hidden state is already normalized. Never apply norm twice.
                projected = state if index == len(states)-1 else norm(state)
                query_logits = logits if index == len(states)-1 else head(projected[:, -1]).float()[0]
                lens.append(float((query_logits[pos] - query_logits[neg]).cpu()))
                patch_states = projected[0, positions.to(projected.device)]
                # Linear vocabulary difference avoids allocating patch x full vocabulary.
                weight = (head.weight[pos] - head.weight[neg]).to(patch_states.device)
                bias = 0.0 if head.bias is None else head.bias[pos] - head.bias[neg]
                patch_lens.append((patch_states.float() @ weight.float() + bias).cpu())
                pooled_states.append(state[0, positions.to(state.device)].float().mean(0).cpu())
            concept_ids = []
            for word in ("bird", "birds"):
                ids = runner.processor.tokenizer.encode(word, add_special_tokens=False)
                if len(ids) != 1 or runner.processor.tokenizer.decode(ids).strip().casefold() != word:
                    raise RuntimeError("generic bird concept does not have an audited lexical token")
                concept_ids.extend(ids)
            concept_state = norm(states[-2])[0, positions.to(states[-2].device)]
            concepts = []
            for chunk in concept_state.split(32):
                probability = head(chunk).float().softmax(-1)
                concepts.append(probability[:, sorted(set(concept_ids))].sum(-1).cpu())
        record.update(final_projection_max_error=max_error, double_norm_max_error=double_error,
                      lens_margin_by_layer=lens)
        auxiliary.update(patch_lens=torch.stack(patch_lens), pooled_states=torch.stack(pooled_states),
                         maps=auxiliary["maps"], logit_concept=torch.cat(concepts))
    del outputs
    return record, auxiliary


def selected_native_tokens(grid: tuple[int, int], boxes: list[tuple]) -> torch.Tensor:
    centers = grid_centers(grid)
    mask = torch.zeros(len(centers), dtype=torch.bool)
    for box in boxes:
        mask |= inside(centers, box)
    if not bool(mask.any()) or bool(mask.all()):
        raise RuntimeError("proposal does not define a nonempty proper native-token region")
    return mask
