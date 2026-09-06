"""One-image, stage-aligned LLaVA extraction for the Phase 2 Kaggle smoke test."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .hf_llava import HfLlavaPatchExtractor, square_grid, visual_positions_and_query
from .stage_cache import resolve_stage_plan


@dataclass(frozen=True)
class StageExtraction:
    """Aligned patch states and generation outputs from one frozen VLM run."""

    stages: dict[str, torch.Tensor]
    stage_plan: dict[str, int | None]
    grid_size: tuple[int, int]
    processed_image_size: tuple[int, int]
    visual_token_positions: torch.Tensor
    prompt_token_count: int
    rendered_prompt: str
    answer_text: str
    answer_token_ids: torch.Tensor
    answer_token_strings: tuple[str, ...]
    answer_token_logits: torch.Tensor


def _model_component(model: nn.Module, name: str) -> nn.Module:
    direct = getattr(model, name, None)
    if isinstance(direct, nn.Module):
        return direct
    base_model = getattr(model, "model", None)
    nested = getattr(base_model, name, None)
    if isinstance(nested, nn.Module):
        return nested
    raise RuntimeError(f"Could not locate LLaVA component {name!r}")


def _hidden_states(output: object, component: str) -> tuple[torch.Tensor, ...]:
    states = getattr(output, "hidden_states", None)
    if states is None and isinstance(output, tuple):
        states = next(
            (
                value
                for value in output
                if isinstance(value, tuple)
                and value
                and isinstance(value[0], torch.Tensor)
            ),
            None,
        )
    if not isinstance(states, tuple) or not states:
        raise RuntimeError(f"{component} did not return hidden states")
    return states


def _cpu_float16(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to(device="cpu", dtype=torch.float16).contiguous()


def _spatial_preprocessing_config(processor: Any) -> dict[str, object]:
    """Validate and serialize the classic LLaVA center-crop contract."""

    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        raise RuntimeError("processor does not expose an image processor")
    size = dict(getattr(image_processor, "size", {}))
    crop_size = dict(getattr(image_processor, "crop_size", {}))
    shortest_edge = int(size.get("shortest_edge", 0))
    crop_height = int(crop_size.get("height", 0))
    crop_width = int(crop_size.get("width", 0))
    if not bool(getattr(image_processor, "do_resize", False)):
        raise RuntimeError("classic LLaVA spatial alignment requires image resizing")
    if not bool(getattr(image_processor, "do_center_crop", False)):
        raise RuntimeError("classic LLaVA spatial alignment requires a center crop")
    if shortest_edge <= 0 or crop_height <= 0 or crop_width <= 0:
        raise RuntimeError("image processor has incomplete resize/crop dimensions")
    if shortest_edge != crop_height or crop_height != crop_width:
        raise RuntimeError(
            "initial spatial mapping requires equal shortest-edge and square-crop sizes"
        )
    resample = getattr(image_processor, "resample", None)
    try:
        serialized_resample: int | str | None = int(resample)
    except (TypeError, ValueError):
        serialized_resample = None if resample is None else str(resample)
    rescale_factor = getattr(image_processor, "rescale_factor", None)
    return {
        "do_resize": True,
        "size": {"shortest_edge": shortest_edge},
        "resample": serialized_resample,
        "do_center_crop": True,
        "crop_size": {"height": crop_height, "width": crop_width},
        "do_rescale": bool(getattr(image_processor, "do_rescale", False)),
        "rescale_factor": (
            None if rescale_factor is None else float(rescale_factor)
        ),
        "do_normalize": bool(getattr(image_processor, "do_normalize", False)),
        "image_mean": [
            float(value) for value in getattr(image_processor, "image_mean", ())
        ],
        "image_std": [
            float(value) for value in getattr(image_processor, "image_std", ())
        ],
    }


class HfLlavaStageExtractor:
    """Capture selected vision, projector, and LLM states during generation."""

    def __init__(self, base_extractor: HfLlavaPatchExtractor) -> None:
        self.base = base_extractor
        self.model = base_extractor.model
        self.processor = base_extractor.processor
        self.vision_tower = _model_component(self.model, "vision_tower")
        self.projector = _model_component(self.model, "multi_modal_projector")
        self.language_model = _model_component(self.model, "language_model")
        vision_config = self.model.config.vision_config
        text_config = self.model.config.text_config
        feature_layer = self.model.config.vision_feature_layer
        if not isinstance(feature_layer, int):
            raise RuntimeError("Phase 2 smoke supports one vision feature layer")
        self.vision_feature_select_strategy = str(
            self.model.config.vision_feature_select_strategy
        )
        if self.vision_feature_select_strategy != "default":
            raise RuntimeError(
                "Initial Phase 2 alignment requires the classic LLaVA 'default' "
                "vision feature strategy (CLS excluded)"
            )
        self.spatial_preprocessing = _spatial_preprocessing_config(self.processor)
        self.stage_plan = resolve_stage_plan(
            vision_num_hidden_layers=int(vision_config.num_hidden_layers),
            llm_num_hidden_layers=int(text_config.num_hidden_layers),
            vision_feature_layer=feature_layer,
        )

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        *,
        revision: str,
        quantization: str = "4bit",
    ) -> "HfLlavaStageExtractor":
        base = HfLlavaPatchExtractor.from_pretrained(
            model_name,
            revision=revision,
            quantization=quantization,
            layer_offset=-2,
            fixed_concepts=(),
        )
        return cls(base)

    @property
    def resolved_revision(self) -> str:
        return self.base.resolved_revision

    def extract(
        self,
        image: Any,
        prompt_text: str,
        *,
        max_new_tokens: int,
    ) -> StageExtraction:
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        rendered_prompt = self.base.build_prompt(prompt_text)
        inputs = self.processor(
            images=image,
            text=rendered_prompt,
            return_tensors="pt",
        )
        visual_positions, _ = visual_positions_and_query(
            inputs["input_ids"], inputs["attention_mask"], self.base.image_token_id
        )
        patch_count = visual_positions.numel()
        grid_size = square_grid(patch_count)
        pixel_values = inputs.get("pixel_values")
        if not isinstance(pixel_values, torch.Tensor) or pixel_values.ndim != 4:
            raise RuntimeError("processor did not return one image tensor")
        processed_image_size = (
            int(pixel_values.shape[-2]),
            int(pixel_values.shape[-1]),
        )
        expected_crop = self.spatial_preprocessing["crop_size"]
        assert isinstance(expected_crop, dict)
        if processed_image_size != (
            int(expected_crop["height"]),
            int(expected_crop["width"]),
        ):
            raise RuntimeError(
                "processed image tensor does not match the pinned center-crop size"
            )
        moved_inputs: dict[str, torch.Tensor] = {}
        for name, value in inputs.items():
            if not isinstance(value, torch.Tensor):
                continue
            if value.is_floating_point():
                value = value.to(
                    device=self.base.input_device,
                    dtype=self.base.compute_dtype,
                )
            else:
                value = value.to(self.base.input_device)
            moved_inputs[name] = value

        captured: dict[str, torch.Tensor] = {}
        raw_answer_logits: list[torch.Tensor] = []
        visual_positions_cpu = visual_positions.detach().cpu()

        def capture_vision(
            _: nn.Module, __: tuple[object, ...], output: object
        ) -> None:
            if "vision.early" in captured:
                return
            states = _hidden_states(output, "vision tower")
            for name, hidden_index in self.stage_plan.items():
                if not name.startswith("vision."):
                    continue
                assert hidden_index is not None
                state = states[hidden_index]
                if state.ndim != 3 or state.shape[0] != 1:
                    raise RuntimeError(f"{name} has unexpected shape {tuple(state.shape)}")
                if state.shape[1] == patch_count + 1:
                    state = state[:, 1:, :]
                elif state.shape[1] != patch_count:
                    raise RuntimeError(
                        f"{name} has {state.shape[1]} tokens; expected "
                        f"{patch_count} patches plus at most one CLS token"
                    )
                captured[name] = _cpu_float16(state[0])

        def capture_projector(
            _: nn.Module, __: tuple[object, ...], output: object
        ) -> None:
            if "projector.output" in captured:
                return
            if not isinstance(output, torch.Tensor):
                raise RuntimeError("multimodal projector did not return a tensor")
            if output.ndim != 3 or output.shape[:2] != (1, patch_count):
                raise RuntimeError(
                    "projector output does not preserve the visual patch sequence"
                )
            captured["projector.output"] = _cpu_float16(output[0])

        def capture_language(
            _: nn.Module, __: tuple[object, ...], output: object
        ) -> None:
            logits = getattr(output, "logits", None)
            if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
                raise RuntimeError("language model did not return vocabulary logits")
            raw_answer_logits.append(_cpu_float16(logits[0, -1]))
            if "llm.early" in captured:
                return
            states = _hidden_states(output, "language model")
            if states[0].ndim != 3 or states[0].shape[1] <= int(
                visual_positions_cpu.max()
            ):
                return
            for name, hidden_index in self.stage_plan.items():
                if not name.startswith("llm."):
                    continue
                assert hidden_index is not None
                state = states[hidden_index][0]
                positions = visual_positions_cpu.to(state.device)
                captured[name] = _cpu_float16(state.index_select(0, positions))

        handles = [
            self.vision_tower.register_forward_hook(capture_vision),
            self.projector.register_forward_hook(capture_projector),
            self.language_model.register_forward_hook(capture_language),
        ]
        try:
            with torch.inference_mode():
                generated = self.model.generate(
                    **moved_inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    return_dict_in_generate=True,
                    output_hidden_states=True,
                )
        finally:
            for handle in handles:
                handle.remove()

        missing = sorted(set(self.stage_plan) - set(captured))
        if missing:
            raise RuntimeError(f"stage hooks did not capture required tensors: {missing}")
        prompt_token_count = int(moved_inputs["input_ids"].shape[1])
        sequences = generated.sequences
        answer_ids = sequences[0, prompt_token_count:].detach().cpu().to(torch.long)
        if len(raw_answer_logits) != answer_ids.numel():
            raise RuntimeError(
                "raw language-model logits do not align with the generated answer tokens"
            )
        if not raw_answer_logits:
            raise RuntimeError("generation returned no answer tokens")
        answer_logits = torch.stack(raw_answer_logits, dim=0).contiguous()
        tokenizer = self.processor.tokenizer
        answer_text = tokenizer.decode(
            answer_ids.tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        answer_token_strings = tuple(
            tokenizer.convert_ids_to_tokens(answer_ids.tolist())
        )
        result = StageExtraction(
            stages=captured,
            stage_plan=dict(self.stage_plan),
            grid_size=grid_size,
            processed_image_size=processed_image_size,
            visual_token_positions=visual_positions_cpu,
            prompt_token_count=prompt_token_count,
            rendered_prompt=rendered_prompt,
            answer_text=answer_text,
            answer_token_ids=answer_ids,
            answer_token_strings=answer_token_strings,
            answer_token_logits=answer_logits,
        )
        del generated, moved_inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return result
