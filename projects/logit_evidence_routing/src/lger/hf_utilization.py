"""Frozen LLaVA answer-margin and question-conditioned routing adapter.

This module contains only model-facing mechanics.  Phase 5 cohort construction,
parsing, aggregation, and gates live in :mod:`lger.phase5` so they can be tested
without downloading a checkpoint.
"""

from __future__ import annotations

import math
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any, Iterator

import torch
import torch.nn.functional as F

from .hf_llava import HfLlavaPatchExtractor, visual_positions_and_query
from .hf_stage_cache import _model_component
from .stage_cache import resolve_stage_plan


@dataclass(frozen=True)
class FrozenDecisionOutput:
    """One frozen-model answer measurement for a binary question."""

    positive_log_likelihood: float
    negative_log_likelihood: float
    answer_margin: float
    positive_token_ids: tuple[int, ...]
    negative_token_ids: tuple[int, ...]
    generated_text: str | None
    attention_scores: torch.Tensor | None
    attention_entropy: float | None
    attention_effective_tokens: float | None


@dataclass(frozen=True)
class HiddenIntervention:
    """Replace selected aligned visual-patch states at one causal stage."""

    stage: str
    patch_indices: torch.Tensor
    replacement: torch.Tensor


def normalized_entropy(scores: torch.Tensor) -> tuple[float, float]:
    """Return Shannon entropy and effective support for non-negative scores."""

    if scores.ndim != 1 or scores.numel() == 0:
        raise ValueError("routing scores must be a non-empty vector")
    values = scores.detach().float().cpu()
    if not bool(torch.isfinite(values).all()) or bool((values < 0).any()):
        raise ValueError("routing scores must be finite and non-negative")
    probabilities = values / values.sum().clamp_min(1e-12)
    entropy = float(-(probabilities * probabilities.clamp_min(1e-12).log()).sum())
    return entropy, math.exp(entropy)


class HfLlavaDecisionRunner:
    """Measure exact yes/no sequence likelihoods in a frozen classic LLaVA."""

    def __init__(
        self,
        base_extractor: HfLlavaPatchExtractor,
        *,
        positive_answer: str = "yes",
        negative_answer: str = "no",
        attention_layer_offset: int = -2,
    ) -> None:
        if attention_layer_offset >= 0:
            raise ValueError("attention layer offset must be negative")
        self.base = base_extractor
        self.model = base_extractor.model
        self.processor = base_extractor.processor
        self.attention_layer_offset = attention_layer_offset
        self.positive_answer = positive_answer.strip()
        self.negative_answer = negative_answer.strip()
        if not self.positive_answer or not self.negative_answer:
            raise ValueError("binary answer strings cannot be empty")
        self.positive_token_ids = self._answer_token_ids(self.positive_answer)
        self.negative_token_ids = self._answer_token_ids(self.negative_answer)
        if self.positive_token_ids == self.negative_token_ids:
            raise ValueError("positive and negative answers tokenize identically")

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        *,
        revision: str,
        quantization: str = "4bit",
        positive_answer: str = "yes",
        negative_answer: str = "no",
        attention_layer_offset: int = -2,
    ) -> "HfLlavaDecisionRunner":
        base = HfLlavaPatchExtractor.from_pretrained(
            model_name,
            revision=revision,
            quantization=quantization,
            layer_offset=-2,
            fixed_concepts=(),
        )
        return cls(
            base,
            positive_answer=positive_answer,
            negative_answer=negative_answer,
            attention_layer_offset=attention_layer_offset,
        )

    @property
    def resolved_revision(self) -> str:
        return self.base.resolved_revision

    def _answer_token_ids(self, answer: str) -> tuple[int, ...]:
        tokenizer = self.processor.tokenizer
        token_ids = tuple(
            int(value)
            for value in tokenizer.encode(answer, add_special_tokens=False)
        )
        special = {int(value) for value in tokenizer.all_special_ids}
        if not token_ids or any(value in special for value in token_ids):
            raise RuntimeError(f"invalid answer tokenization for {answer!r}")
        decoded = tokenizer.decode(
            list(token_ids),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if decoded.strip().casefold() != answer.casefold():
            raise RuntimeError(
                f"answer tokenization does not round-trip: {answer!r} -> {decoded!r}"
            )
        return token_ids

    def _text_only_prompt(self, prompt_text: str) -> str:
        if getattr(self.processor, "chat_template", None):
            conversation = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt_text}],
                }
            ]
            return self.processor.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=False
            )
        return f"USER: {prompt_text} ASSISTANT:"

    def _prepare(
        self, image: Any | None, prompt_text: str
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor | None, int | None]:
        include_image = image is not None
        rendered = (
            self.base.build_prompt(prompt_text)
            if include_image
            else self._text_only_prompt(prompt_text)
        )
        processor_kwargs: dict[str, Any] = {
            "text": rendered,
            "return_tensors": "pt",
        }
        if include_image:
            processor_kwargs["images"] = image
        inputs = self.processor(**processor_kwargs)
        visual_positions: torch.Tensor | None = None
        query_position: int | None = None
        if include_image:
            visual_positions, query_position = visual_positions_and_query(
                inputs["input_ids"],
                inputs["attention_mask"],
                self.base.image_token_id,
            )
        moved: dict[str, torch.Tensor] = {}
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
            moved[name] = value
        return moved, visual_positions, query_position

    @staticmethod
    def _append_tokens(
        inputs: dict[str, torch.Tensor], token_ids: tuple[int, ...]
    ) -> dict[str, torch.Tensor]:
        result = dict(inputs)
        ids = torch.tensor(
            token_ids,
            dtype=inputs["input_ids"].dtype,
            device=inputs["input_ids"].device,
        ).view(1, -1)
        result["input_ids"] = torch.cat((inputs["input_ids"], ids), dim=1)
        mask = torch.ones(
            (1, len(token_ids)),
            dtype=inputs["attention_mask"].dtype,
            device=inputs["attention_mask"].device,
        )
        result["attention_mask"] = torch.cat(
            (inputs["attention_mask"], mask), dim=1
        )
        if "position_ids" in result:
            result.pop("position_ids")
        return result

    @staticmethod
    def _decoder_layers(module: Any) -> Any:
        candidates = (
            getattr(getattr(module, "model", None), "layers", None),
            getattr(getattr(getattr(module, "model", None), "decoder", None), "layers", None),
            getattr(getattr(module, "transformer", None), "h", None),
        )
        for layers in candidates:
            if layers is not None and len(layers) > 0:
                return layers
        raise RuntimeError("could not locate language-model decoder layers")

    @staticmethod
    def _vision_layers(vision_tower: Any) -> Any:
        candidates = (
            getattr(
                getattr(getattr(vision_tower, "vision_model", None), "encoder", None),
                "layers",
                None,
            ),
            getattr(getattr(vision_tower, "encoder", None), "layers", None),
        )
        for layers in candidates:
            if layers is not None and len(layers) > 0:
                return layers
        raise RuntimeError("could not locate vision encoder layers")

    @contextmanager
    def _apply_intervention(
        self,
        intervention: HiddenIntervention,
        visual_positions: torch.Tensor,
    ) -> Iterator[None]:
        indices = intervention.patch_indices.detach().long().flatten()
        if (
            indices.numel() == 0
            or indices.unique().numel() != indices.numel()
            or int(indices.min()) < 0
            or int(indices.max()) >= visual_positions.numel()
        ):
            raise ValueError("intervention patch indices are empty, repeated, or out of range")
        replacement = intervention.replacement.detach().flatten()
        if replacement.ndim != 1 or replacement.numel() == 0:
            raise ValueError("intervention replacement must be a non-empty vector")

        stage = intervention.stage
        vision_layers = int(self.model.config.vision_config.num_hidden_layers)
        language_layers = int(self.model.config.text_config.num_hidden_layers)
        feature_layer = self.model.config.vision_feature_layer
        if not isinstance(feature_layer, int):
            raise RuntimeError("interventions require one classic-LLaVA vision feature layer")
        plan = resolve_stage_plan(
            vision_num_hidden_layers=vision_layers,
            llm_num_hidden_layers=language_layers,
            vision_feature_layer=feature_layer,
        )
        if stage not in plan:
            raise ValueError(f"unknown intervention stage: {stage}")

        handle: Any
        if stage.startswith("llm."):
            hidden_index = plan[stage]
            assert hidden_index is not None
            if hidden_index >= language_layers:
                raise ValueError(
                    f"{stage} is after the last causal decoder transition; choose its predecessor"
                )
            language_model = _model_component(self.model, "language_model")
            layer = self._decoder_layers(language_model)[hidden_index]
            sequence_positions = visual_positions.index_select(0, indices).long()

            def language_pre_hook(
                _: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
            ) -> tuple[tuple[Any, ...], dict[str, Any]]:
                if not args or not isinstance(args[0], torch.Tensor):
                    raise RuntimeError("decoder layer did not receive positional hidden states")
                if args[0].shape[1] <= int(sequence_positions.max()):
                    # Cached autoregressive continuation contains only the new
                    # token; the visual states were already replaced in prefill.
                    return args, kwargs
                hidden = args[0].clone()
                vector = replacement.to(device=hidden.device, dtype=hidden.dtype)
                if vector.numel() != hidden.shape[-1]:
                    raise ValueError("replacement dimension differs from LLM hidden size")
                positions = sequence_positions.to(hidden.device)
                hidden[:, positions, :] = vector.view(1, 1, -1)
                return (hidden, *args[1:]), kwargs

            handle = layer.register_forward_pre_hook(language_pre_hook, with_kwargs=True)
        elif stage == "projector.output":
            projector = _model_component(self.model, "multi_modal_projector")

            def projector_hook(_: Any, __: tuple[Any, ...], output: Any) -> torch.Tensor:
                if not isinstance(output, torch.Tensor) or output.ndim != 3:
                    raise RuntimeError("projector intervention expected [batch, patches, hidden]")
                modified = output.clone()
                vector = replacement.to(device=modified.device, dtype=modified.dtype)
                if vector.numel() != modified.shape[-1]:
                    raise ValueError("replacement dimension differs from projector output")
                modified[:, indices.to(modified.device), :] = vector.view(1, 1, -1)
                return modified

            handle = projector.register_forward_hook(projector_hook)
        elif stage.startswith("vision."):
            hidden_index = plan[stage]
            assert hidden_index is not None
            selected_feature_index = feature_layer % (vision_layers + 1)
            if hidden_index > selected_feature_index:
                raise ValueError(
                    f"{stage} does not feed the frozen projector source hidden state"
                )
            if hidden_index <= 0:
                raise ValueError("vision embedding input is not an intervention stage")
            vision_tower = _model_component(self.model, "vision_tower")
            layer = self._vision_layers(vision_tower)[hidden_index - 1]
            vision_positions = indices + 1  # classic CLIP includes CLS at position zero

            def replace_vision_tensor(hidden: torch.Tensor) -> torch.Tensor:
                modified = hidden.clone()
                vector = replacement.to(device=modified.device, dtype=modified.dtype)
                if vector.numel() != modified.shape[-1]:
                    raise ValueError("replacement dimension differs from vision hidden size")
                modified[:, vision_positions.to(modified.device), :] = vector.view(1, 1, -1)
                return modified

            def vision_hook(_: Any, __: tuple[Any, ...], output: Any) -> Any:
                if isinstance(output, torch.Tensor):
                    return replace_vision_tensor(output)
                if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
                    return (replace_vision_tensor(output[0]), *output[1:])
                raise RuntimeError("vision layer returned an unsupported output structure")

            handle = layer.register_forward_hook(vision_hook)
        else:
            raise ValueError(f"unsupported intervention stage: {stage}")
        try:
            yield
        finally:
            handle.remove()

    def _sequence_log_likelihood(
        self,
        inputs: dict[str, torch.Tensor],
        token_ids: tuple[int, ...],
        prompt_logits: torch.Tensor,
    ) -> float:
        first = F.log_softmax(prompt_logits.float(), dim=-1)[token_ids[0]]
        if len(token_ids) == 1:
            return float(first.detach().cpu())
        expanded = self._append_tokens(inputs, token_ids)
        with torch.inference_mode():
            outputs = self.model(
                **expanded,
                output_attentions=False,
                use_cache=False,
                return_dict=True,
            )
        prompt_length = int(inputs["input_ids"].shape[1])
        continuation_logits = outputs.logits[
            0, prompt_length : prompt_length + len(token_ids) - 1
        ]
        continuation_targets = torch.tensor(
            token_ids[1:], device=continuation_logits.device, dtype=torch.long
        )
        continuation = F.log_softmax(
            continuation_logits.float(), dim=-1
        ).gather(1, continuation_targets[:, None]).sum()
        return float((first + continuation).detach().cpu())

    def evaluate(
        self,
        image: Any | None,
        prompt_text: str,
        *,
        include_attention: bool,
        generate: bool,
        max_new_tokens: int = 3,
        intervention: HiddenIntervention | None = None,
    ) -> FrozenDecisionOutput:
        """Measure sequence-level answer evidence and optional generated output."""

        if not prompt_text.strip():
            raise ValueError("prompt text cannot be empty")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if image is None and include_attention:
            raise ValueError("visual attention is undefined for a prompt-only control")
        inputs, visual_positions, query_position = self._prepare(image, prompt_text)
        if intervention is not None and visual_positions is None:
            raise ValueError("hidden interventions require an image")
        context = (
            self._apply_intervention(intervention, visual_positions)
            if intervention is not None and visual_positions is not None
            else nullcontext()
        )
        attention_scores: torch.Tensor | None = None
        entropy: float | None = None
        effective: float | None = None
        generated_text: str | None = None
        with context:
            with torch.inference_mode():
                outputs = self.model(
                    **inputs,
                    output_attentions=include_attention,
                    use_cache=False,
                    return_dict=True,
                )
            prompt_logits = outputs.logits[0, -1]
            positive = self._sequence_log_likelihood(
                inputs, self.positive_token_ids, prompt_logits
            )
            negative = self._sequence_log_likelihood(
                inputs, self.negative_token_ids, prompt_logits
            )
            if include_attention:
                attentions = getattr(outputs, "attentions", None)
                if not attentions or any(value is None for value in attentions):
                    raise RuntimeError("LLaVA did not return eager attention tensors")
                assert visual_positions is not None and query_position is not None
                layer = attentions[self.attention_layer_offset]
                positions = visual_positions.to(layer.device)
                attention_scores = (
                    layer[0, :, query_position, positions]
                    .float()
                    .mean(dim=0)
                    .detach()
                    .cpu()
                )
                entropy, effective = normalized_entropy(attention_scores)
            if generate:
                with torch.inference_mode():
                    generated = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        use_cache=True,
                    )
                prompt_length = int(inputs["input_ids"].shape[1])
                answer_ids = generated[0, prompt_length:].detach().cpu().tolist()
                generated_text = self.processor.tokenizer.decode(
                    answer_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                ).strip()

        del outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return FrozenDecisionOutput(
            positive_log_likelihood=positive,
            negative_log_likelihood=negative,
            answer_margin=positive - negative,
            positive_token_ids=self.positive_token_ids,
            negative_token_ids=self.negative_token_ids,
            generated_text=generated_text,
            attention_scores=attention_scores,
            attention_entropy=entropy,
            attention_effective_tokens=effective,
        )
