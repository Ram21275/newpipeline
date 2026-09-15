"""Qwen2.5-VL adapter for exact binary answer likelihood replication.

This adapter intentionally starts with the utilization experiment.  Stage
extraction and hidden-state interventions require separate architecture-specific
validation because Qwen uses dynamic visual tokenization and a merger.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .hf_llava import _module_device, validate_bitsandbytes_4bit_runtime
from .hf_utilization import FrozenDecisionOutput, normalized_entropy


def qwen_visual_positions_and_query(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    image_token_id: int,
) -> tuple[torch.Tensor, int]:
    """Locate expanded Qwen image-pad tokens and the final prompt token."""

    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or attention_mask.shape != input_ids.shape:
        raise ValueError("Qwen replication requires one aligned input sequence")
    active = attention_mask[0].bool()
    image_mask = input_ids[0].eq(image_token_id) & active
    visual = image_mask.nonzero(as_tuple=False).flatten()
    text = (active & ~image_mask).nonzero(as_tuple=False).flatten()
    if visual.numel() == 0:
        raise RuntimeError("Qwen processor did not expand an image-pad token sequence")
    if text.numel() == 0:
        raise RuntimeError("Qwen prompt contains no text query token")
    return visual, int(text[-1])


class HfQwenDecisionRunner:
    """Measure exact yes/no sequence likelihoods in frozen Qwen2.5-VL."""

    def __init__(
        self,
        model: Any,
        processor: Any,
        *,
        positive_answer: str = "yes",
        negative_answer: str = "no",
        attention_layer_offset: int = -2,
    ) -> None:
        if attention_layer_offset >= 0:
            raise ValueError("attention_layer_offset must be negative")
        self.model = model.eval()
        self.processor = processor
        self.attention_layer_offset = attention_layer_offset
        self.input_device = _module_device(model)
        config_dtype = getattr(model.config, "torch_dtype", None)
        self.compute_dtype = config_dtype if isinstance(config_dtype, torch.dtype) else torch.float16
        token_id = getattr(model.config, "image_token_id", None)
        if token_id is None:
            token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        if token_id is None or int(token_id) < 0:
            raise RuntimeError("could not resolve Qwen image token ID")
        self.image_token_id = int(token_id)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.set_answer_pair(positive_answer, negative_answer)

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
    ) -> "HfQwenDecisionRunner":
        try:
            import transformers  # type: ignore[import-not-found]
            from transformers import (  # type: ignore[import-not-found]
                AutoProcessor,
                BitsAndBytesConfig,
                Qwen2_5_VLForConditionalGeneration,
            )
        except ImportError as error:
            raise RuntimeError(
                "Install requirements-qwen-kaggle.txt before loading Qwen2.5-VL"
            ) from error
        version = tuple(int(part) for part in transformers.__version__.split(".")[:2])
        if version < (4, 50):
            raise RuntimeError("Qwen2.5-VL replication requires transformers>=4.50")
        if not torch.cuda.is_available():
            raise RuntimeError("Qwen2.5-VL replication requires a Kaggle GPU")
        load_kwargs: dict[str, Any] = {
            "revision": revision,
            "device_map": "auto",
            "low_cpu_mem_usage": True,
            "attn_implementation": "eager",
            "torch_dtype": torch.float16,
        }
        if quantization == "4bit":
            validate_bitsandbytes_4bit_runtime()
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        elif quantization != "none":
            raise ValueError("quantization must be '4bit' or 'none'")
        processor = AutoProcessor.from_pretrained(model_name, revision=revision)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, **load_kwargs)
        return cls(
            model,
            processor,
            positive_answer=positive_answer,
            negative_answer=negative_answer,
            attention_layer_offset=attention_layer_offset,
        )

    @property
    def resolved_revision(self) -> str:
        return str(getattr(self.model.config, "_commit_hash", None) or "unknown")

    def _answer_token_ids(self, answer: str) -> tuple[int, ...]:
        tokenizer = self.processor.tokenizer
        values = tuple(int(value) for value in tokenizer.encode(answer, add_special_tokens=False))
        special = {int(value) for value in tokenizer.all_special_ids}
        if not values or any(value in special for value in values):
            raise RuntimeError(f"invalid Qwen answer tokenization for {answer!r}")
        decoded = tokenizer.decode(values, skip_special_tokens=True,
                                   clean_up_tokenization_spaces=False)
        if decoded.strip().casefold() != answer.casefold():
            raise RuntimeError(f"Qwen answer tokenization does not round-trip: {answer!r}")
        return values

    def set_answer_pair(self, positive_answer: str, negative_answer: str) -> None:
        """Change the scored lexical pair without reloading the frozen model."""

        positive = positive_answer.strip()
        negative = negative_answer.strip()
        if not positive or not negative:
            raise ValueError("binary answer strings cannot be empty")
        positive_ids = self._answer_token_ids(positive)
        negative_ids = self._answer_token_ids(negative)
        if positive_ids == negative_ids:
            raise ValueError("positive and negative answers tokenize identically")
        self.positive_answer = positive
        self.negative_answer = negative
        self.positive_token_ids = positive_ids
        self.negative_token_ids = negative_ids

    def _prepare(
        self, image: Any | None, prompt_text: str
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor | None, int | None]:
        content = ([{"type": "image"}, {"type": "text", "text": prompt_text}]
                   if image is not None else [{"type": "text", "text": prompt_text}])
        rendered = self.processor.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True,
            tokenize=False,
        )
        kwargs: dict[str, Any] = {"text": [rendered], "return_tensors": "pt", "padding": True}
        if image is not None:
            kwargs["images"] = [image]
        inputs = self.processor(**kwargs)
        visual: torch.Tensor | None = None
        query: int | None = None
        if image is not None:
            visual, query = qwen_visual_positions_and_query(
                inputs["input_ids"], inputs["attention_mask"], self.image_token_id
            )
        moved: dict[str, torch.Tensor] = {}
        for name, value in inputs.items():
            if not isinstance(value, torch.Tensor):
                continue
            if value.is_floating_point():
                value = value.to(device=self.input_device, dtype=self.compute_dtype)
            else:
                value = value.to(self.input_device)
            moved[name] = value
        return moved, visual, query

    @staticmethod
    def _append_tokens(inputs: dict[str, torch.Tensor], token_ids: tuple[int, ...]) -> dict[str, torch.Tensor]:
        result = dict(inputs)
        ids = torch.tensor(token_ids, dtype=inputs["input_ids"].dtype,
                           device=inputs["input_ids"].device).view(1, -1)
        result["input_ids"] = torch.cat((inputs["input_ids"], ids), dim=1)
        mask = torch.ones((1, len(token_ids)), dtype=inputs["attention_mask"].dtype,
                          device=inputs["attention_mask"].device)
        result["attention_mask"] = torch.cat((inputs["attention_mask"], mask), dim=1)
        result.pop("position_ids", None)
        result.pop("rope_deltas", None)
        return result

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
                **expanded, output_attentions=False, use_cache=False, return_dict=True
            )
        prompt_length = int(inputs["input_ids"].shape[1])
        logits = outputs.logits[0, prompt_length:prompt_length + len(token_ids) - 1]
        targets = torch.tensor(token_ids[1:], device=logits.device, dtype=torch.long)
        continuation = F.log_softmax(logits.float(), dim=-1).gather(1, targets[:, None]).sum()
        return float((first + continuation).detach().cpu())

    def evaluate(
        self,
        image: Any | None,
        prompt_text: str,
        *,
        include_attention: bool,
        generate: bool,
        max_new_tokens: int = 3,
    ) -> FrozenDecisionOutput:
        if not prompt_text.strip():
            raise ValueError("prompt text cannot be empty")
        if image is None and include_attention:
            raise ValueError("visual attention is undefined for prompt-only input")
        inputs, visual, query = self._prepare(image, prompt_text)
        with torch.inference_mode():
            outputs = self.model(
                **inputs,
                output_attentions=include_attention,
                use_cache=False,
                return_dict=True,
            )
        logits = outputs.logits[0, -1]
        positive = self._sequence_log_likelihood(inputs, self.positive_token_ids, logits)
        negative = self._sequence_log_likelihood(inputs, self.negative_token_ids, logits)
        scores = None
        entropy = None
        effective = None
        if include_attention:
            attentions = getattr(outputs, "attentions", None)
            if not attentions or any(item is None for item in attentions):
                raise RuntimeError("Qwen eager attention tensors are unavailable")
            assert visual is not None and query is not None
            layer = attentions[self.attention_layer_offset]
            positions = visual.to(layer.device)
            scores = layer[0, :, query, positions].float().mean(dim=0).detach().cpu()
            entropy, effective = normalized_entropy(scores)
        generated_text = None
        if generate:
            with torch.inference_mode():
                generated = self.model.generate(
                    **inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True
                )
            prompt_length = int(inputs["input_ids"].shape[1])
            answer_ids = generated[0, prompt_length:].detach().cpu().tolist()
            generated_text = self.processor.tokenizer.decode(
                answer_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            ).strip()
        del outputs
        torch.cuda.empty_cache()
        return FrozenDecisionOutput(
            positive_log_likelihood=positive,
            negative_log_likelihood=negative,
            answer_margin=positive - negative,
            positive_token_ids=self.positive_token_ids,
            negative_token_ids=self.negative_token_ids,
            generated_text=generated_text,
            attention_scores=scores,
            attention_entropy=entropy,
            attention_effective_tokens=effective,
        )
