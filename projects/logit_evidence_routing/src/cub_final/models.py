"""Sequential frozen-model adapters for the two required checkpoints."""

from __future__ import annotations

import gc
import math
import time
from dataclasses import dataclass
from typing import Any, Sequence

from .scoring import assert_final_logit_agreement


LLAVA_CHECKPOINT = "llava-hf/llava-1.5-7b-hf"
LLAVA_REVISION = "b234b804b114d9e37bb655e11cbbb5f5e971b7a9"
QWEN3_CHECKPOINT = "Qwen/Qwen3-VL-2B-Instruct"
QWEN3_REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"


@dataclass(frozen=True)
class RuntimeChoice:
    dtype_name: str
    quantization: str
    device_map: str


def choose_runtime(*, quantization: str = "none") -> RuntimeChoice:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("the real VLM smoke/final stages require a CUDA Kaggle accelerator")
    if quantization not in {"none", "4bit"}:
        raise ValueError("quantization must be none or 4bit")
    dtype_name = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
    return RuntimeChoice(dtype_name=dtype_name, quantization=quantization, device_map="auto")


def _module_device(module: Any) -> Any:
    for parameter in module.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    raise RuntimeError("could not resolve a non-meta model device")


def _resolve_lm_components(model: Any) -> tuple[Any, Any]:
    language = getattr(model, "language_model", None)
    if language is None:
        language = getattr(getattr(model, "model", None), "language_model", None)
    if language is None:
        language = model
    getter = getattr(language, "get_output_embeddings", None)
    head = getter() if getter is not None else None
    if head is None:
        head = getattr(language, "lm_head", None) or getattr(model, "lm_head", None)
    if head is None:
        raise RuntimeError("could not locate the language-model output head")
    candidates = (
        ("model", "norm"),
        ("model", "final_layernorm"),
        ("transformer", "ln_f"),
        ("model", "decoder", "final_layer_norm"),
        ("norm",),
    )
    for path in candidates:
        value = language
        for component in path:
            value = getattr(value, component, None)
            if value is None:
                break
        if value is not None:
            return value, head
    raise RuntimeError("could not locate the language-model final norm")


def _answer_ids(tokenizer: Any, answer: str) -> tuple[int, ...]:
    values = tuple(int(value) for value in tokenizer.encode(answer, add_special_tokens=False))
    special = {int(value) for value in tokenizer.all_special_ids}
    if not values or any(value in special for value in values):
        raise RuntimeError(f"invalid answer tokenization for {answer!r}")
    decoded = tokenizer.decode(
        values, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    if decoded.strip().casefold() != answer.strip().casefold():
        raise RuntimeError(f"answer tokenization does not round-trip: {answer!r}")
    return values


def _image_token_id(model: Any, processor: Any) -> int:
    for name in ("image_token_index", "image_token_id"):
        value = getattr(model.config, name, None)
        if value is not None:
            return int(value)
    for token in ("<|image_pad|>", "<image>"):
        value = processor.tokenizer.convert_tokens_to_ids(token)
        if value is not None and int(value) >= 0:
            return int(value)
    raise RuntimeError("could not resolve the checkpoint's image token ID")


class FrozenVlmRunner:
    """Shared candidate-scoring and layer-readout implementation."""

    def __init__(self, model: Any, processor: Any, *, architecture: str) -> None:
        import torch

        self.model = model.eval()
        self.processor = processor
        self.architecture = architecture
        self.input_device = _module_device(model)
        self.image_token_id = _image_token_id(model, processor)
        self.final_norm, self.lm_head = _resolve_lm_components(model)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        config_dtype = getattr(model.config, "torch_dtype", None)
        self.compute_dtype = config_dtype if isinstance(config_dtype, torch.dtype) else torch.float16

    @property
    def resolved_revision(self) -> str:
        return str(getattr(self.model.config, "_commit_hash", None) or "unknown")

    def architecture_audit(self) -> dict[str, Any]:
        config = self.model.config
        vision = getattr(config, "vision_config", None)
        return {
            "architecture": self.architecture,
            "checkpoint_type": type(self.model).__name__,
            "resolved_revision": self.resolved_revision,
            "image_token_id": self.image_token_id,
            "vision_patch_size": getattr(vision, "patch_size", None),
            "vision_spatial_merge_size": getattr(vision, "spatial_merge_size", None),
            "vision_out_hidden_size": getattr(vision, "out_hidden_size", None),
            "deepstack_visual_indexes": getattr(config, "deepstack_visual_indexes", None),
            "vision_feature_layers": getattr(config, "vision_feature_layer", None),
            "vision_feature_select_strategy": getattr(config, "vision_feature_select_strategy", None),
        }

    def _render(self, prompt: str, *, has_image: bool) -> str:
        if self.architecture == "qwen3":
            content = []
            if has_image:
                content.append({"type": "image"})
            content.append({"type": "text", "text": prompt})
            return self.processor.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False,
                add_generation_prompt=True,
            )
        template = getattr(self.processor, "apply_chat_template", None)
        if template is not None:
            content = []
            if has_image:
                content.append({"type": "image"})
            content.append({"type": "text", "text": prompt})
            try:
                return template(
                    [{"role": "user", "content": content}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except (TypeError, ValueError):
                pass
        return f"USER: <image>\n{prompt}\nASSISTANT:" if has_image else f"USER: {prompt}\nASSISTANT:"

    def prepare(self, image: Any | None, prompt: str) -> tuple[dict[str, Any], dict[str, Any]]:
        import torch

        rendered = self._render(prompt, has_image=image is not None)
        kwargs: dict[str, Any] = {"text": [rendered], "return_tensors": "pt", "padding": True}
        if image is not None:
            kwargs["images"] = [image]
        inputs = self.processor(**kwargs)
        active = inputs["attention_mask"][0].bool()
        image_positions = (
            inputs["input_ids"][0].eq(self.image_token_id) & active
        ).nonzero(as_tuple=False).flatten()
        text_positions = (
            ~inputs["input_ids"][0].eq(self.image_token_id) & active
        ).nonzero(as_tuple=False).flatten()
        if image is not None and image_positions.numel() == 0:
            raise RuntimeError("processor produced no identifiable visual token positions")
        if text_positions.numel() == 0:
            raise RuntimeError("processor produced no prompt positions")
        meta: dict[str, Any] = {
            "rendered_prompt": rendered,
            "visual_token_count": int(image_positions.numel()),
            "visual_positions": image_positions,
            "query_position": int(text_positions[-1]),
        }
        if "image_grid_thw" in inputs:
            grid = inputs["image_grid_thw"][0].detach().cpu().tolist()
            merge = int(
                getattr(getattr(self.model.config, "vision_config", None), "spatial_merge_size", 1)
                or 1
            )
            meta["image_grid_thw"] = grid
            meta["merged_grid_size"] = (int(grid[2]) // merge, int(grid[1]) // merge)
        else:
            side = math.isqrt(int(image_positions.numel()))
            if side * side == int(image_positions.numel()):
                meta["merged_grid_size"] = (side, side)
        moved: dict[str, Any] = {}
        for key, value in inputs.items():
            if not isinstance(value, torch.Tensor):
                continue
            if value.is_floating_point():
                moved[key] = value.to(self.input_device, dtype=self.compute_dtype)
            else:
                moved[key] = value.to(self.input_device)
        return moved, meta

    @staticmethod
    def _append(inputs: dict[str, Any], ids: Sequence[int]) -> dict[str, Any]:
        import torch

        output = dict(inputs)
        values = torch.tensor(ids, device=inputs["input_ids"].device, dtype=inputs["input_ids"].dtype)[None]
        output["input_ids"] = torch.cat((inputs["input_ids"], values), dim=1)
        ones = torch.ones(
            (1, len(ids)), device=inputs["attention_mask"].device, dtype=inputs["attention_mask"].dtype
        )
        output["attention_mask"] = torch.cat((inputs["attention_mask"], ones), dim=1)
        for name in ("position_ids", "rope_deltas", "cache_position"):
            output.pop(name, None)
        return output

    def _sequence_logp(self, inputs: dict[str, Any], ids: Sequence[int], first_logits: Any) -> float:
        import torch
        import torch.nn.functional as F

        first = F.log_softmax(first_logits.float(), dim=-1)[int(ids[0])]
        if len(ids) == 1:
            return float(first.detach().cpu())
        expanded = self._append(inputs, ids)
        with torch.inference_mode():
            outputs = self.model(**expanded, use_cache=False, return_dict=True)
        start = int(inputs["input_ids"].shape[1])
        logits = outputs.logits[0, start : start + len(ids) - 1]
        targets = torch.tensor(ids[1:], device=logits.device, dtype=torch.long)
        rest = F.log_softmax(logits.float(), dim=-1).gather(1, targets[:, None]).sum()
        return float((first + rest).detach().cpu())

    def evaluate(
        self,
        image: Any,
        prompt: str,
        *,
        positive_answer: str = "yes",
        negative_answer: str = "no",
        capture_layers: bool = True,
        capture_attention: bool = True,
        generate: bool = True,
        max_new_tokens: int = 8,
    ) -> dict[str, Any]:
        import torch

        positive_ids = _answer_ids(self.processor.tokenizer, positive_answer)
        negative_ids = _answer_ids(self.processor.tokenizer, negative_answer)
        inputs, meta = self.prepare(image, prompt)
        started = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            outputs = self.model(
                **inputs,
                output_hidden_states=capture_layers,
                output_attentions=capture_attention,
                use_cache=False,
                return_dict=True,
            )
        elapsed = time.perf_counter() - started
        mature = outputs.logits[0, -1]
        if not bool(torch.isfinite(mature).all()):
            raise RuntimeError("non-finite model logits; do not continue with this precision")
        positive_logp = self._sequence_logp(inputs, positive_ids, mature)
        negative_logp = self._sequence_logp(inputs, negative_ids, mature)
        result: dict[str, Any] = {
            "architecture": self.architecture,
            "resolved_revision": self.resolved_revision,
            "positive_answer": positive_answer,
            "negative_answer": negative_answer,
            "positive_token_ids": list(positive_ids),
            "negative_token_ids": list(negative_ids),
            "positive_log_probability": positive_logp,
            "negative_log_probability": negative_logp,
            "raw_margin": positive_logp - negative_logp,
            "runtime_seconds": elapsed,
            "max_gpu_memory_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
            **{key: value for key, value in meta.items() if key not in {"visual_positions"}},
        }
        if capture_attention:
            attentions = getattr(outputs, "attentions", None)
            if attentions and attentions[-1] is not None:
                layer = attentions[-1]
                visual = meta["visual_positions"].to(layer.device)
                query = int(meta["query_position"])
                scores = layer[0, :, query, visual].float().mean(dim=0)
                result["decoder_attention_scores"] = scores.detach().cpu().tolist()
            else:
                result["decoder_attention_scores"] = None
                result["attention_failure"] = "checkpoint/runtime did not return eager attentions"
        if capture_layers:
            hidden = outputs.hidden_states
            projected: list[list[float]] = []
            # Intermediate states need final norm; the last state is validated
            # against model.logits and is never normalised twice.
            for state in hidden[:-1]:
                selected = state[0, int(meta["query_position"])]
                logits = self.lm_head(self.final_norm(selected.to(next(self.lm_head.parameters()).device)))
                projected.append([float(logits[positive_ids[0]].detach().cpu()), float(logits[negative_ids[0]].detach().cpu())])
            last = hidden[-1][0, int(meta["query_position"])].to(next(self.lm_head.parameters()).device)
            direct = self.lm_head(last)
            dtype_tolerance = 5e-2 if mature.dtype == torch.float16 else 2e-2
            agreement = assert_final_logit_agreement(
                direct, mature, atol=dtype_tolerance, rtol=dtype_tolerance
            )
            projected.append([float(mature[positive_ids[0]].detach().cpu()), float(mature[negative_ids[0]].detach().cpu())])
            result["answer_first_token_layer_logits"] = projected
            result["final_logit_agreement"] = agreement
            result["logit_lens_scope"] = "first_answer_token; full candidates scored separately"
        if generate:
            with torch.inference_mode():
                generated = self.model.generate(
                    **inputs, do_sample=False, max_new_tokens=max_new_tokens, use_cache=True
                )
            prompt_length = int(inputs["input_ids"].shape[1])
            result["generated_text"] = self.processor.tokenizer.decode(
                generated[0, prompt_length:].detach().cpu().tolist(),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()
        del outputs
        return result


def load_runner(
    architecture: str,
    *,
    quantization: str = "none",
    local_snapshot: str | None = None,
) -> FrozenVlmRunner:
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig

    runtime = choose_runtime(quantization=quantization)
    dtype = torch.bfloat16 if runtime.dtype_name == "bfloat16" else torch.float16
    load_kwargs: dict[str, Any] = {
        "device_map": runtime.device_map,
        "low_cpu_mem_usage": True,
        "torch_dtype": dtype,
        "attn_implementation": "eager",
    }
    if quantization == "4bit":
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    if architecture == "llava":
        from transformers import LlavaForConditionalGeneration

        checkpoint, revision = LLAVA_CHECKPOINT, LLAVA_REVISION
        source = local_snapshot or checkpoint
        processor = AutoProcessor.from_pretrained(source, revision=None if local_snapshot else revision)
        model = LlavaForConditionalGeneration.from_pretrained(
            source, revision=None if local_snapshot else revision, **load_kwargs
        )
    elif architecture == "qwen3":
        try:
            from transformers import Qwen3VLForConditionalGeneration
        except ImportError as error:
            raise RuntimeError(
                "installed Transformers does not include Qwen3-VL; install requirements-cub-final-kaggle.txt"
            ) from error
        checkpoint, revision = QWEN3_CHECKPOINT, QWEN3_REVISION
        source = local_snapshot or checkpoint
        processor = AutoProcessor.from_pretrained(source, revision=None if local_snapshot else revision)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            source, revision=None if local_snapshot else revision, **load_kwargs
        )
    else:
        raise ValueError("architecture must be llava or qwen3")
    runner = FrozenVlmRunner(model, processor, architecture=architecture)
    expected = revision if local_snapshot is None else None
    if expected is not None and runner.resolved_revision not in {expected, "unknown"}:
        raise RuntimeError(
            f"resolved model revision {runner.resolved_revision} differs from locked {expected}"
        )
    return runner


def unload_runner(runner: FrozenVlmRunner) -> None:
    import torch

    del runner.model
    del runner
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

