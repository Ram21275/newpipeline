"""Hugging Face LLaVA adapter used by the Kaggle Phase 01 extraction job."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .localization import attention_rollout, final_cls_attention
from .scoring import logits_to_evidence, logits_to_token_mass


KAGGLE_BITSANDBYTES_VERSION = "0.50.2"


@dataclass(frozen=True)
class PatchEvidence:
    """One image's aligned hidden states, attention, and vocabulary evidence."""

    hidden_states: torch.Tensor
    attention_scores: torch.Tensor
    localization_scores: dict[str, torch.Tensor]
    evidence_scores: dict[str, torch.Tensor]
    top_token_ids: torch.Tensor
    grid_size: tuple[int, int]
    processed_image: torch.Tensor


def visual_positions_and_query(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    image_token_id: int,
) -> tuple[torch.Tensor, int]:
    """Locate expanded image placeholders and the final non-image prompt token."""

    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("Phase 01 extraction requires input_ids with batch size one")
    if attention_mask.shape != input_ids.shape:
        raise ValueError("attention_mask must match input_ids")
    image_mask = input_ids[0].eq(image_token_id) & attention_mask[0].bool()
    visual_positions = image_mask.nonzero(as_tuple=False).flatten()
    text_positions = (attention_mask[0].bool() & ~image_mask).nonzero(
        as_tuple=False
    ).flatten()
    if visual_positions.numel() <= 1:
        raise RuntimeError(
            "The processor did not expand <image> into visual-token placeholders. "
            "Use the pinned transformers range and the classic LLaVA-1.5 model."
        )
    if text_positions.numel() == 0:
        raise RuntimeError("No text query token was found after image processing")
    return visual_positions, int(text_positions[-1])


def square_grid(patch_count: int) -> tuple[int, int]:
    side = math.isqrt(patch_count)
    if side * side != patch_count:
        raise RuntimeError(
            f"Expected a square classic-LLaVA patch grid, found {patch_count} tokens. "
            "Do not use an any-resolution/OneVision checkpoint for this pilot."
        )
    return side, side


def _module_device(module: nn.Module) -> torch.device:
    for parameter in module.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    raise RuntimeError(f"Could not determine device for {type(module).__name__}")


def _module_dtype(module: nn.Module) -> torch.dtype:
    for parameter in module.parameters():
        if parameter.device.type != "meta" and parameter.is_floating_point():
            return parameter.dtype
    return torch.float32


def fixed_concept_token_ids(
    tokenizer: Any, concepts: tuple[str, ...]
) -> tuple[int, ...]:
    """Tokenize a fixed, image-independent concept vocabulary."""

    token_ids: set[int] = set()
    for raw_concept in concepts:
        concept = raw_concept.strip()
        if not concept:
            raise ValueError("fixed concepts cannot be empty")
        encoded = tokenizer.encode(f" {concept}", add_special_tokens=False)
        if not encoded:
            raise ValueError(f"fixed concept produced no tokens: {concept!r}")
        token_ids.update(int(token_id) for token_id in encoded)
    return tuple(sorted(token_ids))


def _resolve_language_projection(model: nn.Module) -> tuple[nn.Module, nn.Module]:
    language_model = getattr(model, "language_model", model)
    get_embeddings = getattr(language_model, "get_output_embeddings", None)
    lm_head = get_embeddings() if get_embeddings is not None else None
    if lm_head is None:
        lm_head = getattr(language_model, "lm_head", None)
    if lm_head is None:
        raise RuntimeError("Could not locate the frozen language-model output head")

    norm_candidates = (
        ("model", "norm"),
        ("model", "final_layernorm"),
        ("transformer", "ln_f"),
        ("model", "decoder", "final_layer_norm"),
    )
    final_norm: nn.Module | None = None
    for path in norm_candidates:
        current: Any = language_model
        for component in path:
            current = getattr(current, component, None)
            if current is None:
                break
        if isinstance(current, nn.Module):
            final_norm = current
            break
    if final_norm is None:
        raise RuntimeError("Could not locate the frozen language-model final norm")
    return final_norm, lm_head


def validate_bitsandbytes_4bit_runtime() -> str:
    """Run one tiny NF4 operation before Hugging Face downloads model weights."""

    try:
        import bitsandbytes as bnb  # type: ignore[import-not-found]
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "4-bit loading requires a working bitsandbytes installation. "
            f"Install the Kaggle requirements (bitsandbytes=={KAGGLE_BITSANDBYTES_VERSION}) "
            "and retry."
        ) from error

    installed_version = str(getattr(bnb, "__version__", "unknown"))
    cuda_version = str(torch.version.cuda or "unknown")
    try:
        sample = torch.linspace(
            -1,
            1,
            steps=128,
            device="cuda",
            dtype=torch.float16,
        ).reshape(2, 64)
        quantized, quantization_state = bnb.functional.quantize_4bit(
            sample,
            quant_type="nf4",
        )
        restored = bnb.functional.dequantize_4bit(
            quantized,
            quantization_state,
        )
        if not bool(torch.isfinite(restored).all()):
            raise RuntimeError("NF4 preflight produced non-finite values")
    except Exception as error:
        raise RuntimeError(
            "bitsandbytes failed the CUDA NF4 preflight before model download "
            f"(installed={installed_version}, PyTorch CUDA={cuda_version}). "
            f"Reinstall bitsandbytes=={KAGGLE_BITSANDBYTES_VERSION} from "
            "requirements-kaggle.txt, then rerun extraction."
        ) from error
    finally:
        if "sample" in locals():
            del sample
        torch.cuda.empty_cache()
    return installed_version


class HfLlavaPatchExtractor:
    """Extract one late-layer patch signal from classic LLaVA-1.5."""

    def __init__(
        self,
        model: nn.Module,
        processor: Any,
        *,
        layer_offset: int = -2,
        projection_chunk_size: int = 64,
        fixed_concepts: tuple[str, ...] = (),
    ) -> None:
        if layer_offset >= -1:
            raise ValueError(
                "layer_offset must be -2 or earlier: hidden_states[-1] is already "
                "final-normalized and must not be normalized a second time"
            )
        if projection_chunk_size <= 0:
            raise ValueError("projection_chunk_size must be positive")
        self.model = model.eval()
        self.processor = processor
        self.layer_offset = layer_offset
        self.projection_chunk_size = projection_chunk_size
        self.fixed_concepts = tuple(fixed_concepts)
        self.concept_token_ids = fixed_concept_token_ids(
            processor.tokenizer, self.fixed_concepts
        )
        self.final_norm, self.lm_head = _resolve_language_projection(model)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.image_token_id = int(model.config.image_token_index)
        self.input_device = _module_device(model)
        config_dtype = getattr(model.config, "torch_dtype", None)
        self.compute_dtype = (
            config_dtype
            if isinstance(config_dtype, torch.dtype)
            else (torch.float16 if self.input_device.type == "cuda" else torch.float32)
        )

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        *,
        revision: str = "main",
        quantization: str = "4bit",
        layer_offset: int = -2,
        projection_chunk_size: int = 64,
        fixed_concepts: tuple[str, ...] = (),
    ) -> "HfLlavaPatchExtractor":
        """Load the public pilot checkpoint; intended to run on Kaggle GPU."""

        try:
            from transformers import (  # type: ignore[import-not-found]
                AutoProcessor,
                BitsAndBytesConfig,
                LlavaForConditionalGeneration,
            )
        except ImportError as error:
            raise RuntimeError(
                "Install requirements-kaggle.txt before loading LLaVA"
            ) from error

        if not torch.cuda.is_available():
            raise RuntimeError(
                "The real Phase 01 extractor requires a Kaggle GPU accelerator"
            )
        load_kwargs: dict[str, Any] = {
            "revision": revision,
            "device_map": "auto",
            "low_cpu_mem_usage": True,
            "attn_implementation": "eager",
            "torch_dtype": torch.float16,
        }
        if quantization == "4bit":
            installed_bnb = validate_bitsandbytes_4bit_runtime()
            print(
                "bitsandbytes NF4 preflight passed "
                f"(version={installed_bnb}, PyTorch CUDA={torch.version.cuda})."
            )
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        elif quantization != "none":
            raise ValueError("quantization must be '4bit' or 'none'")

        processor = AutoProcessor.from_pretrained(model_name, revision=revision)
        model = LlavaForConditionalGeneration.from_pretrained(
            model_name, **load_kwargs
        )
        vision_config = model.config.vision_config
        if getattr(processor, "patch_size", None) is None:
            processor.patch_size = vision_config.patch_size
        if getattr(processor, "vision_feature_select_strategy", None) is None:
            processor.vision_feature_select_strategy = (
                model.config.vision_feature_select_strategy
            )
        if getattr(processor, "num_additional_image_tokens", None) is None:
            processor.num_additional_image_tokens = 1
        return cls(
            model,
            processor,
            layer_offset=layer_offset,
            projection_chunk_size=projection_chunk_size,
            fixed_concepts=fixed_concepts,
        )

    @property
    def resolved_revision(self) -> str:
        return str(getattr(self.model.config, "_commit_hash", None) or "unknown")

    def build_prompt(self, prompt_text: str) -> str:
        chat_template = getattr(self.processor, "chat_template", None)
        if chat_template:
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]
            return self.processor.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=False
            )
        return f"USER: <image>\n{prompt_text} ASSISTANT:"

    def extract(
        self,
        image: Any,
        prompt_text: str,
        *,
        include_vision_localizers: bool = False,
    ) -> PatchEvidence:
        prompt = self.build_prompt(prompt_text)
        inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        visual_positions, query_position = visual_positions_and_query(
            inputs["input_ids"], inputs["attention_mask"], self.image_token_id
        )
        grid_size = square_grid(visual_positions.numel())
        processed_image = self._image_for_display(inputs["pixel_values"])
        localization_scores = (
            self._vision_localization_scores(inputs["pixel_values"])
            if include_vision_localizers
            else {}
        )
        for name, scores in localization_scores.items():
            if scores.numel() != visual_positions.numel():
                raise RuntimeError(
                    f"Vision localizer {name!r} returned {scores.numel()} patches; "
                    f"LLaVA uses {visual_positions.numel()}"
                )

        moved_inputs: dict[str, torch.Tensor] = {}
        for name, value in inputs.items():
            if not isinstance(value, torch.Tensor):
                continue
            if value.is_floating_point():
                value = value.to(device=self.input_device, dtype=self.compute_dtype)
            else:
                value = value.to(self.input_device)
            moved_inputs[name] = value
        visual_positions = visual_positions.to(self.input_device)

        with torch.inference_mode():
            outputs = self.model(
                **moved_inputs,
                output_hidden_states=True,
                output_attentions=True,
                use_cache=False,
                return_dict=True,
            )
            if outputs.hidden_states is None or outputs.attentions is None:
                raise RuntimeError(
                    "Model did not return hidden states and attentions; eager attention "
                    "is required for the Phase 01 comparison"
                )
            hidden_layer = outputs.hidden_states[self.layer_offset][0]
            patch_hidden = hidden_layer.index_select(0, visual_positions)
            attention_layer = outputs.attentions[self.layer_offset]
            if attention_layer is None:
                raise RuntimeError("Selected LLaVA layer returned no attention tensor")
            attention_scores = attention_layer[
                0, :, query_position, visual_positions
            ].float().mean(dim=0)
            evidence_scores, top_token_ids = self._project_patch_chunks(patch_hidden)

        result = PatchEvidence(
            hidden_states=patch_hidden.detach().to("cpu", dtype=torch.float16),
            attention_scores=attention_scores.detach().cpu(),
            localization_scores=localization_scores,
            evidence_scores={
                key: value.detach().cpu()
                for key, value in evidence_scores.items()
            },
            top_token_ids=top_token_ids.detach().cpu(),
            grid_size=grid_size,
            processed_image=processed_image,
        )
        del outputs, moved_inputs
        torch.cuda.empty_cache()
        return result

    def _vision_localization_scores(
        self, pixel_values: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        vision_tower = getattr(self.model, "vision_tower", None)
        if not isinstance(vision_tower, nn.Module):
            raise RuntimeError("Could not locate LLaVA's frozen vision tower")
        vision_device = _module_device(vision_tower)
        vision_dtype = _module_dtype(vision_tower)
        moved_pixels = pixel_values.to(device=vision_device, dtype=vision_dtype)
        with torch.inference_mode():
            outputs = vision_tower(
                moved_pixels,
                output_attentions=True,
                return_dict=True,
            )
        attentions = getattr(outputs, "attentions", None)
        if not attentions or any(layer is None for layer in attentions):
            raise RuntimeError(
                "Vision tower returned no attention tensors; eager attention is required"
            )
        scores = {
            "vision_cls_attention": final_cls_attention(attentions).detach().cpu(),
            "vision_attention_rollout": attention_rollout(attentions).detach().cpu(),
        }
        del outputs, moved_pixels
        torch.cuda.empty_cache()
        return scores

    def _image_for_display(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if pixel_values.ndim != 4 or pixel_values.shape[0] != 1:
            raise RuntimeError("Classic LLaVA should produce one [3, H, W] image tensor")
        image = pixel_values[0].detach().float().cpu()
        image_processor = self.processor.image_processor
        mean = torch.tensor(image_processor.image_mean).view(-1, 1, 1)
        std = torch.tensor(image_processor.image_std).view(-1, 1, 1)
        return ((image * std + mean).clamp(0, 1) * 255).round().to(torch.uint8)

    def _project_patch_chunks(
        self, patch_hidden: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        score_parts: dict[str, list[torch.Tensor]] = {
            "maxprob": [],
            "margin": [],
            "negentropy": [],
        }
        if self.concept_token_ids:
            score_parts["concept_logprob"] = []
        top_id_parts: list[torch.Tensor] = []
        projection_device = _module_device(self.final_norm)
        for chunk in patch_hidden.split(self.projection_chunk_size, dim=0):
            chunk = chunk.to(projection_device)
            logits = self.lm_head(self.final_norm(chunk))
            for method in ("maxprob", "margin", "negentropy"):
                score_parts[method].append(logits_to_evidence(logits, method))
            if self.concept_token_ids:
                concept_ids = torch.tensor(
                    self.concept_token_ids,
                    device=logits.device,
                    dtype=torch.long,
                )
                score_parts["concept_logprob"].append(
                    logits_to_token_mass(logits, concept_ids)
                )
            top_id_parts.append(torch.topk(logits, k=5, dim=-1).indices)
            del logits
        return (
            {method: torch.cat(parts) for method, parts in score_parts.items()},
            torch.cat(top_id_parts),
        )

    def decode_token_ids(self, token_ids: torch.Tensor) -> list[list[str]]:
        tokenizer = self.processor.tokenizer
        return [
            tokenizer.convert_ids_to_tokens([int(token_id) for token_id in row])
            for row in token_ids.tolist()
        ]
