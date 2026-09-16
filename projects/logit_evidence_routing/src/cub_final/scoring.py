"""Binary answer scoring, parsing, and polarity-safe effect definitions."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Sequence


@dataclass(frozen=True)
class BinaryScore:
    positive_answer: str
    negative_answer: str
    positive_log_probability: float
    negative_log_probability: float

    @property
    def raw_margin(self) -> float:
        return self.positive_log_probability - self.negative_log_probability


def sequence_log_probability(
    prompt_inputs: dict[str, Any],
    token_ids: Sequence[int],
    *,
    first_logits: Any,
    append_and_forward: Callable[[dict[str, Any], Sequence[int]], Any],
) -> float:
    """Score a complete candidate string, including multi-token labels.

    ``append_and_forward`` returns logits for the appended positions.  Keeping
    this helper framework-neutral makes its indexing testable without a GPU.
    """

    if not token_ids:
        raise ValueError("candidate answer must contain at least one token")
    import torch
    import torch.nn.functional as F

    first = F.log_softmax(first_logits.float(), dim=-1)[int(token_ids[0])]
    if len(token_ids) == 1:
        return float(first.detach().cpu())
    continuation_logits = append_and_forward(prompt_inputs, token_ids)
    if continuation_logits.shape[0] != len(token_ids) - 1:
        raise ValueError("continuation logits are not aligned to candidate tokens")
    targets = torch.tensor(
        token_ids[1:], dtype=torch.long, device=continuation_logits.device
    )
    rest = F.log_softmax(continuation_logits.float(), dim=-1).gather(
        1, targets[:, None]
    ).sum()
    return float((first + rest).detach().cpu())


def polarity_effects(
    *, baseline_margin: float, intervention_margin: float, target: int
) -> dict[str, float]:
    """Return the mandated raw and correctness-signed intervention effects."""

    if target not in (0, 1):
        raise ValueError("target must be 0 or 1")
    if not all(math.isfinite(value) for value in (baseline_margin, intervention_margin)):
        raise ValueError("margins must be finite")
    delta_raw = baseline_margin - intervention_margin
    return {
        "margin_baseline": baseline_margin,
        "margin_intervention": intervention_margin,
        "delta_raw": delta_raw,
        "delta_correct": (2 * target - 1) * delta_raw,
    }


def fixed_binary_parser(text: str, *, positive: str, negative: str) -> str:
    """Parse only a unique leading lexical answer; never silently drop output."""

    normalized = text.strip().casefold()
    tokens = re.findall(r"[\w'-]+", normalized)
    if not tokens:
        return "invalid"
    first = tokens[0]
    pos = positive.strip().casefold()
    neg = negative.strip().casefold()
    if first == pos and first != neg:
        return "positive"
    if first == neg and first != pos:
        return "negative"
    return "invalid"


def assert_final_logit_agreement(
    reconstructed_logits: Any,
    model_logits: Any,
    *,
    atol: float,
    rtol: float,
) -> dict[str, float]:
    """Fail closed unless a reconstructed final distribution matches the model."""

    import torch

    if reconstructed_logits.shape != model_logits.shape:
        raise AssertionError("reconstructed and model logits have different shapes")
    # ``device_map="auto"`` can shard the language-model head and the final
    # decoder block across different GPUs.  The reconstructed logits then stay
    # on the head device while Transformers returns ``model.logits`` on another
    # device.  This is a small validation-only vector, so compare detached fp32
    # host copies rather than assuming both tensors share a CUDA device.
    reconstructed = reconstructed_logits.detach().to(device="cpu", dtype=torch.float32)
    reference = model_logits.detach().to(device="cpu", dtype=torch.float32)
    difference = (reconstructed - reference).abs()
    maximum = float(difference.max().detach().cpu())
    mean = float(difference.mean().detach().cpu())
    if not torch.allclose(reconstructed, reference, atol=atol, rtol=rtol):
        raise AssertionError(
            f"final logit reconstruction disagrees with model.logits: max_abs={maximum:.6g}"
        )
    return {"max_abs_error": maximum, "mean_abs_error": mean, "atol": atol, "rtol": rtol}
