"""Typed configuration and results shared by all routing methods."""

from dataclasses import dataclass
from typing import Mapping

import torch


VALID_SELECTORS = {"random", "attention", "logit", "attention_logit", "lger"}
VALID_SCORES = {"maxprob", "margin", "negentropy"}
VALID_AGGREGATES = {"last", "mean_layers", "persistent"}
VALID_CONTEXT = {"none", "random"}


@dataclass(frozen=True)
class RouterConfig:
    """Configuration for a leakage-safe, non-learned patch router."""

    selector: str = "lger"
    score: str = "margin"
    aggregate: str = "mean_layers"
    k: int = 32
    context: str = "none"
    context_k: int = 0
    persistent_lambda: float = 0.5
    seed: int = 0

    def __post_init__(self) -> None:
        if self.selector not in VALID_SELECTORS:
            raise ValueError(f"selector must be one of {sorted(VALID_SELECTORS)}")
        if self.score not in VALID_SCORES:
            raise ValueError(f"score must be one of {sorted(VALID_SCORES)}")
        if self.aggregate not in VALID_AGGREGATES:
            raise ValueError(f"aggregate must be one of {sorted(VALID_AGGREGATES)}")
        if self.context not in VALID_CONTEXT:
            raise ValueError(f"context must be one of {sorted(VALID_CONTEXT)}")
        if self.k <= 0:
            raise ValueError("k must be positive")
        if self.context_k < 0:
            raise ValueError("context_k cannot be negative")
        if self.context == "none" and self.context_k != 0:
            raise ValueError("context_k must be zero when context='none'")
        if self.context == "random" and self.context_k == 0:
            raise ValueError("context_k must be positive when context='random'")
        if self.persistent_lambda < 0:
            raise ValueError("persistent_lambda cannot be negative")


@dataclass(frozen=True)
class SelectionResult:
    """Selected representations, their identity, and routing diagnostics."""

    hidden_states: torch.Tensor
    patch_indices: torch.Tensor
    patch_coords: torch.Tensor
    scores: torch.Tensor
    layer_ids: torch.Tensor
    context_mask: torch.Tensor
    layer_statistics: Mapping[str, torch.Tensor]

    @property
    def evidence_count(self) -> int:
        return int((~self.context_mask).sum().item())

    @property
    def context_count(self) -> int:
        return int(self.context_mask.sum().item())
