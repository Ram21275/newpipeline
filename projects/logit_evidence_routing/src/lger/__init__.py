"""Core components for Logit-Guided Evidence Routing (LGER)."""

from .classifier import EvidenceClassifier
from .extraction import FrozenLogitProjector, gather_visual_hidden_states
from .routing import EvidenceRouter
from .scoring import aggregate_layer_scores, logits_to_evidence
from .types import RouterConfig, SelectionResult

__all__ = [
    "EvidenceClassifier",
    "EvidenceRouter",
    "FrozenLogitProjector",
    "RouterConfig",
    "SelectionResult",
    "aggregate_layer_scores",
    "gather_visual_hidden_states",
    "logits_to_evidence",
]
