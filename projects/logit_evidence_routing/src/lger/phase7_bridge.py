"""Bridge Phase 6 diagnostics to a bounded Phase 7 causal cohort."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import Any, Iterable, Mapping


CAUSAL_STAGE_ORDER = (
    "vision.early",
    "vision.middle",
    "vision.late",
    "projector.output",
    "llm.early",
    "llm.middle",
    "llm.late",
)

TRANSITION_STAGE_PAIRS = {
    "visual_to_language_bottleneck": ("projector.output", "vision.late"),
    "representation_utilization_gap": ("llm.late", "llm.middle"),
    "discriminative_semantic_localization_mismatch": (
        "vision.late",
        "projector.output",
    ),
    # Phase 6 detects whole-trajectory redistribution.  The first causal test is
    # therefore the narrow visual-to-language interface, not an invented layer.
    "evidence_redistribution": ("projector.output", "vision.late"),
}


def resolve_intervention_stages(transition: str) -> tuple[str, str]:
    """Map one unambiguous Phase 6 diagnosis to adjacent causal stages."""

    if transition == "mixed_or_null":
        raise RuntimeError(
            "Phase 7 is blocked: Phase 6 selected mixed_or_null, so no single "
            "causal transition is justified"
        )
    if transition not in TRANSITION_STAGE_PAIRS:
        raise RuntimeError(f"unknown Phase 6 transition: {transition}")
    selected, neighbor = TRANSITION_STAGE_PAIRS[transition]
    if abs(CAUSAL_STAGE_ORDER.index(selected) - CAUSAL_STAGE_ORDER.index(neighbor)) != 1:
        raise RuntimeError("configured intervention stages are not adjacent")
    return selected, neighbor


def _rank(seed: int, *parts: object) -> str:
    payload = json.dumps([seed, *parts], separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def select_matched_causal_cohort(
    rows: Iterable[Mapping[str, Any]],
    *,
    max_per_outcome_per_attribute: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Sample equal failures/successes within attribute and target strata.

    The cap bounds expensive intervention forwards.  Strata without both
    outcomes are transparently excluded rather than imputed.
    """

    if max_per_outcome_per_attribute <= 0:
        raise ValueError("max_per_outcome_per_attribute must be positive")
    grouped: dict[tuple[int, int], dict[int, list[dict[str, Any]]]] = defaultdict(
        lambda: {0: [], 1: []}
    )
    seen: set[str] = set()
    for source in rows:
        row = dict(source)
        decision_id = str(row.get("decision_id", "")).strip()
        if not decision_id or decision_id in seen:
            raise RuntimeError("joint decisions need unique non-empty decision_id values")
        seen.add(decision_id)
        try:
            attribute_id = int(row["attribute_id"])
            target = int(row["target"])
            failure = int(row["phase5_failure"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError("joint decision identity/outcome is malformed") from error
        if target not in (0, 1) or failure not in (0, 1):
            raise RuntimeError("target and phase5_failure must be binary")
        grouped[(attribute_id, target)][failure].append(row)

    selected: list[dict[str, Any]] = []
    for (attribute_id, target), outcomes in sorted(grouped.items()):
        count = min(
            len(outcomes[0]),
            len(outcomes[1]),
            max_per_outcome_per_attribute,
        )
        if count == 0:
            continue
        for failure in (0, 1):
            ranked = sorted(
                outcomes[failure],
                key=lambda row: _rank(
                    seed,
                    attribute_id,
                    target,
                    failure,
                    row["decision_id"],
                ),
            )
            selected.extend(ranked[:count])
    if not selected:
        raise RuntimeError("no attribute/target stratum contains failures and successes")
    return sorted(selected, key=lambda row: str(row["decision_id"]))
