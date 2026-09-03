#!/usr/bin/env python3
"""Exercise every selector on planted synthetic evidence; not a research result."""

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lger import EvidenceClassifier, EvidenceRouter, RouterConfig  # noqa: E402


def build_inputs(seed: int) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(seed)
    layers, patches, hidden_dim, vocabulary = 4, 64, 24, 32
    hidden = torch.randn(layers, patches, hidden_dim, generator=generator)
    logits = 0.25 * torch.randn(layers, patches, vocabulary, generator=generator)
    attention = torch.randn(patches, generator=generator)
    planted = torch.tensor([9, 18, 27, 36])
    for layer in range(layers):
        logits[layer, planted, 3] += 3.0 + 0.25 * layer
    attention[torch.tensor([1, 2, 3, 4])] += 3.0
    coords = torch.stack(
        (torch.arange(patches) % 8, torch.arange(patches) // 8), dim=-1
    ).float()
    layer_ids = torch.tensor([29, 30, 31, 32])
    return hidden, logits, attention, coords, layer_ids, planted


def run(seed: int, k: int) -> dict[str, object]:
    hidden, logits, attention, coords, layer_ids, planted = build_inputs(seed)
    variants = {
        "random": RouterConfig(selector="random", score="maxprob", aggregate="last", k=k, seed=seed),
        "attention": RouterConfig(selector="attention", score="maxprob", aggregate="last", k=k, seed=seed),
        "logit": RouterConfig(selector="logit", score="maxprob", aggregate="last", k=k, seed=seed),
        "attention_logit": RouterConfig(selector="attention_logit", score="maxprob", aggregate="last", k=k, seed=seed),
        "lger": RouterConfig(selector="lger", score="margin", aggregate="mean_layers", k=k, seed=seed),
    }
    output: dict[str, object] = {"synthetic": True, "seed": seed, "k": k, "selectors": {}}
    selected_for_classifier = None
    for name, config in variants.items():
        selected = EvidenceRouter(config)(
            hidden,
            logits,
            coords,
            layer_ids,
            attention_scores=attention,
        )
        overlap = len(set(selected.patch_indices.tolist()) & set(planted.tolist()))
        output["selectors"][name] = {
            "patch_indices": selected.patch_indices.tolist(),
            "planted_evidence_overlap": overlap,
        }
        if name == "lger":
            selected_for_classifier = selected

    assert selected_for_classifier is not None
    classifier = EvidenceClassifier(
        hidden_dim=hidden.shape[-1],
        num_classes=5,
        model_dim=32,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
    ).eval()
    with torch.no_grad():
        prediction = classifier(
            selected_for_classifier.hidden_states,
            selected_for_classifier.patch_coords,
            selected_for_classifier.layer_ids,
        )
    output["classifier_output_shape"] = list(prediction.shape)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(args.seed, args.k)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
