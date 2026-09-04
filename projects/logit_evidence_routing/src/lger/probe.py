"""Matched linear-probe training and metrics for the Phase 01 comparison."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .reproducibility import set_deterministic_seed


@dataclass(frozen=True)
class ProbeResult:
    accuracy: float
    macro_f1: float
    train_loss: float


def classification_metrics(
    predictions: torch.Tensor, targets: torch.Tensor, num_classes: int
) -> tuple[float, float]:
    if predictions.shape != targets.shape or predictions.ndim != 1:
        raise ValueError("predictions and targets must be aligned 1D tensors")
    accuracy = predictions.eq(targets).float().mean().item()
    f1_scores: list[float] = []
    for class_index in range(num_classes):
        predicted = predictions.eq(class_index)
        actual = targets.eq(class_index)
        true_positive = (predicted & actual).sum().item()
        false_positive = (predicted & ~actual).sum().item()
        false_negative = (~predicted & actual).sum().item()
        denominator = 2 * true_positive + false_positive + false_negative
        f1_scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return accuracy, sum(f1_scores) / len(f1_scores)


def run_linear_probe(
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    val_features: torch.Tensor,
    val_targets: torch.Tensor,
    *,
    num_classes: int,
    seed: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
) -> ProbeResult:
    """Train the exact same mean-feature linear classifier for each selector."""

    if train_features.ndim != 2 or val_features.ndim != 2:
        raise ValueError("probe features must have shape [examples, hidden]")
    if train_features.shape[1] != val_features.shape[1]:
        raise ValueError("train and validation feature dimensions differ")
    if train_targets.numel() != train_features.shape[0]:
        raise ValueError("train targets do not align with features")
    if val_targets.numel() != val_features.shape[0]:
        raise ValueError("validation targets do not align with features")
    if epochs <= 0:
        raise ValueError("epochs must be positive")

    set_deterministic_seed(seed)
    train_features = F.normalize(train_features.float(), dim=-1).to(device)
    val_features = F.normalize(val_features.float(), dim=-1).to(device)
    train_targets = train_targets.long().to(device)
    val_targets = val_targets.long().to(device)
    classifier = nn.Linear(train_features.shape[1], num_classes).to(device)
    optimizer = torch.optim.AdamW(
        classifier.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    classifier.train()
    loss = torch.tensor(float("nan"), device=device)
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(classifier(train_features), train_targets)
        loss.backward()
        optimizer.step()

    classifier.eval()
    with torch.no_grad():
        predictions = classifier(val_features).argmax(dim=-1).cpu()
    accuracy, macro_f1 = classification_metrics(
        predictions, val_targets.cpu(), num_classes
    )
    return ProbeResult(
        accuracy=accuracy,
        macro_f1=macro_f1,
        train_loss=float(loss.detach().cpu()),
    )


def jaccard(indices_a: torch.Tensor, indices_b: torch.Tensor) -> float:
    set_a = set(int(value) for value in indices_a.tolist())
    set_b = set(int(value) for value in indices_b.tolist())
    union = set_a | set_b
    return 0.0 if not union else len(set_a & set_b) / len(union)
