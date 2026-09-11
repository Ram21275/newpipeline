"""Matched masked multi-label probes for Phase 3 attribute recoverability."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .cub import certainty_policy_target
from .reproducibility import set_deterministic_seed
from .stage_cache import REQUIRED_STAGE_NAMES, config_digest


@dataclass(frozen=True)
class StageAttributeDataset:
    image_ids: torch.Tensor
    splits: tuple[str, ...]
    attribute_ids: tuple[int, ...]
    attribute_names: tuple[str, ...]
    attribute_groups: tuple[str, ...]
    targets: torch.Tensor
    features: dict[tuple[str, str], torch.Tensor]
    cache_config_digest: str
    certainty_policy_overrides_masked: int


@dataclass(frozen=True)
class MultiLabelProbeResult:
    train_logits: torch.Tensor
    validation_logits: torch.Tensor
    thresholds: torch.Tensor
    train_loss: float
    normalization_mean: torch.Tensor
    normalization_scale: torch.Tensor
    classifier_weight: torch.Tensor
    classifier_bias: torch.Tensor


def mean_pool_probe_patch_contributions(
    patch_features: torch.Tensor,
    result: MultiLabelProbeResult,
    *,
    thresholds: torch.Tensor | None = None,
) -> torch.Tensor:
    """Decompose a mean-pooled linear probe margin into exact patch terms.

    The returned tensor has shape ``[examples, patches, attributes]``.  Its
    patch sum reconstructs either the raw classifier logit or, when
    ``thresholds`` is supplied, the threshold-centered decision margin.  The
    classifier bias (and optional threshold) is divided evenly over patches;
    this convention makes the decomposition exact without implying that the
    constant offset is spatial evidence.
    """

    if (
        patch_features.ndim != 3
        or patch_features.shape[-1] != result.classifier_weight.shape[1]
    ):
        raise ValueError(
            "patch features must have shape [examples, patches, feature_dim]"
        )
    if patch_features.shape[1] == 0:
        raise ValueError("at least one patch is required")
    attributes = result.classifier_weight.shape[0]
    if thresholds is not None and tuple(thresholds.shape) != (attributes,):
        raise ValueError("thresholds must contain one value per attribute")
    mean = result.normalization_mean.to(dtype=torch.float32)
    scale = result.normalization_scale.to(dtype=torch.float32)
    weight = result.classifier_weight.to(dtype=torch.float32)
    bias = result.classifier_bias.to(dtype=torch.float32)
    if (
        mean.ndim != 1
        or scale.shape != mean.shape
        or mean.numel() != patch_features.shape[-1]
    ):
        raise ValueError("probe normalization parameters do not match patch features")
    normalized = (patch_features.float() - mean) / scale
    patch_count = patch_features.shape[1]
    contributions = torch.einsum("npd,ad->npa", normalized, weight) / patch_count
    offset = bias
    if thresholds is not None:
        offset = offset - thresholds.float()
    return contributions + offset.view(1, 1, -1) / patch_count


def binary_auroc(scores: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute tie-aware binary AUROC from the Mann-Whitney rank statistic."""

    scores = scores.detach().double().flatten()
    targets = targets.detach().bool().flatten()
    if scores.shape != targets.shape or scores.numel() == 0:
        raise ValueError("AUROC scores and targets must be aligned and non-empty")
    positives = int(targets.sum())
    negatives = targets.numel() - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    order = torch.argsort(scores, stable=True)
    sorted_scores = scores[order]
    ranks = torch.empty_like(scores)
    start = 0
    while start < scores.numel():
        end = start + 1
        while end < scores.numel() and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    positive_rank_sum = float(ranks[targets].sum())
    return (
        positive_rank_sum - positives * (positives + 1) / 2
    ) / (positives * negatives)


def binary_f1(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    predictions = predictions.detach().bool().flatten()
    targets = targets.detach().bool().flatten()
    if predictions.shape != targets.shape or predictions.numel() == 0:
        raise ValueError("F1 predictions and targets must be aligned and non-empty")
    true_positive = int((predictions & targets).sum())
    false_positive = int((predictions & ~targets).sum())
    false_negative = int((~predictions & targets).sum())
    denominator = 2 * true_positive + false_positive + false_negative
    return 0.0 if denominator == 0 else 2 * true_positive / denominator


def select_training_f1_threshold(scores: torch.Tensor, targets: torch.Tensor) -> float:
    """Choose an F1 threshold from training scores only, with stable tie-breaking."""

    scores = scores.detach().float().flatten()
    targets = targets.detach().bool().flatten()
    if scores.shape != targets.shape or scores.numel() == 0:
        raise ValueError("threshold scores and targets must be aligned and non-empty")
    unique = torch.unique(scores, sorted=True)
    epsilon = max(torch.finfo(scores.dtype).eps, 1e-6)
    candidates = torch.cat(
        (unique[:1] - epsilon, unique, unique[-1:] + epsilon)
    )
    best_threshold = float(candidates[0])
    best_f1 = -1.0
    for candidate in candidates:
        value = binary_f1(scores >= candidate, targets)
        threshold = float(candidate)
        if value > best_f1 or (value == best_f1 and threshold > best_threshold):
            best_f1 = value
            best_threshold = threshold
    return best_threshold


def shuffle_observed_targets(targets: torch.Tensor, *, seed: int) -> torch.Tensor:
    """Permute each attribute only among its observed training labels."""

    if targets.ndim != 2:
        raise ValueError("multi-label targets must have shape [examples, attributes]")
    generator = torch.Generator().manual_seed(seed)
    shuffled = targets.clone()
    for column in range(targets.shape[1]):
        observed = torch.isfinite(targets[:, column]).nonzero().flatten()
        permutation = torch.randperm(observed.numel(), generator=generator)
        shuffled[observed, column] = targets[observed[permutation], column]
    return shuffled


def random_project_features(
    train_features: torch.Tensor,
    validation_features: torch.Tensor,
    *,
    output_dim: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply one frozen Gaussian projection fitted without validation labels."""

    if train_features.ndim != 2 or validation_features.ndim != 2:
        raise ValueError("features must be rank-2")
    if train_features.shape[1] != validation_features.shape[1]:
        raise ValueError("training and validation feature dimensions differ")
    if not 0 < output_dim <= train_features.shape[1]:
        raise ValueError("random projection dimension is invalid")
    generator = torch.Generator().manual_seed(seed)
    projection = torch.randn(
        train_features.shape[1], output_dim, generator=generator
    ) / math.sqrt(output_dim)
    return train_features.float() @ projection, validation_features.float() @ projection


def run_masked_multilabel_probe(
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    validation_features: torch.Tensor,
    *,
    seed: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
) -> MultiLabelProbeResult:
    """Fit one shared linear layer while masking uncertain/missing labels."""

    if train_features.ndim != 2 or validation_features.ndim != 2:
        raise ValueError("probe features must have shape [examples, hidden]")
    if train_targets.ndim != 2 or train_targets.shape[0] != train_features.shape[0]:
        raise ValueError("training targets do not align with features")
    if train_features.shape[1] != validation_features.shape[1]:
        raise ValueError("training and validation feature dimensions differ")
    if epochs <= 0 or learning_rate <= 0 or weight_decay < 0:
        raise ValueError("probe optimization arguments are invalid")
    observed = torch.isfinite(train_targets)
    if not bool(observed.any(dim=0).all()):
        raise ValueError("every selected attribute needs observed training labels")

    set_deterministic_seed(seed)
    mean = train_features.float().mean(dim=0, keepdim=True)
    scale = train_features.float().std(dim=0, unbiased=False, keepdim=True)
    scale = torch.where(scale > 1e-6, scale, torch.ones_like(scale))
    normalized_train = ((train_features.float() - mean) / scale).to(device)
    normalized_validation = (
        (validation_features.float() - mean) / scale
    ).to(device)
    targets = torch.nan_to_num(train_targets.float(), nan=0.0).to(device)
    mask = observed.to(device)
    classifier = nn.Linear(train_features.shape[1], train_targets.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        classifier.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    loss = torch.tensor(float("nan"), device=device)
    classifier.train()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = classifier(normalized_train)
        loss = F.binary_cross_entropy_with_logits(logits[mask], targets[mask])
        loss.backward()
        optimizer.step()

    classifier.eval()
    with torch.no_grad():
        train_logits = classifier(normalized_train).cpu()
        validation_logits = classifier(normalized_validation).cpu()
    thresholds = []
    for column in range(train_targets.shape[1]):
        column_mask = observed[:, column]
        thresholds.append(
            select_training_f1_threshold(
                train_logits[column_mask, column],
                train_targets[column_mask, column],
            )
        )
    return MultiLabelProbeResult(
        train_logits=train_logits,
        validation_logits=validation_logits,
        thresholds=torch.tensor(thresholds),
        train_loss=float(loss.detach().cpu()),
        normalization_mean=mean.squeeze(0).cpu(),
        normalization_scale=scale.squeeze(0).cpu(),
        classifier_weight=classifier.weight.detach().cpu(),
        classifier_bias=classifier.bias.detach().cpu(),
    )


def _validated_cache_documents(
    cache_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str]:
    run_config = json.loads((cache_dir / "run_config.json").read_text())
    index = json.loads((cache_dir / "index.json").read_text())
    report = json.loads((cache_dir / "validation_report.json").read_text())
    digest = config_digest(run_config)
    if (
        run_config.get("purpose")
        != "full_240_image_development_pilot_stage_cache"
        or int(run_config.get("official_test_images", -1)) != 0
        or index.get("config_digest") != digest
        or index.get("complete") is not True
        or report.get("config_digest") != digest
        or report.get("status") != "PASS"
        or int(report.get("images", 0)) != 240
        or int(report.get("official_test_images", -1)) != 0
    ):
        raise RuntimeError("Phase 3 requires a validated 240-image development cache")
    return run_config, index, report, digest


def load_stage_attribute_dataset(
    cache_dir: Path,
    *,
    stages: tuple[str, ...] = REQUIRED_STAGE_NAMES,
    pooling: tuple[str, ...] = ("mean",),
) -> StageAttributeDataset:
    """Read only requested stage tensors and selected attribute metadata."""

    if not stages or not set(stages).issubset(REQUIRED_STAGE_NAMES):
        raise ValueError("requested stages are empty or unknown")
    if not pooling or not set(pooling).issubset({"mean", "max"}):
        raise ValueError("pooling must contain mean and/or max")
    _, index, _, digest = _validated_cache_documents(cache_dir)
    try:
        from safetensors import safe_open
    except ImportError as error:  # pragma: no cover - Kaggle dependency guard
        raise RuntimeError("Phase 3 cache loading requires safetensors") from error

    feature_rows: dict[tuple[str, str], list[torch.Tensor]] = {
        (method, stage): [] for method in pooling for stage in stages
    }
    image_ids: list[int] = []
    splits: list[str] = []
    target_rows: list[torch.Tensor] = []
    attribute_ids: tuple[int, ...] | None = None
    attribute_names: tuple[str, ...] | None = None
    attribute_groups: tuple[str, ...] | None = None
    certainty_policy_overrides_masked = 0

    for shard in index["shards"]:
        tensor_path = cache_dir / shard["tensor_path"]
        metadata_path = cache_dir / shard["metadata_path"]
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("config_digest") != digest or metadata.get("complete") is not True:
            raise RuntimeError(f"incompatible shard metadata: {metadata_path}")
        with safe_open(tensor_path, framework="pt", device="cpu") as handle:
            for item in metadata["records"]:
                packed = item["packed_record"]
                image = packed["image"]
                image_id = int(image["image_id"])
                selected_ids = tuple(int(value) for value in image["selected_attribute_ids"])
                by_id = {
                    int(attribute["attribute_id"]): attribute
                    for attribute in image["attributes"]
                }
                selected_rows = [by_id[value] for value in selected_ids]
                names = tuple(str(row["name"]) for row in selected_rows)
                groups = tuple(str(row["group"]) for row in selected_rows)
                if attribute_ids is None:
                    attribute_ids = selected_ids
                    attribute_names = names
                    attribute_groups = groups
                elif (
                    selected_ids != attribute_ids
                    or names != attribute_names
                    or groups != attribute_groups
                ):
                    raise RuntimeError("selected attribute identity changes across records")
                filtered_targets = []
                for row in selected_rows:
                    target, was_masked = certainty_policy_target(row)
                    certainty_policy_overrides_masked += int(was_masked)
                    filtered_targets.append(target)
                targets = [
                    float(target) if target is not None else float("nan")
                    for target in filtered_targets
                ]
                target_rows.append(torch.tensor(targets))
                image_ids.append(image_id)
                splits.append(str(image["development_split"]))
                for stage in stages:
                    reference = packed["stages"][stage]
                    key = reference.get("__stage_cache_tensor__")
                    if not isinstance(key, str):
                        raise RuntimeError(f"missing tensor reference for {image_id}/{stage}")
                    tensor = handle.get_tensor(key).float()
                    for method in pooling:
                        pooled = (
                            tensor.mean(dim=0)
                            if method == "mean"
                            else tensor.max(dim=0).values
                        )
                        feature_rows[(method, stage)].append(pooled)

    expected_ids = [int(item["image_id"]) for item in index.get("records", [])]
    if image_ids != expected_ids or len(image_ids) != 240:
        raise RuntimeError("loaded Phase 3 image order differs from the cache index")
    if splits.count("train") != 160 or splits.count("val") != 80:
        raise RuntimeError("Phase 3 cache does not contain the frozen 160/80 split")
    if attribute_ids is None or attribute_names is None or attribute_groups is None:
        raise RuntimeError("Phase 3 cache contains no selected attributes")
    return StageAttributeDataset(
        image_ids=torch.tensor(image_ids, dtype=torch.long),
        splits=tuple(splits),
        attribute_ids=attribute_ids,
        attribute_names=attribute_names,
        attribute_groups=attribute_groups,
        targets=torch.stack(target_rows),
        features={key: torch.stack(rows) for key, rows in feature_rows.items()},
        cache_config_digest=digest,
        certainty_policy_overrides_masked=certainty_policy_overrides_masked,
    )
