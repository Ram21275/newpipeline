"""Regularized linear diagnostic probes with saved preprocessing state."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .core import atomic_write_json


class AttributeProbe:
    def __init__(self, *, regularization_c: float = 1.0, seed: int = 20260916) -> None:
        if regularization_c <= 0:
            raise ValueError("regularization_c must be positive")
        self.regularization_c = regularization_c
        self.seed = seed
        self.pipeline: Any | None = None

    def fit(self, features: Any, labels: Any) -> "AttributeProbe":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        self.pipeline = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=self.regularization_c,
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=self.seed,
                    ),
                ),
            ]
        )
        self.pipeline.fit(features, labels)
        return self

    def predict_score(self, features: Any) -> Any:
        if self.pipeline is None:
            raise RuntimeError("probe has not been fit")
        return self.pipeline.predict_proba(features)[:, 1]

    def save(self, destination: Path, *, training_manifest_hash: str) -> None:
        if self.pipeline is None:
            raise RuntimeError("probe has not been fit")
        scale = self.pipeline.named_steps["scale"]
        classifier = self.pipeline.named_steps["classifier"]
        atomic_write_json(
            destination,
            {
                "probe_type": "standardized_l2_logistic_regression",
                "regularization_c": self.regularization_c,
                "seed": self.seed,
                "training_manifest_hash": training_manifest_hash,
                "mean": scale.mean_.tolist(),
                "scale": scale.scale_.tolist(),
                "classes": classifier.classes_.tolist(),
                "coefficients": classifier.coef_.tolist(),
                "intercept": classifier.intercept_.tolist(),
            },
        )


def shuffled_labels(labels: Any, *, seed: int) -> Any:
    import numpy as np

    values = np.asarray(labels).copy()
    np.random.default_rng(seed).shuffle(values)
    return values

