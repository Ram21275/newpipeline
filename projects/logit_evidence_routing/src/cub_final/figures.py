"""Programmatic paper figures from completed clustered analysis only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def render_final_figures(analysis_path: Path, output_dir: Path) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    report = json.loads(analysis_path.read_text(encoding="utf-8"))
    if report.get("status") != "COMPLETE_FROM_PROVIDED_PER_EXAMPLE_OUTPUTS":
        raise RuntimeError("figures require completed per-example analysis")
    families = report["cluster_bootstrap"]["families"]
    if not families:
        raise RuntimeError("analysis contains no estimand families")
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = sorted(families)
    estimates = [float(families[label]["estimate"]) for label in labels]
    lows = [
        float(families[label].get("simultaneous_ci_low", families[label]["ci_low"]))
        for label in labels
    ]
    highs = [
        float(families[label].get("simultaneous_ci_high", families[label]["ci_high"]))
        for label in labels
    ]
    figure_height = max(4.0, 0.28 * len(labels) + 1.5)
    fig, ax = plt.subplots(figsize=(9.5, figure_height), constrained_layout=True)
    y = list(range(len(labels)))
    ax.errorbar(
        estimates,
        y,
        xerr=[
            [estimate - low for estimate, low in zip(estimates, lows)],
            [high - estimate for estimate, high in zip(estimates, highs)],
        ],
        fmt="o",
        color="#1f5a99",
        ecolor="#555555",
        capsize=2,
    )
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_yticks(y, labels)
    ax.set_xlabel("Paired effect (simultaneous 95% interval where applicable)")
    ax.set_title("CUB final paired effects; image-cluster bootstrap")
    ax.grid(axis="x", alpha=0.2)
    ax.invert_yaxis()
    outputs = []
    for suffix in ("svg", "pdf"):
        path = output_dir / f"final_paired_effects.{suffix}"
        fig.savefig(path)
        outputs.append(str(path))
    plt.close(fig)
    return {"status": "PASS", "files": outputs, "family_count": len(labels)}

