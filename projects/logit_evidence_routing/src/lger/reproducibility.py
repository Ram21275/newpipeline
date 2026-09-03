"""Experiment records required for reproducible paper results."""

import json
import random
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class RunRecord:
    git_commit: str
    config: dict[str, Any]
    seed: int
    dataset_split_id: str
    model_checkpoint: str
    metrics: dict[str, float]
    selection_statistics: dict[str, float]
    runtime_seconds: float
    max_gpu_memory_bytes: int


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def current_git_commit(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def save_run_record(record: RunRecord, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(asdict(record), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
