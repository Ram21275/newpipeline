#!/usr/bin/env python3
"""Resolve retained Kaggle inputs quickly and fail with named diagnostics."""

from __future__ import annotations

import argparse
import json
import os
import tarfile
from pathlib import Path
from typing import Iterable


PRUNED_DIRECTORIES = {
    ".git", "__pycache__", "dense_scores", "images", "qualitative", "records"
}


def bounded_directories(root: Path, max_depth: int = 5) -> Iterable[tuple[Path, set[str]]]:
    """Walk metadata directories without descending into image/record payloads."""

    if not root.is_dir():
        return
    root_depth = len(root.parts)
    for current, directories, files in os.walk(root):
        path = Path(current)
        depth = len(path.parts) - root_depth
        directories[:] = [
            name for name in directories
            if name not in PRUNED_DIRECTORIES and depth < max_depth
        ]
        yield path, set(files)


def scan(root: Path) -> dict[str, list[Path]]:
    found: dict[str, list[Path]] = {
        "run": [], "followup": [], "stage_cache": [], "localizer_cache": [],
        "phase4": [], "cub": [], "followup_archive": [],
    }
    for path, files in bounded_directories(root):
        if "logit_evidence_followup_complete.tar.gz" in files:
            found["followup_archive"].append(
                path / "logit_evidence_followup_complete.tar.gz"
            )
        if "multiphase_status.json" in files and (path / "phase7_bridge").is_dir():
            found["run"].append(path)
        if {"cub_replication_manifest.csv", "phase9_plan.json"} <= files:
            found["followup"].append(path)
        if {"validation_report.json", "index.json", "run_config.json"} <= files:
            found["stage_cache"].append(path)
        if "extraction_config.json" in files and (path / "records").is_dir():
            found["localizer_cache"].append(path)
        if {"phase4_run_report.json", "attribute_eligibility.csv"} <= files:
            found["phase4"].append(path)
        if path.name == "CUB_200_2011" and "images.txt" in files \
                and (path / "images").is_dir():
            found["cub"].append(path)
    return found


def choose(label: str, candidates: list[Path], preferred_name: str) -> Path:
    unique = sorted({path.resolve() for path in candidates}, key=str)
    exact = [path for path in unique if path.name == preferred_name]
    selected = exact or [path for path in unique if preferred_name in str(path)] or unique
    if not selected:
        raise RuntimeError(
            f"MISSING {label}: attach the retained dataset containing {preferred_name!r} "
            "to this Kaggle notebook"
        )
    if len(selected) > 1:
        listing = "\n  ".join(str(path) for path in selected)
        raise RuntimeError(f"AMBIGUOUS {label}; found multiple candidates:\n  {listing}")
    return selected[0]


def link_directory(source: Path, destination: Path, marker: str) -> Path:
    if destination.exists() or destination.is_symlink():
        if (destination / marker).exists():
            return destination
        raise RuntimeError(
            f"INVALID existing path: {destination}\n"
            f"Expected marker: {destination / marker}\n"
            "Start a fresh Kaggle session or move this incomplete directory aside."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source, target_is_directory=True)
    return destination


def safe_extract(archive: Path, working: Path) -> None:
    print(f"Extracting retained follow-up archive once: {archive}", flush=True)
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle.getmembers():
            target = (working / member.name).resolve()
            if working.resolve() not in target.parents and target != working.resolve():
                raise RuntimeError(f"unsafe archive member: {member.name}")
        handle.extractall(working, filter="data")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--working-root", type=Path, default=Path("/kaggle/working"))
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--output", type=Path,
                        default=Path("/kaggle/working/priority1_input_paths.json"))
    args = parser.parse_args()
    working = args.working_root.resolve()
    inputs = args.input_root.resolve()
    working.mkdir(parents=True, exist_ok=True)
    if not inputs.is_dir():
        raise RuntimeError(f"Kaggle input root does not exist: {inputs}")

    discovered = scan(inputs)
    working_discovered = scan(working)
    for label in discovered:
        if label != "followup_archive":
            discovered[label].extend(working_discovered[label])

    followup_target = working / "logit_evidence_followup_complete"
    if not (followup_target / "cub_replication_manifest.csv").is_file():
        if not discovered["followup"]:
            archive = choose(
                "follow-up archive", discovered["followup_archive"],
                "logit_evidence_followup_complete.tar.gz",
            )
            safe_extract(archive, working)
            discovered["followup"].extend(scan(working)["followup"])

    run = link_directory(
        choose("multiphase run", discovered["run"],
               "multiphase_development_5b55fde6a51f"),
        working / "multiphase_development_5b55fde6a51f", "multiphase_status.json",
    )
    followup = link_directory(
        choose("follow-up results", discovered["followup"],
               "logit_evidence_followup_complete"),
        followup_target, "cub_replication_manifest.csv",
    )
    stage_cache = link_directory(
        choose("Phase 2 stage cache", discovered["stage_cache"], "phase2_stage_cache"),
        working / "phase2_stage_cache", "validation_report.json",
    )
    localizer_cache = link_directory(
        choose("corrected Phase 1B cache", discovered["localizer_cache"], "phase1b_corrected"),
        working / "phase1b_corrected" / "cache", "extraction_config.json",
    )
    phase4 = link_directory(
        choose("reviewed Phase 4 bundle", discovered["phase4"],
               "reviewed_phase4_db1219637723"),
        working / "reviewed_phase4_db1219637723", "phase4_run_report.json",
    )
    cub = choose("CUB dataset", discovered["cub"], "CUB_200_2011")

    result = {
        "status": "PASS",
        "bounded_search_max_depth": 5,
        "bulk_directories_pruned": sorted(PRUNED_DIRECTORIES),
        "run": str(run),
        "followup": str(followup),
        "stage_cache": str(stage_cache),
        "localizer_cache": str(localizer_cache),
        "phase4": str(phase4),
        "cub_root": str(cub),
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
