#!/usr/bin/env python3
"""Run Phase 3R and Phase 5 concurrently, then gate Phases 6 and 7."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def command(script: str, *arguments: object) -> list[str]:
    return [sys.executable, str(PROJECT / "scripts" / script), *map(str, arguments)]


def run(args: list[str], *, log_path: Path | None = None) -> None:
    print("$", " ".join(args), flush=True)
    if log_path is None:
        subprocess.run(args, cwd=PROJECT, check=True)
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("\n$ " + " ".join(args) + "\n")
        handle.flush()
        process = subprocess.Popen(
            args,
            cwd=PROJECT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            handle.write(line)
            handle.flush()
        returncode = process.wait()
        if returncode:
            raise subprocess.CalledProcessError(returncode, args)


def main() -> None:
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-cache", type=Path, required=True)
    parser.add_argument("--phase4-dir", type=Path, required=True)
    parser.add_argument("--cub-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--part-vocabulary", type=Path)
    parser.add_argument("--phase3-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--phase3-epochs", type=int, default=300)
    parser.add_argument("--phase7-max-per-outcome-per-attribute", type=int, default=5)
    args = parser.parse_args()
    if Path.cwd().resolve() != PROJECT.resolve():
        raise RuntimeError(f"run from {PROJECT}")
    if not args.stage_cache.is_dir() or not args.phase4_dir.is_dir() or not args.cub_root.is_dir():
        raise FileNotFoundError("stage cache, Phase 4 directory, and CUB root must exist")
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "p3": output / "phase3r",
        "p3patch": output / "phase3r_patch",
        "p5smoke": output / "phase5_smoke",
        "p5": output / "phase5",
        "p5analysis": output / "phase5_analysis",
        "p6": output / "phase6",
        "p7bridge": output / "phase7_bridge",
        "p7plan": output / "phase7_plan.json",
        "p7smoke": output / "phase7_smoke",
        "p7": output / "phase7",
        "p7analysis": output / "phase7_analysis.json",
        "p8freeze": output / "phase8_frozen_protocol.json",
        "paper": output / "paper_package",
    }
    env = dict(os.environ, PYTHONPATH="src", PYTHONUNBUFFERED="1")

    # Track A is CPU by default and starts first.  Track B owns the GPU, so the
    # two long computations overlap without loading two 7B checkpoints.
    phase3_log = output / "logs" / "phase3r.log"
    phase3_log.parent.mkdir(parents=True, exist_ok=True)
    phase3_handle = phase3_log.open("a", encoding="utf-8")
    phase3_args = command(
        "run_phase3_attribute_probes.py",
        "--cache-dir", args.stage_cache,
        "--output-dir", paths["p3"],
        "--device", args.phase3_device,
        "--epochs", args.phase3_epochs,
    )
    print("$", " ".join(phase3_args), "(background)", flush=True)
    phase3 = subprocess.Popen(
        phase3_args,
        cwd=PROJECT,
        env=env,
        stdout=phase3_handle,
        stderr=subprocess.STDOUT,
    )
    try:
        common_phase5 = [
            "--stage-cache", args.stage_cache,
            "--phase4-dir", args.phase4_dir,
            "--cub-root", args.cub_root,
        ]
        run(command(
            "extract_phase5_vqa.py", "--mode", "smoke", *common_phase5,
            "--output-dir", paths["p5smoke"],
        ), log_path=output / "logs" / "phase5_smoke.log")
        run(command(
            "extract_phase5_vqa.py", "--mode", "development", *common_phase5,
            "--smoke-dir", paths["p5smoke"], "--output-dir", paths["p5"],
        ), log_path=output / "logs" / "phase5.log")
        phase3_returncode = phase3.wait()
    finally:
        phase3_handle.close()
        if phase3.poll() is None:
            phase3.terminate()
            phase3.wait()
    if phase3_returncode:
        raise RuntimeError(f"Phase 3R failed; inspect {phase3_log}")

    patch_args: list[object] = [
        "--stage-cache", args.stage_cache,
        "--phase3-dir", paths["p3"],
        "--output-dir", paths["p3patch"],
        "--stages", "vision.late", "llm.final",
    ]
    if args.part_vocabulary is not None:
        patch_args.extend(("--part-vocabulary", args.part_vocabulary))
    run(command("export_phase3_patch_evidence.py", *patch_args))
    run(command(
        "run_phase5_utilization.py",
        "--records", paths["p5"] / "phase5_decisions.csv",
        "--output-dir", paths["p5analysis"],
    ))
    run(command(
        "run_phase6_joint_analysis.py",
        "--phase3-csv", paths["p3"] / "decision_probe_scores.csv",
        "--phase3-patch-csv", paths["p3patch"] / "probe_patch_evidence.csv",
        "--phase4-csv", args.phase4_dir / "attribute_metrics.csv",
        "--phase5-csv", paths["p5analysis"] / "decision_metrics.csv",
        "--output-dir", paths["p6"],
    ))
    transition = json.loads((paths["p6"] / "transition_decision.json").read_text())
    if transition["selected_transition"] == "mixed_or_null":
        run(command(
            "build_paper_results_package.py",
            "--phase3-dir", paths["p3"],
            "--phase4-dir", args.phase4_dir,
            "--phase4-review", PROJECT / "reports/development_20260911/phase4_review_decision.json",
            "--phase5-analysis-dir", paths["p5analysis"],
            "--phase6-dir", paths["p6"],
            "--output-dir", paths["paper"],
        ))
        status = {
            "schema_version": 1,
            "status": "PASS_PHASE7_BLOCKED_BY_PREDECLARED_GATE",
            "completed": ["phase3r", "phase3r_patch", "phase5", "phase5_analysis", "phase6"],
            "selected_transition": "mixed_or_null",
            "reason": "No single Phase 7 target is justified; inspect Phase 6 findings.",
            "paper_package": str(paths["paper"]),
            "elapsed_seconds": time.monotonic() - started,
            "official_test_images_used": 0,
        }
        (output / "multiphase_status.json").write_text(
            json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(status, indent=2), flush=True)
        return

    run(command(
        "export_phase7_token_metadata.py",
        "--stage-cache", args.stage_cache,
        "--phase3-dir", paths["p3"],
        "--phase6-dir", paths["p6"],
        "--output-dir", paths["p7bridge"],
        "--max-per-outcome-per-attribute", args.phase7_max_per_outcome_per_attribute,
    ))
    bridge = json.loads((paths["p7bridge"] / "token_metadata_report.json").read_text())
    run(command(
        "run_phase7_interventions.py", "plan",
        "--token-metadata", paths["p7bridge"] / "token_metadata.csv",
        "--output", paths["p7plan"],
        "--selected-stage", bridge["selected_stage"],
        "--neighbor-stage", bridge["neighbor_stage"],
        "--stage-order", *bridge["stage_order"],
        "--k", 32,
    ))
    common_phase7 = [
        "--phase5-dir", paths["p5"],
        "--phase7-bridge-dir", paths["p7bridge"],
        "--plan", paths["p7plan"],
        "--cub-root", args.cub_root,
    ]
    run(command(
        "extract_phase7_interventions.py", "--mode", "smoke", *common_phase7,
        "--output-dir", paths["p7smoke"],
    ), log_path=output / "logs" / "phase7_smoke.log")
    run(command(
        "extract_phase7_interventions.py", "--mode", "development", *common_phase7,
        "--smoke-dir", paths["p7smoke"], "--output-dir", paths["p7"],
    ), log_path=output / "logs" / "phase7.log")
    run(command(
        "run_phase7_interventions.py", "aggregate",
        "--outcomes", paths["p7"] / "intervention_outcomes.csv",
        "--plan", paths["p7plan"],
        "--output", paths["p7analysis"],
    ))
    run(command(
        "freeze_phase8_protocol.py",
        "--phase3-dir", paths["p3"],
        "--phase4-dir", args.phase4_dir,
        "--phase5-dir", paths["p5"],
        "--phase5-analysis-dir", paths["p5analysis"],
        "--phase6-dir", paths["p6"],
        "--phase7-bridge-dir", paths["p7bridge"],
        "--phase7-plan", paths["p7plan"],
        "--phase7-dir", paths["p7"],
        "--phase7-analysis", paths["p7analysis"],
        "--output", paths["p8freeze"],
    ))
    run(command(
        "build_paper_results_package.py",
        "--phase3-dir", paths["p3"],
        "--phase4-dir", args.phase4_dir,
        "--phase4-review", PROJECT / "reports/development_20260911/phase4_review_decision.json",
        "--phase5-analysis-dir", paths["p5analysis"],
        "--phase6-dir", paths["p6"],
        "--phase7-analysis", paths["p7analysis"],
        "--frozen-protocol", paths["p8freeze"],
        "--output-dir", paths["paper"],
    ))
    status = {
        "schema_version": 1,
        "status": "PASS",
        "completed": [
            "phase3r", "phase3r_patch", "phase5", "phase5_analysis",
            "phase6", "phase7_bridge", "phase7_interventions", "phase7_analysis",
        ],
        "selected_transition": transition["selected_transition"],
        "phase8_frozen_protocol": str(paths["p8freeze"]),
        "paper_package": str(paths["paper"]),
        "elapsed_seconds": time.monotonic() - started,
        "official_test_images_used": 0,
    }
    (output / "multiphase_status.json").write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(status, indent=2), flush=True)


if __name__ == "__main__":
    main()
