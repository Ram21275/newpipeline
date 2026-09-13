#!/usr/bin/env python3
"""Run the CUB-only development follow-up in resumable Kaggle stages.

This script is the compact counterpart of the detailed 31-cell runbook.  It
keeps every smoke and pilot gate, validates reports and artifact identities,
and prints the exact stage and command when a subprocess fails. CelebA is
deliberately excluded from this continuation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import queue
import shlex
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import traceback
from collections import deque
from pathlib import Path
from typing import Any, Iterable


PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parents[1]
PYTHON = sys.executable
BOOTSTRAP_ARGS = ("--bootstrap-samples", "10000", "--confidence-level", "0.95", "--seed", "20260913")
HEARTBEAT_SECONDS = 60.0
CUB_CONTROLS = ("image", "prompt_only", "image_shuffled", "opposite_label_image")
TIE_POLICY = "retain exact zero margins as abstentions and score incorrect"


class WorkflowError(RuntimeError):
    """A named workflow stage failed or violated an artifact invariant."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise WorkflowError(message)


def read_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing JSON artifact: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise WorkflowError(f"cannot read JSON artifact: {path}") from error
    require(isinstance(payload, dict), f"expected a JSON object: {path}")
    return payload


def read_csv(path: Path) -> list[dict[str, str]]:
    require(path.is_file(), f"missing CSV artifact: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    require(bool(rows), f"CSV artifact is empty: {path}")
    return rows


def validate_report(path: Path, **expected: Any) -> dict[str, Any]:
    report = read_json(path)
    require(report.get("status") == "PASS", f"report is not PASS: {path}")
    require(
        report.get("official_test_images_used", report.get("official_test_images", 0)) == 0,
        f"official-test use must remain zero: {path}",
    )
    for key, value in expected.items():
        require(report.get(key) == value, f"{path}: expected {key}={value!r}, got {report.get(key)!r}")
    return report


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_artifact_hashes(report_path: Path, report: dict[str, Any] | None = None) -> None:
    report = report or read_json(report_path)
    artifacts = report.get("artifacts", {})
    require(isinstance(artifacts, dict), f"artifact hash map is malformed: {report_path}")
    for relative, expected in artifacts.items():
        artifact = report_path.parent / str(relative)
        require(artifact.is_file(), f"hashed artifact is missing: {artifact}")
        require(sha256(artifact) == expected, f"artifact SHA-256 differs: {artifact}")


def validate_source_hashes(report_path: Path, expected: dict[str, Path]) -> None:
    report = read_json(report_path)
    source_hashes = report.get("source_hashes", {})
    require(isinstance(source_hashes, dict), f"source hash map is malformed: {report_path}")
    for label, source in expected.items():
        require(source.is_file(), f"analysis source is missing: {source}")
        require(source_hashes.get(label) == sha256(source),
                f"analysis source SHA-256 differs for {label}: {report_path}")


def validate_intervals(report_path: Path, report: dict[str, Any], fields: Iterable[str]) -> None:
    for field in fields:
        rows = report.get(field, [])
        require(isinstance(rows, list), f"{field} must be a list: {report_path}")
        for index, row in enumerate(rows):
            require(isinstance(row, dict), f"{field}[{index}] is malformed: {report_path}")
            try:
                estimate = float(row["estimate"])
                low = float(row["ci_low"])
                high = float(row["ci_high"])
            except (KeyError, TypeError, ValueError) as error:
                raise WorkflowError(f"{field}[{index}] lacks a numeric interval: {report_path}") from error
            require(all(math.isfinite(value) for value in (estimate, low, high)),
                    f"{field}[{index}] has a non-finite interval: {report_path}")
            require(low <= high, f"{field}[{index}] has a reversed interval: {report_path}")
            interpretation = row.get("interpretation")
            if low <= 0 <= high:
                require(interpretation == "null_compatible",
                        f"{field}[{index}] must be null-compatible: {report_path}")
            else:
                require(interpretation != "null_compatible",
                        f"{field}[{index}] incorrectly reports null compatibility: {report_path}")


def find_cub_root(input_root: Path) -> Path | None:
    for current, _directories, files in os.walk(input_root):
        path = Path(current)
        if path.name == "CUB_200_2011" and "images.txt" in files and (path / "images").is_dir():
            return path
    return None


def resolve_localizer_cache(phase1b_root: Path) -> Path:
    """Resolve the corrected Phase 1b cache without rerunning extraction."""
    candidates = (phase1b_root / "cache", phase1b_root)
    for candidate in candidates:
        if (
            (candidate / "extraction_config.json").is_file()
            and (candidate / "records").is_dir()
        ):
            return candidate
    expected = " or ".join(str(candidate) for candidate in candidates)
    raise WorkflowError(
        "missing corrected Phase 1b localizer cache; expected "
        f"extraction_config.json and records/ under {expected}"
    )


class Workflow:
    def __init__(self, working_root: Path, input_root: Path) -> None:
        self.working = working_root.resolve()
        self.inputs = input_root.resolve()
        self.run_root = self.working / "multiphase_development_5b55fde6a51f"
        self.cub_root: Path | None = None
        self._cub_searched = False

    def p(self, name: str) -> Path:
        return self.working / name

    def script(self, name: str, *arguments: object) -> list[str]:
        return [PYTHON, str(PROJECT / "scripts" / name), *(str(value) for value in arguments)]

    def run(self, stage: str, command: list[str], *, cwd: Path | None = None,
            env: dict[str, str] | None = None) -> None:
        command = [str(part) for part in command]
        print(f"\n===== {stage} =====", flush=True)
        print(shlex.join(command), flush=True)
        process_env = os.environ.copy()
        process_env.setdefault("PYTHONUNBUFFERED", "1")
        if env:
            process_env.update(env)
        process = subprocess.Popen(
            command, cwd=cwd, env=process_env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        output_tail: deque[str] = deque(maxlen=200)
        assert process.stdout is not None
        output_queue: queue.Queue[str | None] = queue.Queue()

        def relay_output() -> None:
            assert process.stdout is not None
            try:
                for line in process.stdout:
                    output_queue.put(line)
            finally:
                output_queue.put(None)

        reader = threading.Thread(target=relay_output, daemon=True)
        reader.start()
        started = time.monotonic()
        while True:
            try:
                line = output_queue.get(timeout=HEARTBEAT_SECONDS)
            except queue.Empty:
                elapsed_minutes = (time.monotonic() - started) / 60.0
                print(
                    f"[workflow heartbeat] {stage} is still running; "
                    f"elapsed={elapsed_minutes:.1f} min",
                    flush=True,
                )
                continue
            if line is None:
                break
            print(line, end="", flush=True)
            output_tail.append(line)
        reader.join()
        process.stdout.close()
        return_code = process.wait()
        if return_code != 0:
            detail_lines = [line.strip() for line in output_tail if line.strip()]
            detail = detail_lines[-1] if detail_lines else "no subprocess output"
            print(f"\nFAILED STAGE: {stage}", file=sys.stderr, flush=True)
            print(f"FAILED COMMAND: {shlex.join(command)}", file=sys.stderr, flush=True)
            print(f"FAILED DETAIL: {detail}", file=sys.stderr, flush=True)
            print("Fix the reported error, then rerun this same compact cell; completed outputs resume.",
                  file=sys.stderr, flush=True)
            error = WorkflowError(
                f"{stage} exited with status {return_code}: {detail}"
            )
            error.stage = stage
            error.failed_command = command
            error.output_tail = "".join(output_tail)
            raise error

    def require_cub(self) -> Path:
        if not self._cub_searched:
            print(f"Locating CUB_200_2011 below {self.inputs} ...", flush=True)
            self.cub_root = find_cub_root(self.inputs)
            self._cub_searched = True
        require(self.cub_root is not None, f"CUB_200_2011 was not found below {self.inputs}")
        print(f"CUB_ROOT {self.cub_root}", flush=True)
        return self.cub_root

    def inspect_storage(self) -> None:
        self.working.mkdir(parents=True, exist_ok=True)
        usage = shutil.disk_usage(self.working)
        gib = 1024 ** 3
        print(
            f"Storage {self.working}: used={usage.used / gib:.1f} GiB, "
            f"free={usage.free / gib:.1f} GiB, total={usage.total / gib:.1f} GiB"
        )
        retained = (
            self.run_root,
            self.p("phase1b_corrected"),
            self.p("phase2_stage_cache"),
            self.p("phase3_development_1a6e0c96681f_20260906T171249251219Z"),
            self.p("reviewed_phase4_db1219637723"),
        )
        for path in retained:
            print(f"retained {'OK' if path.exists() else 'MISSING'}: {path}")

    def validate_upstream(self) -> None:
        checks = {
            "multiphase": self.run_root / "multiphase_status.json",
            "phase3r": self.run_root / "phase3r" / "phase3_run_report.json",
            "phase5": self.run_root / "phase5" / "phase5_run_report.json",
            "phase6": self.run_root / "phase6" / "phase6_run_report.json",
            "phase7 bridge": self.run_root / "phase7_bridge" / "token_metadata_report.json",
        }
        for label, path in checks.items():
            validate_report(path)
            print(f"{label}: PASS, official-test images used=0")
        for relative in (
            "phase7_bridge/development_train_stage_means.safetensors",
            "phase5/decision_manifest.csv",
            "phase5/phase5_decisions.csv",
            "phase6/joint_decisions.csv",
        ):
            require((self.run_root / relative).is_file(), f"missing retained artifact: {relative}")

    @staticmethod
    def validate_replication_manifest(path: Path) -> list[dict[str, str]]:
        rows = read_csv(path)
        required = {
            "decision_id", "image_id", "attribute_id", "target", "prompt_text",
            "relative_path", "shuffled_image_id", "shuffled_relative_path",
            "opposite_label_image_id", "opposite_label_relative_path",
            "opposite_label_target", "opposite_label_seed",
        }
        require(all(required <= set(row) for row in rows),
                f"replication manifest lacks required columns: {path}")
        require(len({row["decision_id"] for row in rows}) == len(rows),
                f"decision IDs must be unique: {path}")
        require({int(row["target"]) for row in rows} == {0, 1}, f"both labels are required: {path}")
        require(all(int(row["opposite_label_target"]) == 1 - int(row["target"]) for row in rows),
                f"opposite-label controls are invalid: {path}")
        require(all(row["opposite_label_image_id"] != row["image_id"] for row in rows),
                f"opposite-label donors must use different images: {path}")
        require(all(row["opposite_label_relative_path"] != row["relative_path"] for row in rows),
                f"opposite-label donor paths must differ: {path}")
        require(all(row["shuffled_image_id"] != row["image_id"] for row in rows),
                f"shuffled controls must use different images: {path}")
        require(all(row["shuffled_relative_path"] != row["relative_path"] for row in rows),
                f"shuffled control paths must differ: {path}")
        require({int(row["opposite_label_seed"]) for row in rows} == {31415},
                f"opposite-label seed differs from the frozen plan: {path}")
        require(all(int(row.get("official_test_image", "0")) == 0 for row in rows),
                f"replication manifest contains official-test images: {path}")
        require(all(row.get("official_split", "train") == "train" for row in rows),
                f"replication manifest contains a non-training official split: {path}")
        print(
            f"manifest PASS: {len(rows)} unique decisions, both labels, "
            "shuffled and opposite-label donors verified"
        )
        return rows

    def validate_cub_preparation(self) -> tuple[dict[str, Any], list[dict[str, str]]]:
        selector_path = self.p("phase9_selector_metadata/selector_metadata_report.json")
        selector = validate_report(
            selector_path,
            purpose="phase9_full_cache_only_selector_metadata",
            development_only=True,
            phase8_protocol_unchanged=True,
            model_extraction_performed=False,
            selected_stage="vision.late",
            neighbor_stage="projector.output",
            selector_methods=["vision_cls_attention", "logit_concept"],
        )
        validate_artifact_hashes(selector_path, selector)

        plan_path = self.p("phase9_plan.json")
        plan = validate_report(
            plan_path,
            development_only=True,
            phase8_protocol_unchanged=True,
            selected_stage="vision.late",
            neighbor_stage="projector.output",
            selector_methods=["vision_cls_attention", "logit_concept"],
            selection_k=32,
            matched_random_seeds=[0, 1, 2],
            fixed_mask_across_stages=True,
            replacement="development_train_stage_mean",
        )
        plan_sources = plan.get("source_hashes")
        if plan_sources is not None:
            require(isinstance(plan_sources, dict)
                    and plan_sources.get("token_metadata") == sha256(
                        self.p("phase9_selector_metadata/selector_token_metadata.csv")
                    ), "Phase 9 plan token-metadata SHA-256 differs")
        require(selector.get("decisions") == plan.get("decision_count"),
                "selector export and intervention plan decision counts differ")
        records = plan.get("records")
        require(isinstance(records, list) and records, "Phase 9 plan records are missing")
        require(len(records) == plan.get("intervention_count"),
                "Phase 9 intervention count differs from its records")
        require(len({str(row["decision_id"]) for row in records}) == plan.get("decision_count"),
                "Phase 9 decision count differs from its records")
        require(len({str(row["image_id"]) for row in records}) == plan.get("image_count"),
                "Phase 9 image count differs from its records")
        require(len({str(row["intervention_id"]) for row in records}) == len(records),
                "Phase 9 intervention IDs are not unique")

        paired_masks: dict[tuple[str, str, str, int], dict[str, tuple[int, ...]]] = {}
        for row in records:
            key = (
                str(row["decision_id"]), str(row["selection_method"]),
                str(row["intervention_type"]), int(row["replicate"]),
            )
            paired_masks.setdefault(key, {})[str(row["stage"])] = tuple(
                int(index) for index in row["token_indices"]
            )
        expected_stages = {str(plan["selected_stage"]), str(plan["neighbor_stage"])}
        for key, stages in paired_masks.items():
            require(set(stages) == expected_stages, f"Phase 9 stage pair is incomplete: {key}")
            require(len(set(stages.values())) == 1,
                    f"Phase 9 mask changes across stages: {key}")

        manifest = self.validate_replication_manifest(self.p("cub_replication_manifest.csv"))
        print(
            "CUB PREPARATION PASS: selector metadata, fixed Phase 9 plan, and "
            f"four-condition manifest verified; decisions={len(manifest)}, official-test images used=0"
        )
        return plan, manifest

    @staticmethod
    def phase9_expected_counts(
        plan: dict[str, Any], mode: str, pilot_decisions: int = 20
    ) -> tuple[int, int]:
        decision_ids = sorted({str(row["decision_id"]) for row in plan["records"]})
        if mode == "smoke":
            chosen = set(decision_ids[:1])
        elif mode == "pilot":
            chosen = set(decision_ids[:pilot_decisions])
        else:
            chosen = set(decision_ids)
        interventions = sum(str(row["decision_id"]) in chosen for row in plan["records"])
        return len(chosen), interventions

    def phase9_stage_complete(self, mode: str, output: Path, plan: dict[str, Any]) -> bool:
        report_path = output / "phase9_extraction_report.json"
        if not report_path.is_file():
            return False
        report = validate_report(
            report_path, mode=mode, development_only=True, phase8_protocol_unchanged=True
        )
        expected_decisions, expected_interventions = self.phase9_expected_counts(plan, mode)
        require(report.get("decisions") == expected_decisions,
                f"completed Phase 9 {mode} decision count differs")
        require(report.get("interventions") == expected_interventions,
                f"completed Phase 9 {mode} intervention count differs")
        evaluation = read_json(output / "evaluation_config.json")
        require(evaluation.get("mode") == mode, f"Phase 9 {mode} evaluation identity differs")
        expected_sources = {
            "phase5_report_sha256": self.run_root / "phase5/phase5_run_report.json",
            "bridge_report_sha256": self.run_root / "phase7_bridge/token_metadata_report.json",
            "plan_sha256": self.p("phase9_plan.json"),
            "phase9_config_sha256": PROJECT / "configs/phase9_mechanism.json",
            "means_sha256": self.run_root / "phase7_bridge/development_train_stage_means.safetensors",
        }
        for field, source in expected_sources.items():
            require(source.is_file() and evaluation.get(field) == sha256(source),
                    f"Phase 9 {mode} {field} differs")
        outcomes = output / "intervention_outcomes.csv"
        require(outcomes.is_file(), f"Phase 9 {mode} outcomes are missing")
        require(report.get("outcomes_sha256") == sha256(outcomes),
                f"Phase 9 {mode} outcome hash differs")
        print(f"SKIP completed Phase 9 LLaVA {mode}: hashes and counts verified")
        return True

    def replication_stage_complete(
        self, model: str, mode: str, output: Path, config: Path,
        manifest: Path, pilot_decisions: int,
    ) -> bool:
        report_path = output / "replication_vqa_report.json"
        if not report_path.is_file():
            return False
        report = validate_report(report_path, mode=mode)
        expected_adapter = "llava" if model == "llava" else "qwen2_5_vl"
        require(report.get("adapter") == expected_adapter,
                f"completed CUB/{model}/{mode} adapter differs")
        manifest_rows = self.validate_replication_manifest(manifest)
        expected_decisions = 2 if mode == "smoke" else (
            min(pilot_decisions, len(manifest_rows)) if mode == "pilot" else len(manifest_rows)
        )
        require(report.get("decisions") == expected_decisions,
                f"completed CUB/{model}/{mode} decision count differs")
        require(report.get("control_rows") == len(CUB_CONTROLS) * expected_decisions,
                f"completed CUB/{model}/{mode} control count differs")
        evaluation = read_json(output / "evaluation_config.json")
        require(evaluation.get("mode") == mode, f"CUB/{model}/{mode} evaluation identity differs")
        require(evaluation.get("config_sha256") == sha256(config),
                f"CUB/{model}/{mode} config hash differs")
        require(evaluation.get("manifest_sha256") == sha256(manifest),
                f"CUB/{model}/{mode} manifest hash differs")
        validate_artifact_hashes(report_path, report)
        print(f"SKIP completed CUB {model.upper()} {mode}: hashes, counts, and controls verified")
        return True

    def analysis_complete(
        self, report_path: Path, sources: dict[str, Path], *, label: str
    ) -> bool:
        if not report_path.is_file():
            return False
        report = validate_report(report_path)
        recorded = report.get("source_hashes")
        if not isinstance(recorded, dict):
            print(f"REBUILD {label}: existing analysis predates source-hash auditing")
            return False
        expected = {name: sha256(path) for name, path in sources.items()}
        if any(recorded.get(name) != digest for name, digest in expected.items()):
            print(f"REBUILD {label}: source hashes changed")
            return False
        validate_artifact_hashes(report_path, report)
        print(f"SKIP completed {label}: source and artifact hashes verified")
        return True

    def prepare(self, *, skip_tests: bool) -> None:
        self.inspect_storage()
        self.require_cub()
        localizer_cache = resolve_localizer_cache(self.p("phase1b_corrected"))
        print(f"Phase 1b localizer cache: {localizer_cache}")
        required_scripts = (
            "export_phase9_full_selector_metadata.py",
            "extract_phase9_llava_interventions.py",
            "extract_replication_vqa.py",
            "analyze_replication_vqa.py",
            "plot_followup_results.py",
        )
        for name in required_scripts:
            require((PROJECT / "scripts" / name).is_file(), f"repository lacks entry point: {name}")
        self.run(
            "install LLaVA/test dependencies",
            [PYTHON, "-m", "pip", "install", "-r", str(PROJECT / "requirements-kaggle.txt")],
            cwd=PROJECT,
        )
        if not skip_tests:
            self.run(
                "CPU test suite",
                [PYTHON, "-m", "pytest", "-ra", "--tb=short"],
                cwd=PROJECT,
                env={"PYTHONPYCACHEPREFIX": "/tmp/lger_pycache"},
            )
        self.validate_upstream()
        self.run(
            "render existing audited paper figures",
            self.script(
                "plot_multiphase_paper_figures.py",
                "--paper-package", self.run_root / "paper_package",
                "--phase4-dir", self.p("reviewed_phase4_db1219637723"),
                "--output-dir", self.p("paper_figures_development"),
            ),
            env={"MPLCONFIGDIR": "/tmp/matplotlib"},
        )
        self.run(
            "export Phase 9 selector metadata",
            self.script(
                "export_phase9_full_selector_metadata.py",
                "--stage-cache", self.p("phase2_stage_cache"),
                "--localizer-cache", localizer_cache,
                "--phase6-dir", self.run_root / "phase6",
                "--output-dir", self.p("phase9_selector_metadata"),
            ),
        )
        selector = validate_report(self.p("phase9_selector_metadata/selector_metadata_report.json"))
        self.run(
            "build fixed Phase 9 intervention plan",
            self.script(
                "run_phase9_mechanism.py", "plan",
                "--token-metadata", self.p("phase9_selector_metadata/selector_token_metadata.csv"),
                "--output", self.p("phase9_plan.json"),
                "--selected-stage", "vision.late", "--neighbor-stage", "projector.output",
                "--selectors", "vision_cls_attention", "logit_concept",
                "--k", "32", "--random-seeds", "0", "1", "2",
            ),
        )
        plan = validate_report(self.p("phase9_plan.json"))
        require(plan.get("phase8_protocol_unchanged") is True, "Phase 9 plan changed the frozen protocol")
        require(plan.get("fixed_mask_across_stages") is True, "Phase 9 masks are not fixed across stages")
        require(plan.get("selector_methods") == ["vision_cls_attention", "logit_concept"],
                "Phase 9 selector set differs")
        require(plan.get("matched_random_seeds") == [0, 1, 2], "Phase 9 random seeds differ")
        require(selector.get("decisions") == plan.get("decision_count"),
                "selector export and intervention plan decision counts differ")
        self.run(
            "build CUB opposite-label controls",
            self.script(
                "build_opposite_label_controls.py",
                "--input", self.run_root / "phase5" / "decision_manifest.csv",
                "--output", self.p("cub_replication_manifest.csv"), "--seed", "31415",
            ),
        )
        self.validate_cub_preparation()
        print("\nPREPARE PASS. Run compact Cell 2 (LLaVA).")

    def run_phase9_llava(self) -> None:
        cub = self.require_cub()
        plan_path = self.p("phase9_plan.json")
        plan = validate_report(plan_path)
        modes = (
            ("smoke", self.p("phase9_llava_smoke"), []),
            ("pilot", self.p("phase9_llava_pilot"), [
                "--smoke-dir", self.p("phase9_llava_smoke"), "--pilot-decisions", 20,
            ]),
            ("full", self.p("phase9_llava_full"), [
                "--smoke-dir", self.p("phase9_llava_smoke"),
                "--pilot-dir", self.p("phase9_llava_pilot"), "--pilot-decisions", 20,
            ]),
        )
        for mode, output, extra in modes:
            if self.phase9_stage_complete(mode, output, plan):
                continue
            self.run(
                f"Phase 9 LLaVA {mode}",
                self.script(
                    "extract_phase9_llava_interventions.py",
                    "--mode", mode,
                    "--phase5-dir", self.run_root / "phase5",
                    "--phase7-bridge-dir", self.run_root / "phase7_bridge",
                    "--plan", plan_path, "--cub-root", cub,
                    *(str(value) for value in extra), "--output-dir", output,
                ),
            )
            report = validate_report(output / "phase9_extraction_report.json", mode=mode)
            if mode == "pilot":
                require(report.get("decisions") == 20, "Phase 9 pilot must contain 20 decisions")
            if mode == "full":
                require(report.get("decisions") == plan.get("decision_count"),
                        "Phase 9 full decision count differs from the plan")
                require(report.get("interventions") == plan.get("intervention_count"),
                        "Phase 9 full intervention count differs from the plan")
        outcomes = self.p("phase9_llava_full/intervention_outcomes.csv")
        analysis = self.p("phase9_llava_analysis.json")
        if not self.analysis_complete(
            analysis, {"outcomes": outcomes, "plan": plan_path}, label="Phase 9 analysis"
        ):
            self.run(
                "aggregate Phase 9 intervals",
                self.script(
                    "run_phase9_mechanism.py", "aggregate",
                    "--outcomes", outcomes,
                    "--plan", plan_path, "--output", analysis,
                    *BOOTSTRAP_ARGS,
                ),
            )
        validate_report(analysis)

    def run_replication(self, model: str, dataset: str, image_root: Path,
                        manifest: Path, pilot_decisions: int) -> None:
        config = PROJECT / "configs" / (
            "llava_full_replication.json" if model == "llava" else "qwen25vl_7b_replication.json"
        )
        outputs = {mode: self.p(f"{dataset}_{model}_{mode}") for mode in ("smoke", "pilot", "full")}
        for mode in ("smoke", "pilot", "full"):
            if self.replication_stage_complete(
                model, mode, outputs[mode], config, manifest, pilot_decisions
            ):
                continue
            extra: list[object] = []
            if mode in ("pilot", "full"):
                extra += ["--smoke-dir", outputs["smoke"], "--pilot-decisions", pilot_decisions]
            if mode == "full":
                extra += ["--pilot-dir", outputs["pilot"]]
            self.run(
                f"{dataset.upper()} {model.upper()} VQA {mode}",
                self.script(
                    "extract_replication_vqa.py", "--mode", mode,
                    "--config", config, "--manifest", manifest,
                    "--image-root", image_root, *extra, "--output-dir", outputs[mode],
                ),
            )
            report = validate_report(outputs[mode] / "replication_vqa_report.json", mode=mode)
            if mode == "smoke":
                require(report.get("decisions") == 2 and report.get("control_rows") == 8,
                        f"{dataset}/{model} smoke coverage differs")
            elif mode == "pilot":
                require(report.get("decisions") == pilot_decisions,
                        f"{dataset}/{model} pilot decision count differs")
                require(report.get("control_rows") == 4 * pilot_decisions,
                        f"{dataset}/{model} pilot control count differs")
            else:
                require(report.get("control_rows") == 4 * report.get("decisions", -1),
                        f"{dataset}/{model} full control count differs")

    def analyze_replication(self, dataset: str, labels: Iterable[str]) -> None:
        labels = list(labels)
        output_name = (
            f"{dataset}_{labels[0]}_analysis" if len(labels) == 1
            else f"{dataset}_cross_model_analysis"
        )
        sources = {
            f"input_{label}": self.p(f"{dataset}_{label}_full/replication_vqa_decisions.csv")
            for label in labels
        }
        report_path = self.p(f"{output_name}/replication_vqa_analysis.json")
        if self.analysis_complete(
            report_path, sources, label=f"{dataset.upper()} {'/'.join(labels)} analysis"
        ):
            return
        command = self.script("analyze_replication_vqa.py")
        for label in labels:
            command += ["--input", label, self.p(f"{dataset}_{label}_full/replication_vqa_decisions.csv")]
        command += ["--output-dir", self.p(output_name), *BOOTSTRAP_ARGS]
        self.run(f"analyze {dataset.upper()} {'/'.join(labels)} replication", command)
        validate_report(report_path)

    def create_archive(self, filename: str, members: Iterable[str]) -> Path:
        member_list = list(members)
        for member in member_list:
            require(self.p(member).exists(), f"archive member is missing: {self.p(member)}")
        archive = self.p(filename)
        self.run(f"create {filename}", ["tar", "-czf", str(archive), *member_list], cwd=self.working)
        base_name = archive.name[:-7] if archive.name.endswith(".tar.gz") else archive.stem
        checksum_path = archive.parent / f"{base_name}.sha256"
        archive_hash = sha256(archive)
        checksum_path.write_text(f"{archive_hash}  {archive.name}\n", encoding="utf-8")
        print(f"SHA-256 {archive_hash}  {archive}")
        return archive

    def verify_archive(self, filename: str) -> str:
        archive = self.p(filename)
        base_name = archive.name[:-7] if archive.name.endswith(".tar.gz") else archive.stem
        checksum_path = archive.parent / f"{base_name}.sha256"
        require(archive.is_file() and checksum_path.is_file(),
                f"archive or SHA-256 sidecar is missing: {archive}")
        fields = checksum_path.read_text(encoding="utf-8").split()
        require(len(fields) >= 2 and fields[1] == archive.name,
                f"malformed SHA-256 sidecar: {checksum_path}")
        actual = sha256(archive)
        require(fields[0] == actual, f"archive SHA-256 differs from its sidecar: {archive}")
        return actual

    def audit_archive(self, filename: str, required_members: Iterable[str]) -> str:
        actual = self.verify_archive(filename)
        archive = self.p(filename)
        with tarfile.open(archive, "r:gz") as handle:
            members = {member.name.removeprefix("./") for member in handle.getmembers()}
        require(not any("celeba" in member.lower() for member in members),
                f"CUB-only archive contains a CelebA member: {archive}")
        for required_member in required_members:
            require(any(member == required_member or member.startswith(required_member + "/")
                        for member in members),
                    f"archive member is missing: {required_member}")
        print(f"PASS archive SHA-256 and CUB-only members: {actual}  {archive.name}")
        return actual

    def audit_replication_stage(
        self, model: str, mode: str, manifest: list[dict[str, str]]
    ) -> dict[str, Any]:
        output = self.p(f"cub_{model}_{mode}")
        report_path = output / "replication_vqa_report.json"
        report = validate_report(report_path, mode=mode)
        config_path = PROJECT / "configs" / (
            "llava_full_replication.json" if model == "llava" else "qwen25vl_7b_replication.json"
        )
        config = read_json(config_path)
        expected_adapter = "llava" if model == "llava" else "qwen2_5_vl"
        require(
            report.get("adapter") == expected_adapter
            and report.get("model") == config.get("model")
            and report.get("resolved_revision") == config.get("revision"),
            f"CUB/{model}/{mode} model identity differs from the frozen config",
        )
        evaluation = read_json(output / "evaluation_config.json")
        require(
            evaluation.get("config_sha256") == sha256(config_path)
            and evaluation.get("manifest_sha256") == sha256(self.p("cub_replication_manifest.csv")),
            f"CUB/{model}/{mode} input hashes differ from the frozen inputs",
        )
        validate_artifact_hashes(report_path, report)
        rows = read_csv(output / "replication_vqa_decisions.csv")
        require(len(rows) == report.get("control_rows"),
                f"CUB/{model}/{mode} CSV and report row counts differ")
        by_decision: dict[str, list[dict[str, str]]] = {}
        for row in rows:
            by_decision.setdefault(row["decision_id"], []).append(row)
            tie = int(row["margin_tie"])
            correct = int(row["margin_correct"])
            margin = float(row["answer_margin"])
            likelihoods = (
                float(row["positive_log_likelihood"]),
                float(row["negative_log_likelihood"]),
                float(row["correct_answer_margin"]),
                margin,
            )
            require(all(math.isfinite(value) for value in likelihoods),
                    f"CUB/{model}/{mode} contains a non-finite likelihood or margin")
            require(tie in (0, 1) and correct in (0, 1),
                    f"CUB/{model}/{mode} has malformed tie/correct flags")
            require(tie == int(margin == 0.0),
                    f"CUB/{model}/{mode} tie flag differs from the exact margin")
            require(not tie or (correct == 0 and row.get("margin_prediction", "") == ""),
                    f"CUB/{model}/{mode} exact tie was not retained as an incorrect abstention")
        require(len(by_decision) == report.get("decisions"),
                f"CUB/{model}/{mode} CSV and report decision counts differ")
        require(all({row["control"] for row in decision_rows} == set(CUB_CONTROLS)
                    and len(decision_rows) == len(CUB_CONTROLS)
                    for decision_rows in by_decision.values()),
                f"CUB/{model}/{mode} does not contain exactly four controls per decision")
        manifest_by_id = {row["decision_id"]: row for row in manifest}
        for decision_id, decision_rows in by_decision.items():
            source = manifest_by_id.get(decision_id)
            require(source is not None,
                    f"CUB/{model}/{mode} contains a decision outside the frozen manifest")
            evaluated = {row["control"]: row["evaluated_image_id"] for row in decision_rows}
            require(evaluated == {
                "image": source["image_id"],
                "prompt_only": "",
                "image_shuffled": source["shuffled_image_id"],
                "opposite_label_image": source["opposite_label_image_id"],
            }, f"CUB/{model}/{mode}/{decision_id} control image identities differ")
        computed_ties = {
            control: sum(int(row["margin_tie"]) for row in rows if row["control"] == control)
            for control in CUB_CONTROLS
        }
        require(report.get("margin_ties_by_control") == computed_ties,
                f"CUB/{model}/{mode} tie counts differ from the CSV")
        manifest_ids = {row["decision_id"] for row in manifest}
        require(set(by_decision) <= manifest_ids,
                f"CUB/{model}/{mode} contains decisions outside the frozen manifest")
        computed_targets: dict[str, int] = {}
        for decision_rows in by_decision.values():
            target = str(int(decision_rows[0]["target"]))
            computed_targets[target] = computed_targets.get(target, 0) + 1
        require(report.get("target_counts") == computed_targets,
                f"CUB/{model}/{mode} target counts differ from the CSV")
        positive_ids = tuple(int(value) for value in report.get("positive_token_ids", []))
        negative_ids = tuple(int(value) for value in report.get("negative_token_ids", []))
        require(positive_ids and negative_ids and positive_ids != negative_ids,
                f"CUB/{model}/{mode} answer token identities are missing or identical")
        print(
            f"PASS CUB {model.upper()} {mode}: decisions={len(by_decision)}, "
            f"controls={len(rows)}, ties={sum(computed_ties.values())}, official-test images used=0"
        )
        return report

    def audit_replication_analysis(
        self, relative: str, labels: list[str], manifest_count: int
    ) -> dict[str, Any]:
        report_path = self.p(relative) / "replication_vqa_analysis.json"
        report = validate_report(
            report_path,
            model_labels=labels,
            bootstrap_unit="image_id",
            bootstrap_samples=10000,
            confidence_level=0.95,
            tie_policy=TIE_POLICY,
        )
        sources = {
            f"input_{label}": self.p(f"cub_{label}_full/replication_vqa_decisions.csv")
            for label in labels
        }
        validate_source_hashes(report_path, sources)
        validate_artifact_hashes(report_path, report)
        summaries = report.get("summaries")
        require(isinstance(summaries, list), f"analysis summaries are malformed: {report_path}")
        require({(row.get("model_label"), row.get("control")) for row in summaries}
                == {(label, control) for label in labels for control in CUB_CONTROLS},
                f"analysis summaries do not cover every model/control: {report_path}")
        require(all(row.get("decisions") == manifest_count for row in summaries),
                f"analysis decision counts differ from the manifest: {report_path}")
        validate_intervals(report_path, report, ("paired_contrasts", "cross_model_contrasts"))
        expected_cross_model = 3 if len(labels) == 2 else 0
        require(len(report.get("cross_model_contrasts", [])) == expected_cross_model,
                f"cross-model contrast count differs: {report_path}")
        print(
            f"PASS {relative}: image-clustered 95% intervals, conservative ties, "
            "all controls and source/artifact hashes verified"
        )
        return report

    def llava(self) -> None:
        cub = self.require_cub()
        self.validate_cub_preparation()
        self.run_phase9_llava()
        self.run_replication("llava", "cub", cub / "images", self.p("cub_replication_manifest.csv"), 40)
        self.analyze_replication("cub", ["llava"])
        archive_members = [
            "phase9_selector_metadata", "phase9_plan.json",
            "phase9_llava_smoke", "phase9_llava_pilot", "phase9_llava_full",
            "phase9_llava_analysis.json", "cub_replication_manifest.csv",
            "cub_llava_smoke", "cub_llava_pilot", "cub_llava_full", "cub_llava_analysis",
            "paper_figures_development",
        ]
        self.create_archive("llava_phase9_and_cub_results.tar.gz", archive_members)
        print("\nLLAVA PASS. Download the LLaVA archive as a checkpoint, then run compact Cell 3 (Qwen).")

    def clear_llava_checkpoint(self) -> None:
        targets = (
            Path("/root/.cache/huggingface/hub/models--llava-hf--llava-1.5-7b-hf"),
            self.p("hf_cache/hub/models--llava-hf--llava-1.5-7b-hf"),
        )
        require(self.p("llava_phase9_and_cub_results.tar.gz").is_file(),
                "refusing to clear LLaVA before its results archive exists")
        manifest = self.validate_replication_manifest(self.p("cub_replication_manifest.csv"))
        for mode in ("smoke", "pilot", "full"):
            self.audit_replication_stage("llava", mode, manifest)
        self.audit_replication_analysis("cub_llava_analysis", ["llava"], len(manifest))
        self.audit_archive("llava_phase9_and_cub_results.tar.gz", (
            "phase9_selector_metadata", "phase9_plan.json", "phase9_llava_full",
            "phase9_llava_analysis.json", "cub_replication_manifest.csv", "cub_llava_full",
            "cub_llava_analysis",
        ))
        for path in targets:
            if path.exists():
                print(f"removing checkpoint cache: {path}")
                shutil.rmtree(path)

    def qwen(self, *, clear_llava: bool) -> None:
        cub = self.require_cub()
        self.validate_cub_preparation()
        if clear_llava:
            self.clear_llava_checkpoint()
        self.run(
            "install pinned Qwen dependencies",
            [PYTHON, "-m", "pip", "install", "-r", str(PROJECT / "requirements-qwen-kaggle.txt")],
            cwd=PROJECT,
        )
        self.run(
            "verify pinned Transformers version",
            [PYTHON, "-c", "import transformers; assert transformers.__version__ == '4.51.3', transformers.__version__; print('transformers', transformers.__version__)"],
        )
        manifest = self.p("cub_replication_manifest.csv")
        self.validate_replication_manifest(manifest)
        self.run_replication("qwen", "cub", cub / "images", manifest, 40)
        self.analyze_replication("cub", ["llava", "qwen"])
        self.run(
            "plot Phase 9 and CUB cross-model intervals",
            self.script(
                "plot_followup_results.py",
                "--phase9-analysis", self.p("phase9_llava_analysis.json"),
                "--replication-analysis", self.p("cub_cross_model_analysis/replication_vqa_analysis.json"),
                "--output-dir", self.p("followup_paper_figures"),
            ),
            env={"MPLCONFIGDIR": "/tmp/matplotlib"},
        )
        core_members = [
            "phase9_selector_metadata", "phase9_plan.json", "phase9_llava_smoke",
            "phase9_llava_pilot", "phase9_llava_full", "phase9_llava_analysis.json",
            "cub_replication_manifest.csv", "cub_llava_smoke", "cub_llava_pilot",
            "cub_llava_full", "cub_llava_analysis", "cub_qwen_smoke", "cub_qwen_pilot",
            "cub_qwen_full", "cub_cross_model_analysis", "paper_figures_development",
            "followup_paper_figures",
        ]
        self.create_archive("logit_evidence_followup_complete.tar.gz", core_members)
        print("\nQWEN PASS. Run compact Cell 4 for the final audit and download paths.")

    def status(self) -> None:
        plan, manifest = self.validate_cub_preparation()

        phase9_reports = {
            mode: validate_report(
                self.p(f"phase9_llava_{mode}/phase9_extraction_report.json"),
                mode=mode, development_only=True, phase8_protocol_unchanged=True,
            )
            for mode in ("smoke", "pilot", "full")
        }
        for mode, report in phase9_reports.items():
            expected_decisions, expected_interventions = self.phase9_expected_counts(plan, mode)
            require(report.get("decisions") == expected_decisions,
                    f"Phase 9 {mode} decision count differs")
            require(report.get("interventions") == expected_interventions,
                    f"Phase 9 {mode} intervention count differs")
            outcomes = self.p(f"phase9_llava_{mode}/intervention_outcomes.csv")
            require(outcomes.is_file(), f"Phase 9 {mode} outcomes are missing")
            require(report.get("outcomes_sha256") == sha256(outcomes),
                    f"Phase 9 {mode} outcome SHA-256 differs")
            print(
                f"PASS Phase 9 {mode}: decisions={expected_decisions}, "
                f"interventions={expected_interventions}, outcome hash verified"
            )
        require(len({report["policy_digest"] for report in phase9_reports.values()}) == 1,
                "Phase 9 smoke/pilot/full policy digests differ")

        phase9_analysis_path = self.p("phase9_llava_analysis.json")
        phase9_analysis = validate_report(
            phase9_analysis_path,
            development_only=True,
            phase8_protocol_unchanged=True,
            bootstrap_unit="image_id",
            bootstrap_samples=10000,
            confidence_level=0.95,
        )
        validate_source_hashes(phase9_analysis_path, {
            "outcomes": self.p("phase9_llava_full/intervention_outcomes.csv"),
            "plan": self.p("phase9_plan.json"),
        })
        require(phase9_analysis.get("decision_count") == plan.get("decision_count"),
                "Phase 9 analysis decision count differs from the plan")
        require(phase9_analysis.get("outcome_count") == plan.get("intervention_count"),
                "Phase 9 analysis outcome count differs from the plan")
        validate_intervals(phase9_analysis_path, phase9_analysis, (
            "selector_specific_contrasts", "stage_interactions", "selector_interactions",
            "failure_success_interactions", "global_manipulation_checks",
        ))
        print(
            "PASS Phase 9 analysis: image-clustered 95% intervals, source hashes, "
            "frozen protocol, and outcome counts verified"
        )

        stage_reports: dict[tuple[str, str], dict[str, Any]] = {}
        for model in ("llava", "qwen"):
            for mode in ("smoke", "pilot", "full"):
                stage_reports[(model, mode)] = self.audit_replication_stage(model, mode, manifest)
            require(len({stage_reports[(model, mode)]["policy_digest"]
                         for mode in ("smoke", "pilot", "full")}) == 1,
                    f"CUB {model} smoke/pilot/full policy digests differ")

        self.audit_replication_analysis("cub_llava_analysis", ["llava"], len(manifest))
        self.audit_replication_analysis(
            "cub_cross_model_analysis", ["llava", "qwen"], len(manifest)
        )
        self.audit_archive("llava_phase9_and_cub_results.tar.gz", (
            "phase9_selector_metadata", "phase9_plan.json", "phase9_llava_full",
            "phase9_llava_analysis.json", "cub_replication_manifest.csv", "cub_llava_full",
            "cub_llava_analysis",
        ))
        self.audit_archive("logit_evidence_followup_complete.tar.gz", (
            "phase9_selector_metadata", "phase9_plan.json", "phase9_llava_full",
            "phase9_llava_analysis.json", "cub_replication_manifest.csv", "cub_llava_full",
            "cub_qwen_full", "cub_cross_model_analysis", "followup_paper_figures",
        ))

        archive = self.p("logit_evidence_followup_complete.tar.gz")
        checksum = self.p("logit_evidence_followup_complete.sha256")
        print("AUDIT PASS: CUB-only outputs are internally consistent; official-test image use=0")
        print(
            "CLAIM BOUNDARY: attention/localization is diagnostic, probes establish accessibility, "
            "and only controlled interventions support causal claims. Null or mixed intervals remain "
            "null-compatible and do not establish absence."
        )
        print(f"DOWNLOAD {archive}")
        print(f"DOWNLOAD {checksum}")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--working-root", type=Path, default=Path("/kaggle/working"))
    result.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    result.add_argument(
        "--celeba", choices=("no",), default="no",
        help="compatibility flag; this continuation is CUB-only and requires 'no'",
    )
    result.add_argument(
        "--notebook-safe", action="store_true",
        help="print and save failures without replacing useful output with an IPython wrapper",
    )
    commands = result.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="install, test, validate caches, and build plans")
    prepare.add_argument("--skip-tests", action="store_true")
    commands.add_parser("preflight", help="verify retained CUB preparation artifacts without a model")
    commands.add_parser("llava", help="run gated Phase 9 and LLaVA replications")
    qwen = commands.add_parser("qwen", help="run gated Qwen replications and package outputs")
    qwen.add_argument("--clear-llava-checkpoint", action="store_true")
    commands.add_parser("status", help="verify final reports, official-test counts, and archive hash")
    return result


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    workflow = Workflow(args.working_root, args.input_root)
    if args.command == "prepare":
        workflow.prepare(skip_tests=args.skip_tests)
    elif args.command == "preflight":
        workflow.validate_cub_preparation()
    elif args.command == "llava":
        workflow.llava()
    elif args.command == "qwen":
        workflow.qwen(clear_llava=args.clear_llava_checkpoint)
    else:
        workflow.status()


def notebook_failure_path(argv: list[str]) -> Path:
    working_root = Path("/kaggle/working")
    for index, argument in enumerate(argv):
        if argument == "--working-root" and index + 1 < len(argv):
            working_root = Path(argv[index + 1])
        elif argument.startswith("--working-root="):
            working_root = Path(argument.split("=", 1)[1])
    return working_root / "lger_compact_last_failure.json"


if __name__ == "__main__":
    notebook_safe = "--notebook-safe" in sys.argv[1:]
    failure_path = notebook_failure_path(sys.argv[1:])
    try:
        main()
    except Exception as error:
        if not isinstance(error, WorkflowError) or error.__cause__ is None:
            traceback.print_exc()
        print(f"\nCOMPACT WORKFLOW FAILED: {error}", file=sys.stderr, flush=True)
        if notebook_safe:
            failure_path.parent.mkdir(parents=True, exist_ok=True)
            failure_path.write_text(json.dumps({
                "status": "FAILED",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "command": sys.argv[1:],
                "failed_stage": getattr(error, "stage", None),
                "failed_command": getattr(error, "failed_command", None),
                "subprocess_output_tail": getattr(error, "output_tail", ""),
            }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(f"DIAGNOSTIC SAVED: {failure_path}", file=sys.stderr, flush=True)
            print("STOP HERE. Do not run the next compact cell.", file=sys.stderr, flush=True)
        else:
            raise SystemExit(1) from error
    else:
        if notebook_safe:
            failure_path.unlink(missing_ok=True)
