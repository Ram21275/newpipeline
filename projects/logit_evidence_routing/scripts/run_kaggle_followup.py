#!/usr/bin/env python3
"""Run the development-only Kaggle follow-up in four resumable stages.

This script is the compact counterpart of the detailed 31-cell runbook.  It
keeps every smoke and pilot gate, validates every JSON report, and prints the
exact stage and command when a subprocess fails.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import traceback
from collections import deque
from pathlib import Path
from typing import Any, Iterable


PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parents[1]
PYTHON = sys.executable
BOOTSTRAP_ARGS = ("--bootstrap-samples", "10000", "--confidence-level", "0.95", "--seed", "20260913")


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


def find_cub_root(input_root: Path) -> Path | None:
    for current, _directories, files in os.walk(input_root):
        path = Path(current)
        if path.name == "CUB_200_2011" and "images.txt" in files and (path / "images").is_dir():
            return path
    return None


def find_celeba_root(input_root: Path) -> Path | None:
    for current, directories, _files in os.walk(input_root):
        path = Path(current)
        if path.name == "Img" and "img_celeba" in directories:
            candidate = path.parent
            if (candidate / "Img" / "img_celeba").is_dir():
                return candidate
    return None


class Workflow:
    def __init__(self, working_root: Path, input_root: Path, celeba: str) -> None:
        self.working = working_root.resolve()
        self.inputs = input_root.resolve()
        self.celeba_policy = celeba
        self.run_root = self.working / "multiphase_development_5b55fde6a51f"
        self.cub_root = find_cub_root(self.inputs)
        self.celeba_root = find_celeba_root(self.inputs)

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
        if env:
            process_env.update(env)
        process = subprocess.Popen(
            command, cwd=cwd, env=process_env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        output_tail: deque[str] = deque(maxlen=200)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            output_tail.append(line)
        process.stdout.close()
        return_code = process.wait()
        if return_code != 0:
            print(f"\nFAILED STAGE: {stage}", file=sys.stderr, flush=True)
            print(f"FAILED COMMAND: {shlex.join(command)}", file=sys.stderr, flush=True)
            print("Fix the reported error, then rerun this same compact cell; completed outputs resume.",
                  file=sys.stderr, flush=True)
            error = WorkflowError(f"{stage} exited with status {return_code}")
            error.stage = stage
            error.failed_command = command
            error.output_tail = "".join(output_tail)
            raise error

    def require_cub(self) -> Path:
        require(self.cub_root is not None, f"CUB_200_2011 was not found below {self.inputs}")
        return self.cub_root

    def should_run_celeba(self) -> bool:
        if self.celeba_policy == "no":
            return False
        if self.celeba_policy == "yes":
            require(self.celeba_root is not None, f"CelebA in-the-wild layout was not found below {self.inputs}")
            return True
        return self.celeba_root is not None

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

    def prepare_celeba(self) -> None:
        if not self.should_run_celeba():
            print("CelebA: not attached; optional replication will be skipped.")
            return
        assert self.celeba_root is not None
        output = self.p("celeba_replication_development")
        self.run(
            "prepare optional CelebA development split",
            self.script(
                "prepare_celeba_replication.py",
                "--celeba-root", self.celeba_root,
                "--config", PROJECT / "configs" / "celeba_replication.json",
                "--output-dir", output,
            ),
        )
        source = output / "celeba_development_decisions.csv"
        rows = [row for row in read_csv(source) if row["development_split"] == "val"]
        require(rows and all(row["official_test_image"] == "0" for row in rows),
                "CelebA validation rows must exclude the official test split")
        target = self.p("celeba_validation_decisions.csv")
        temporary = target.with_suffix(".csv.tmp")
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(target)
        self.run(
            "build CelebA opposite-label controls",
            self.script(
                "build_opposite_label_controls.py", "--input", target,
                "--output", self.p("celeba_replication_manifest.csv"), "--seed", "31415",
            ),
        )
        self.validate_replication_manifest(self.p("celeba_replication_manifest.csv"))

    @staticmethod
    def validate_replication_manifest(path: Path) -> None:
        rows = read_csv(path)
        require({int(row["target"]) for row in rows} == {0, 1}, f"both labels are required: {path}")
        require(all(int(row["opposite_label_target"]) == 1 - int(row["target"]) for row in rows),
                f"opposite-label controls are invalid: {path}")
        require(all(row["opposite_label_image_id"] != row["image_id"] for row in rows),
                f"opposite-label donors must use different images: {path}")
        print(f"manifest PASS: {len(rows)} decisions, both labels, opposite-label donors verified")

    def prepare(self, *, skip_tests: bool) -> None:
        self.inspect_storage()
        self.require_cub()
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
                "--localizer-cache", self.p("phase1b_corrected"),
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
        self.validate_replication_manifest(self.p("cub_replication_manifest.csv"))
        self.prepare_celeba()
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
        self.run(
            "aggregate Phase 9 intervals",
            self.script(
                "run_phase9_mechanism.py", "aggregate",
                "--outcomes", self.p("phase9_llava_full/intervention_outcomes.csv"),
                "--plan", plan_path, "--output", self.p("phase9_llava_analysis.json"),
                *BOOTSTRAP_ARGS,
            ),
        )
        validate_report(self.p("phase9_llava_analysis.json"))

    def run_replication(self, model: str, dataset: str, image_root: Path,
                        manifest: Path, pilot_decisions: int) -> None:
        config = PROJECT / "configs" / (
            "llava_full_replication.json" if model == "llava" else "qwen25vl_7b_replication.json"
        )
        outputs = {mode: self.p(f"{dataset}_{model}_{mode}") for mode in ("smoke", "pilot", "full")}
        for mode in ("smoke", "pilot", "full"):
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
        command = self.script("analyze_replication_vqa.py")
        for label in labels:
            command += ["--input", label, self.p(f"{dataset}_{label}_full/replication_vqa_decisions.csv")]
        command += ["--output-dir", self.p(output_name), *BOOTSTRAP_ARGS]
        self.run(f"analyze {dataset.upper()} {'/'.join(labels)} replication", command)
        validate_report(self.p(f"{output_name}/replication_vqa_analysis.json"))

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

    def llava(self) -> None:
        cub = self.require_cub()
        validate_report(self.p("phase9_selector_metadata/selector_metadata_report.json"))
        self.validate_replication_manifest(self.p("cub_replication_manifest.csv"))
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
        if self.should_run_celeba():
            assert self.celeba_root is not None
            manifest = self.p("celeba_replication_manifest.csv")
            self.validate_replication_manifest(manifest)
            self.run_replication("llava", "celeba", self.celeba_root, manifest, 72)
            archive_members += [
                "celeba_replication_development", "celeba_validation_decisions.csv",
                "celeba_replication_manifest.csv", "celeba_llava_smoke",
                "celeba_llava_pilot", "celeba_llava_full",
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
        validate_report(self.p("cub_llava_analysis/replication_vqa_analysis.json"))
        self.verify_archive("llava_phase9_and_cub_results.tar.gz")
        for path in targets:
            if path.exists():
                print(f"removing checkpoint cache: {path}")
                shutil.rmtree(path)

    def qwen(self, *, clear_llava: bool) -> None:
        cub = self.require_cub()
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
        optional_members: list[str] = []
        if self.should_run_celeba():
            assert self.celeba_root is not None
            celeba_manifest = self.p("celeba_replication_manifest.csv")
            self.validate_replication_manifest(celeba_manifest)
            self.run_replication("qwen", "celeba", self.celeba_root, celeba_manifest, 72)
            self.analyze_replication("celeba", ["llava", "qwen"])
            self.run(
                "plot CelebA cross-model intervals",
                self.script(
                    "plot_followup_results.py",
                    "--phase9-analysis", self.p("phase9_llava_analysis.json"),
                    "--replication-analysis", self.p("celeba_cross_model_analysis/replication_vqa_analysis.json"),
                    "--output-dir", self.p("celeba_followup_figures"),
                ),
                env={"MPLCONFIGDIR": "/tmp/matplotlib"},
            )
            optional_members = [
                "celeba_replication_development", "celeba_validation_decisions.csv",
                "celeba_replication_manifest.csv", "celeba_llava_smoke", "celeba_llava_pilot",
                "celeba_llava_full", "celeba_qwen_smoke", "celeba_qwen_pilot", "celeba_qwen_full",
                "celeba_cross_model_analysis", "celeba_followup_figures",
            ]
            self.create_archive("celeba_cross_model_development_results.tar.gz", optional_members)
        core_members = [
            "phase9_selector_metadata", "phase9_plan.json", "phase9_llava_smoke",
            "phase9_llava_pilot", "phase9_llava_full", "phase9_llava_analysis.json",
            "cub_replication_manifest.csv", "cub_llava_smoke", "cub_llava_pilot",
            "cub_llava_full", "cub_llava_analysis", "cub_qwen_smoke", "cub_qwen_pilot",
            "cub_qwen_full", "cub_cross_model_analysis", "paper_figures_development",
            "followup_paper_figures", *optional_members,
        ]
        self.create_archive("logit_evidence_followup_complete.tar.gz", core_members)
        print("\nQWEN PASS. Run compact Cell 4 for the final audit and download paths.")

    def status(self) -> None:
        checks = (
            ("selector metadata", self.p("phase9_selector_metadata/selector_metadata_report.json")),
            ("Phase 9 smoke", self.p("phase9_llava_smoke/phase9_extraction_report.json")),
            ("Phase 9 pilot", self.p("phase9_llava_pilot/phase9_extraction_report.json")),
            ("Phase 9 full", self.p("phase9_llava_full/phase9_extraction_report.json")),
            ("Phase 9 analysis", self.p("phase9_llava_analysis.json")),
            ("CUB LLaVA full", self.p("cub_llava_full/replication_vqa_report.json")),
            ("CUB Qwen full", self.p("cub_qwen_full/replication_vqa_report.json")),
            ("CUB cross-model analysis", self.p("cub_cross_model_analysis/replication_vqa_analysis.json")),
        )
        for label, path in checks:
            report = validate_report(path)
            details = [f"{key}={report[key]}" for key in ("mode", "decisions", "interventions", "control_rows") if key in report]
            print(f"PASS {label}: {', '.join(details) if details else path.name}")
        archive = self.p("logit_evidence_followup_complete.tar.gz")
        checksum = self.p("logit_evidence_followup_complete.sha256")
        actual = self.verify_archive(archive.name)
        print(f"PASS archive SHA-256: {actual}")
        print(f"DOWNLOAD {archive}")
        print(f"DOWNLOAD {checksum}")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--working-root", type=Path, default=Path("/kaggle/working"))
    result.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    result.add_argument("--celeba", choices=("auto", "yes", "no"), default="auto",
                        help="auto runs CelebA only when its in-the-wild layout is attached")
    result.add_argument(
        "--notebook-safe", action="store_true",
        help="print and save failures without replacing useful output with an IPython wrapper",
    )
    commands = result.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="install, test, validate caches, and build plans")
    prepare.add_argument("--skip-tests", action="store_true")
    commands.add_parser("llava", help="run gated Phase 9 and LLaVA replications")
    qwen = commands.add_parser("qwen", help="run gated Qwen replications and package outputs")
    qwen.add_argument("--clear-llava-checkpoint", action="store_true")
    commands.add_parser("status", help="verify final reports, official-test counts, and archive hash")
    return result


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    workflow = Workflow(args.working_root, args.input_root, args.celeba)
    if args.command == "prepare":
        workflow.prepare(skip_tests=args.skip_tests)
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
