"""Command line entry point for the fresh-session Kaggle workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .analysis import analyze_per_example
from .archive import build_archive
from .core import atomic_write_json, environment_inventory
from .data import (
    HISTORICAL_ATTRIBUTE_IDS,
    audit_cub,
    build_manifests,
    discover_under_kaggle,
)
from .figures import render_final_figures
from .models import (
    LLAVA_CHECKPOINT,
    LLAVA_REVISION,
    QWEN3_CHECKPOINT,
    QWEN3_REVISION,
    load_runner,
    unload_runner,
)
from .protocol import lock_protocol
from .shared import run_shared_experiment


def _cub_root(value: str | None, search_root: str) -> Path:
    return Path(value).resolve() if value else discover_under_kaggle(Path(search_root))


def _training_smoke_question(cub_root: Path) -> tuple[Any, str, dict[str, Any]]:
    from PIL import Image
    from lger.cub import (
        load_cub_attributes,
        load_cub_certainties,
        load_cub_image_attribute_labels,
        load_cub_records,
    )

    records = [row for row in load_cub_records(cub_root) if row.official_split == "train"]
    names = {value.attribute_id: value for value in load_cub_attributes(cub_root)}
    certainties = {
        value.certainty_id: value.name.strip().casefold()
        for value in load_cub_certainties(cub_root)
    }
    for record in records:
        labels = load_cub_image_attribute_labels(cub_root, image_ids={record.image_id})[record.image_id]
        for label in labels:
            if (
                label.attribute_id in HISTORICAL_ATTRIBUTE_IDS
                and certainties[label.certainty_id] in {"probably", "definitely"}
            ):
                attribute = names[label.attribute_id]
                prompt = (
                    f"Is the bird's {attribute.name.replace('has_', '').replace('::', ' ')} present? "
                    "Answer yes or no."
                )
                image = Image.open(cub_root / "images" / record.relative_path).convert("RGB")
                return image, prompt, {
                    "image_id": record.image_id,
                    "attribute_id": label.attribute_id,
                    "target": int(label.is_present),
                    "official_split": "train",
                }
    raise RuntimeError("no eligible training-only smoke decision found")


def command_inventory(args: argparse.Namespace) -> None:
    value = environment_inventory()
    atomic_write_json(Path(args.output), value)
    print(json.dumps(value, indent=2, sort_keys=True))


def command_audit(args: argparse.Namespace) -> None:
    root = _cub_root(args.cub_root, args.search_root)
    value = audit_cub(root)
    if args.output:
        atomic_write_json(Path(args.output), value)
    print(json.dumps(value, indent=2, sort_keys=True))


def command_manifests(args: argparse.Namespace) -> None:
    root = _cub_root(args.cub_root, args.search_root)
    value = build_manifests(
        root,
        Path(args.output_dir),
        seed=args.seed,
        max_expensive_images=args.max_expensive_images,
        max_decisions_per_image=args.max_decisions_per_image,
    )
    print(json.dumps(value, indent=2, sort_keys=True))


def command_smoke(args: argparse.Namespace) -> None:
    root = _cub_root(args.cub_root, args.search_root)
    image, prompt, identity = _training_smoke_question(root)
    runner = load_runner(
        args.architecture,
        quantization=args.quantization,
        local_snapshot=args.local_snapshot,
    )
    try:
        result = runner.evaluate(image, prompt, generate=True)
        checkpoint, revision = {
            "llava": (LLAVA_CHECKPOINT, LLAVA_REVISION),
            "qwen3": (QWEN3_CHECKPOINT, QWEN3_REVISION),
        }[args.architecture]
        value = {
            "status": "PASS",
            "official_test_images_used": 0,
            "checkpoint": checkpoint,
            "required_revision": revision,
            "requested_quantization": args.quantization,
            "architecture_audit": runner.architecture_audit(),
            "training_decision": identity,
            "measurement": result,
        }
        atomic_write_json(Path(args.output), value)
        print(json.dumps(value, indent=2, sort_keys=True))
    finally:
        unload_runner(runner)


def command_lock(args: argparse.Namespace) -> None:
    value = lock_protocol(
        manifest_dir=Path(args.manifest_dir),
        smoke_dir=Path(args.smoke_dir),
        destination=Path(args.output),
        force_rebuild=args.force_rebuild,
    )
    print(json.dumps(value, indent=2, sort_keys=True))


def command_analyze(args: argparse.Namespace) -> None:
    value = analyze_per_example(
        Path(args.input),
        Path(args.output_dir),
        bootstrap_resamples=args.bootstrap_resamples,
        seed=args.seed,
    )
    print(json.dumps(value, indent=2, sort_keys=True))


def command_archive(args: argparse.Namespace) -> None:
    value = build_archive(
        Path(args.root),
        [Path(item) for item in args.include],
        Path(args.output),
    )
    print(json.dumps(value, indent=2, sort_keys=True))


def command_shared(args: argparse.Namespace) -> None:
    root = _cub_root(args.cub_root, args.search_root)
    value = run_shared_experiment(
        architecture=args.architecture,
        cub_root=root,
        protocol_path=Path(args.protocol),
        question_manifest=Path(args.questions),
        output_dir=Path(args.output_dir),
        quantization=args.quantization,
        local_snapshot=args.local_snapshot,
    )
    print(json.dumps(value, indent=2, sort_keys=True))


def command_figures(args: argparse.Namespace) -> None:
    value = render_final_figures(Path(args.analysis), Path(args.output_dir))
    print(json.dumps(value, indent=2, sort_keys=True))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="command", required=True)
    inventory = subparsers.add_parser("inventory")
    inventory.add_argument("--output", required=True)
    inventory.set_defaults(function=command_inventory)

    audit = subparsers.add_parser("audit")
    audit.add_argument("--cub-root")
    audit.add_argument("--search-root", default="/kaggle/input")
    audit.add_argument("--output")
    audit.set_defaults(function=command_audit)

    manifests = subparsers.add_parser("manifests")
    manifests.add_argument("--cub-root")
    manifests.add_argument("--search-root", default="/kaggle/input")
    manifests.add_argument("--output-dir", required=True)
    manifests.add_argument("--seed", type=int, default=20260916)
    manifests.add_argument("--max-expensive-images", type=int, default=256)
    manifests.add_argument("--max-decisions-per-image", type=int, default=4)
    manifests.set_defaults(function=command_manifests)

    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--architecture", choices=("llava", "qwen3"), required=True)
    smoke.add_argument("--cub-root")
    smoke.add_argument("--search-root", default="/kaggle/input")
    smoke.add_argument("--quantization", choices=("none", "4bit"), default="none")
    smoke.add_argument("--local-snapshot")
    smoke.add_argument("--output", required=True)
    smoke.set_defaults(function=command_smoke)

    lock = subparsers.add_parser("lock")
    lock.add_argument("--manifest-dir", required=True)
    lock.add_argument("--smoke-dir", required=True)
    lock.add_argument("--output", required=True)
    lock.add_argument("--force-rebuild", action="store_true")
    lock.set_defaults(function=command_lock)

    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--input", required=True)
    analyze.add_argument("--output-dir", required=True)
    analyze.add_argument("--bootstrap-resamples", type=int, default=10000)
    analyze.add_argument("--seed", type=int, default=20260916)
    analyze.set_defaults(function=command_analyze)

    shared = subparsers.add_parser("run-shared")
    shared.add_argument("--architecture", choices=("llava", "qwen3"), required=True)
    shared.add_argument("--cub-root")
    shared.add_argument("--search-root", default="/kaggle/input")
    shared.add_argument("--protocol", required=True)
    shared.add_argument("--questions", required=True)
    shared.add_argument("--output-dir", required=True)
    shared.add_argument("--quantization", choices=("none", "4bit"), default="none")
    shared.add_argument("--local-snapshot")
    shared.set_defaults(function=command_shared)

    figures = subparsers.add_parser("figures")
    figures.add_argument("--analysis", required=True)
    figures.add_argument("--output-dir", required=True)
    figures.set_defaults(function=command_figures)

    archive = subparsers.add_parser("archive")
    archive.add_argument("--root", required=True)
    archive.add_argument("--include", action="append", required=True)
    archive.add_argument("--output", required=True)
    archive.set_defaults(function=command_archive)
    return result


def main() -> None:
    args = parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
