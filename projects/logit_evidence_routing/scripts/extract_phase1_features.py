#!/usr/bin/env python3
"""Extract compact Random/Attention/Logit pilot features on a Kaggle GPU."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lger.cub import discover_cub_root, load_pilot_manifest  # noqa: E402
from lger.hf_llava import HfLlavaPatchExtractor  # noqa: E402
from lger.reproducibility import current_git_commit  # noqa: E402
from lger.scoring import stable_topk  # noqa: E402


def random_patch_indices(patch_count: int, k: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randperm(patch_count, generator=generator)[:k]


def feature_key(selector: str, k: int, seed: int | None = None) -> str:
    suffix = f"_seed{seed}" if seed is not None else ""
    return f"{selector}_k{k}{suffix}"


def _write_or_validate_run_config(path: Path, config: dict[str, object]) -> None:
    if path.is_file():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous != config:
            raise RuntimeError(
                f"Existing cache configuration differs: {path}. "
                "Choose a new output directory or pass the original arguments."
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--search-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--cub-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--layer-offset", type=int, default=-2)
    parser.add_argument("--projection-chunk-size", type=int, default=64)
    parser.add_argument("--quantization", choices=("4bit", "none"), default="4bit")
    parser.add_argument("--prompt", default="Describe the image briefly.")
    parser.add_argument("--k", type=int, nargs="+", default=[16, 32])
    parser.add_argument("--random-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Enable a GPU accelerator in Kaggle before extraction")
    k_values = sorted(set(args.k))
    random_seeds = sorted(set(args.random_seeds))
    if not k_values or min(k_values) <= 0:
        raise ValueError("all K values must be positive")
    if not random_seeds:
        raise ValueError("at least one random seed is required")

    cub_root = args.cub_root or discover_cub_root(args.search_root)
    records = load_pilot_manifest(args.manifest)
    if args.max_images is not None:
        if args.max_images <= 0:
            raise ValueError("max-images must be positive")
        records = records[: args.max_images]

    print(f"Loading {args.model} on GPU; this is the slow first-run step...")
    extractor = HfLlavaPatchExtractor.from_pretrained(
        args.model,
        revision=args.revision,
        quantization=args.quantization,
        layer_offset=args.layer_offset,
        projection_chunk_size=args.projection_chunk_size,
    )
    run_config: dict[str, object] = {
        "git_commit": current_git_commit(REPO_ROOT),
        "manifest": str(args.manifest.resolve()),
        "model": args.model,
        "requested_revision": args.revision,
        "resolved_revision": extractor.resolved_revision,
        "layer_offset": args.layer_offset,
        "projection_chunk_size": args.projection_chunk_size,
        "quantization": args.quantization,
        "prompt": args.prompt,
        "k_values": k_values,
        "random_seeds": random_seeds,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records_dir = args.output_dir / "records"
    records_dir.mkdir(parents=True, exist_ok=True)
    _write_or_validate_run_config(args.output_dir / "extraction_config.json", run_config)

    completed = 0
    skipped = 0
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for position, record in enumerate(records, start=1):
        destination = records_dir / f"{record.image_id:05d}.pt"
        if destination.is_file() and not args.overwrite:
            skipped += 1
            continue

        image_path = cub_root / "images" / record.relative_path
        with Image.open(image_path) as image_file:
            evidence = extractor.extract(image_file.convert("RGB"), args.prompt)
        patch_count = evidence.hidden_states.shape[0]
        if max(k_values) > patch_count:
            raise ValueError(f"K={max(k_values)} exceeds {patch_count} visual patches")

        features: dict[str, torch.Tensor] = {}
        selections: dict[str, torch.Tensor] = {}
        for k in k_values:
            attention_indices = stable_topk(evidence.attention_scores, k)
            logit_indices = stable_topk(evidence.evidence_scores["maxprob"], k)
            for selector, indices in (
                ("attention", attention_indices),
                ("logit", logit_indices),
            ):
                key = feature_key(selector, k)
                selections[key] = indices
                features[key] = evidence.hidden_states.index_select(0, indices).float().mean(0)
            for seed in random_seeds:
                indices = random_patch_indices(
                    patch_count,
                    k,
                    seed=seed * 1_000_003 + record.image_id,
                )
                key = feature_key("random", k, seed)
                selections[key] = indices
                features[key] = evidence.hidden_states.index_select(0, indices).float().mean(0)

        max_k = max(k_values)
        decoded_indices = stable_topk(evidence.evidence_scores["maxprob"], max_k)
        decoded_tokens = extractor.decode_token_ids(
            evidence.top_token_ids.index_select(0, decoded_indices)
        )
        payload = {
            "schema_version": 1,
            "image_id": record.image_id,
            "relative_path": record.relative_path,
            "label": record.label,
            "class_name": record.class_name,
            "split": record.split,
            "patch_count": patch_count,
            "grid_size": evidence.grid_size,
            "processed_image": evidence.processed_image,
            "features": features,
            "selections": selections,
            "attention_scores": evidence.attention_scores,
            "evidence_scores": evidence.evidence_scores,
            "decoded_logit_patch_indices": decoded_indices,
            "decoded_logit_top_tokens": decoded_tokens,
        }
        temporary = destination.with_suffix(".pt.tmp")
        torch.save(payload, temporary)
        temporary.replace(destination)
        completed += 1
        elapsed = time.perf_counter() - started
        print(
            f"[{position}/{len(records)}] {record.relative_path} | "
            f"new={completed} resumed={skipped} elapsed={elapsed / 60:.1f}m"
        )

    summary = {
        **run_config,
        "cub_root": str(cub_root.resolve()),
        "requested_records": len(records),
        "new_records": completed,
        "resumed_records": skipped,
        "runtime_seconds": time.perf_counter() - started,
        "max_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
    }
    (args.output_dir / "extraction_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Cache complete: {records_dir}")


if __name__ == "__main__":
    main()
