#!/usr/bin/env python3
"""Repair concept and fusion fields from cached Phase 01B patch hidden states."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lger.hf_llava import HfLlavaPatchExtractor  # noqa: E402
from lger.phase1b import refresh_concept_features  # noqa: E402
from lger.reproducibility import current_git_commit  # noqa: E402


TOKENIZATION_POLICY = "single_lexical_token_v1"


def _read_source_config(cache_dir: Path) -> dict[str, object]:
    path = cache_dir / "extraction_config.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing Phase 01B extraction config: {path}")
    config = json.loads(path.read_text(encoding="utf-8"))
    if int(config.get("schema_version", 0)) != 2:
        raise RuntimeError("Concept repair requires a Phase 01B schema-2 cache")
    if not config.get("stores_patch_hidden_states"):
        raise RuntimeError("Concept repair requires cached patch hidden states")
    return config


def _write_or_validate_config(path: Path, config: dict[str, object]) -> None:
    if path.is_file():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous != config:
            raise RuntimeError(
                f"Existing corrected-cache configuration differs: {path}. "
                "Choose another output directory."
            )
        return
    path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fixed-concepts", nargs="+", default=["bird", "birds"])
    parser.add_argument("--projection-chunk-size", type=int)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source_dir = args.source_cache_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if source_dir == output_dir:
        raise ValueError("Use a new output directory; in-place cache repair is disabled")
    if args.max_images is not None and args.max_images <= 0:
        raise ValueError("max-images must be positive")
    fixed_concepts = tuple(dict.fromkeys(args.fixed_concepts))
    if not fixed_concepts:
        raise ValueError("at least one fixed concept is required")

    source_config = _read_source_config(source_dir)
    records = sorted((source_dir / "records").glob("*.pt"))
    if not records:
        raise FileNotFoundError(f"No cache records found below {source_dir / 'records'}")
    selected_records = records[: args.max_images] if args.max_images is not None else records
    k_values = [int(value) for value in source_config["k_values"]]
    projection_chunk_size = args.projection_chunk_size or int(
        source_config.get("projection_chunk_size", 64)
    )
    source_revision = str(
        source_config.get("resolved_revision")
        or source_config.get("requested_revision", "main")
    )

    print(
        f"Loading {source_config['model']} to reproject cached hidden states; "
        "no images or VLM forward passes are needed..."
    )
    extractor = HfLlavaPatchExtractor.from_pretrained(
        str(source_config["model"]),
        revision=source_revision,
        quantization=str(source_config.get("quantization", "4bit")),
        layer_offset=int(source_config["layer_offset"]),
        projection_chunk_size=projection_chunk_size,
        fixed_concepts=fixed_concepts,
    )
    expected_revision = str(source_config.get("resolved_revision", "unknown"))
    if (
        expected_revision != "unknown"
        and extractor.resolved_revision != expected_revision
    ):
        raise RuntimeError(
            "Loaded model revision does not match the cache: "
            f"{extractor.resolved_revision} != {expected_revision}"
        )
    corrected_config = {
        **source_config,
        "git_commit": current_git_commit(REPO_ROOT),
        "requested_revision": source_revision,
        "resolved_revision": extractor.resolved_revision,
        "fixed_concepts": list(fixed_concepts),
        "concept_token_ids": list(extractor.concept_token_ids),
        "concept_tokens": list(extractor.concept_tokens),
        "concept_tokenization_policy": TOKENIZATION_POLICY,
        "cache_correction": {
            "kind": "phase1b_concept_tokenization",
            "source_cache_dir": str(source_dir),
            "source_git_commit": source_config.get("git_commit", "unknown"),
            "recomputed_fields": ["logit_concept", "attention_logit_fusion"],
            "reused_fields": "all non-concept score maps, selections, and features",
        },
    }
    output_records = output_dir / "records"
    output_records.mkdir(parents=True, exist_ok=True)
    _write_or_validate_config(output_dir / "extraction_config.json", corrected_config)

    completed = 0
    skipped = 0
    started = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    for position, source in enumerate(selected_records, start=1):
        destination = output_records / source.name
        if destination.is_file() and not args.overwrite:
            skipped += 1
            continue
        record = torch.load(source, map_location="cpu", weights_only=False)
        if int(record.get("schema_version", 0)) != 2:
            raise RuntimeError(f"Expected schema version 2: {source}")
        patch_hidden = record.get("patch_hidden_states")
        if not isinstance(patch_hidden, torch.Tensor):
            raise RuntimeError(f"Missing patch hidden states: {source}")
        concept_scores = extractor.project_concept_logprob(patch_hidden)
        refresh_concept_features(record, concept_scores, k_values)
        record["concept_token_ids"] = extractor.concept_token_ids
        record["concept_tokens"] = extractor.concept_tokens
        record["concept_tokenization_policy"] = TOKENIZATION_POLICY
        temporary = destination.with_suffix(".pt.tmp")
        torch.save(record, temporary)
        temporary.replace(destination)
        completed += 1
        del record, patch_hidden, concept_scores
        print(
            f"[{position}/{len(selected_records)}] {source.name} | "
            f"new={completed} resumed={skipped}"
        )

    summary = {
        "git_commit": current_git_commit(REPO_ROOT),
        "source_cache_dir": str(source_dir),
        "output_dir": str(output_dir),
        "source_records": len(records),
        "requested_records": len(selected_records),
        "new_records": completed,
        "resumed_records": skipped,
        "fixed_concepts": list(fixed_concepts),
        "concept_token_ids": list(extractor.concept_token_ids),
        "concept_tokens": list(extractor.concept_tokens),
        "concept_tokenization_policy": TOKENIZATION_POLICY,
        "runtime_seconds": time.perf_counter() - started,
        "max_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
        ),
    }
    (output_dir / "correction_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Corrected Phase 01B cache complete: {output_records}")


if __name__ == "__main__":
    main()
