#!/usr/bin/env python3
"""Join precomputed Phase 3R/4/5 CSVs and run the frozen Phase 6 analysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from lger.phase6 import Phase6Config, run_phase6_analysis


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase3-csv", type=Path, required=True,
                        help="Phase 3R decision_probe_scores.csv")
    parser.add_argument("--phase4-csv", type=Path, required=True,
                        help="Phase 4 decision-level/attribute_metrics.csv")
    parser.add_argument("--phase5-csv", type=Path, required=True,
                        help="Phase 5 decision results CSV")
    parser.add_argument(
        "--phase3-patch-csv",
        type=Path,
        help="Optional Phase 3R probe_patch_evidence.csv for dilution/redistribution tests",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt-id", default="cub_attribute_yes_no_v1")
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--phase3-pooling", default="mean")
    parser.add_argument("--phase3-control", default="primary")
    parser.add_argument("--vision-stage", default="vision.late")
    parser.add_argument("--language-stage", default="llm.final")
    parser.add_argument("--phase4-k", type=int, default=32)
    parser.add_argument("--discriminative-selector", default="vision_cls_attention")
    parser.add_argument("--semantic-selector", default="logit_concept")
    parser.add_argument("--localization-metric", default="part_patch_recall")
    parser.add_argument("--phase5-control", default="image")
    parser.add_argument("--phase5-outcome", choices=("margin_correct", "generation_correct"),
                        default="margin_correct")
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260911)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--minimum-effect", type=float, default=0.05)
    args = parser.parse_args()
    config = Phase6Config(
        prompt_id=args.prompt_id, split=args.split,
        phase3_pooling=args.phase3_pooling, phase3_control=args.phase3_control,
        vision_stage=args.vision_stage, language_stage=args.language_stage,
        phase4_k=args.phase4_k,
        discriminative_selector=args.discriminative_selector,
        semantic_selector=args.semantic_selector,
        localization_metric=args.localization_metric,
        phase5_control=args.phase5_control,
        phase5_outcome=args.phase5_outcome,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
        confidence_level=args.confidence_level,
        minimum_effect=args.minimum_effect,
    )
    report = run_phase6_analysis(args.phase3_csv, args.phase4_csv, args.phase5_csv,
                                 args.output_dir, config, args.phase3_patch_csv)
    print(f"Phase 6 PASS: {args.output_dir}")
    print(f"Selected transition: {report['selected_transition']}")


if __name__ == "__main__":
    main()
