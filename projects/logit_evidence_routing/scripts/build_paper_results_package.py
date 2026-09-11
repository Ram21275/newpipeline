#!/usr/bin/env python3
"""Build evidence-bounded paper tables, claims, and an abstract scaffold."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"input CSV is empty: {path}")
    return rows


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    values = list(rows)
    if not values:
        raise RuntimeError(f"refusing to write empty paper table: {path}")
    fields = list(dict.fromkeys(key for row in values for key in row))
    with path.with_suffix(path.suffix + ".tmp").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(values)
    path.with_suffix(path.suffix + ".tmp").replace(path)


def fmt(value: object) -> str:
    return f"{float(value):.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase3-dir", type=Path, required=True)
    parser.add_argument("--phase4-dir", type=Path, required=True)
    parser.add_argument("--phase4-review", type=Path, required=True)
    parser.add_argument("--phase5-analysis-dir", type=Path, required=True)
    parser.add_argument("--phase6-dir", type=Path, required=True)
    parser.add_argument("--phase7-analysis", type=Path)
    parser.add_argument("--frozen-protocol", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    p3_report = read_json(args.phase3_dir / "phase3_run_report.json")
    p4_report = read_json(args.phase4_dir / "phase4_validation_report.json")
    p4_review = read_json(args.phase4_review)
    p5_report = read_json(args.phase5_analysis_dir / "phase5_run_report.json")
    p6_report = read_json(args.phase6_dir / "phase6_run_report.json")
    transition = read_json(args.phase6_dir / "transition_decision.json")
    if not (
        p3_report.get("status") == p4_report.get("status") == p5_report.get("status")
        == p6_report.get("status") == "PASS"
        and p4_review.get("status") == "PASS_DESCRIPTIVE_ONLY"
    ):
        raise RuntimeError("paper package requires passing Phase 3R-6 reports and reviewed Phase 4")
    if any(report.get("official_test_images_used", 0) != 0
           for report in (p3_report, p4_report, p5_report, p6_report)):
        raise RuntimeError("unexpected official-test evidence in a development paper package")

    p3 = read_csv(args.phase3_dir / "attribute_probe_by_stage.csv")
    p4 = [
        row for row in read_csv(args.phase4_dir / "attribute_macro_summary.csv")
        if row["split"] == "val"
        and row["attribute_group"] == "ALL_SELECTED_ATTRIBUTES"
        and row["selector"] in ("vision_cls_attention", "logit_concept")
        and row["K"] == "32"
        and row["metric"] in ("part_patch_recall", "any_part_hit", "top1_part_hit")
    ]
    p5 = [
        row for row in read_csv(args.phase5_analysis_dir / "utilization_summary.csv")
        if row["scope"] == "macro_attribute"
    ]
    p6 = [
        row for row in read_csv(args.phase6_dir / "inference_summary.csv")
        if row["aggregation"] == "fixed_attribute_macro"
    ]
    write_csv(args.output_dir / "table_phase3_accessibility.csv", p3)
    write_csv(args.output_dir / "table_phase4_localization.csv", p4)
    write_csv(args.output_dir / "table_phase5_vqa.csv", p5)
    write_csv(args.output_dir / "table_phase6_joint_inference.csv", p6)

    causal_sentence = "Phase 7 has not run because the development rubric did not justify one target."
    causal_status = "NOT_RUN_OR_BLOCKED"
    p7: dict[str, Any] | None = None
    primary_causal: dict[str, Any] | None = None
    if args.phase7_analysis is not None:
        p7 = read_json(args.phase7_analysis)
        if p7.get("purpose") != "phase7_targeted_causal_intervention_aggregation":
            raise RuntimeError("unexpected Phase 7 analysis artifact")
        primary_rows = [
            row for row in p7["paired_contrasts"]
            if row["stage_role"] == "selected"
            and row["reference"] == "matched_random"
            and row["metric"] == p7["primary_metric"]
        ]
        if len(primary_rows) != 1:
            raise RuntimeError("Phase 7 has no unique selected-stage primary causal contrast")
        primary_causal = primary_rows[0]
        causal_status = str(primary_causal["interpretation"]).upper()
        causal_sentence = (
            f"At {primary_causal['stage']}, top-evidence replacement changed the correct-answer "
            f"margin by {fmt(primary_causal['estimate'])} more than matched-random replacement "
            f"(95% image-clustered CI [{fmt(primary_causal['ci_low'])}, "
            f"{fmt(primary_causal['ci_high'])}]; {primary_causal['interpretation']})."
        )
        write_csv(args.output_dir / "table_phase7_causal.csv", p7["paired_contrasts"])

    claim_rows = [
        {
            "claim": "Fine-grained attributes are linearly accessible at measured frozen stages.",
            "status": "SUPPORTED_DEVELOPMENT",
            "source": "Phase 3R attribute_probe_by_stage.csv",
            "boundary": "Linear accessibility is not model use and uses the frozen development cohort.",
        },
        {
            "claim": p4_review["supported_result"],
            "status": "SUPPORTED_DEVELOPMENT_DESCRIPTIVE",
            "source": "Reviewed Phase 4 localization bundle",
            "boundary": "Landmarks are coarse proxies; attention and similarity are not causal.",
        },
        {
            "claim": f"The development rubric selected {transition['selected_transition']}.",
            "status": "SUPPORTED_AS_TARGET_SELECTION",
            "source": "Phase 6 clustered joint analysis",
            "boundary": "The label selects the next test; it is not causal proof.",
        },
        {
            "claim": causal_sentence,
            "status": causal_status,
            "source": "Phase 7 matched hidden-state intervention" if p7 else "Phase 7 gate",
            "boundary": "Development causal evidence still requires untouched official-test confirmation.",
        },
        {
            "claim": "The result generalizes to the official CUB test split.",
            "status": "NOT_YET_TESTED",
            "source": "Phase 8",
            "boundary": "Forbidden until the frozen official-test run is complete.",
        },
    ]
    write_csv(args.output_dir / "claim_ledger.csv", claim_rows)

    p5_image = [row for row in p5 if row["control"] == "image"]
    p5_accuracy = (
        sum(float(row["margin_accuracy"]) for row in p5_image) / len(p5_image)
        if p5_image else float("nan")
    )
    frozen_note = (
        f"Phase 8 protocol: `{args.frozen_protocol}`."
        if args.frozen_protocol is not None and args.frozen_protocol.is_file()
        else "Phase 8 is not frozen; do not make held-out/generalization claims."
    )
    markdown = f"""# Paper findings package

## Evidence-backed result

Phase 3R measures linear accessibility across nine frozen stages. Phase 4 shows the reviewed selector-dependent localization mismatch: {p4_review['supported_result']} Phase 5 adds fixed-prompt VQA behavior with prompt-only and image-shuffled controls; the macro mean teacher-forced margin accuracy across its two grounded cohorts is {p5_accuracy:.3f}. Phase 6 selects **{transition['selected_transition']}** under the predeclared clustered-bootstrap rubric.

{causal_sentence}

## Abstract scaffold

Fine-grained VLM decisions can hinge on small visual attributes, yet ordinary accuracy does not reveal whether the relevant evidence is absent, poorly localized, or unused. We trace a fixed set of CUB attributes through a frozen LLaVA vision encoder, multimodal projector, language model, and binary VQA answer using matched linear-accessibility, localization, utilization-control, and causal-intervention measurements. On the development cohort, the predeclared joint analysis selects **{transition['selected_transition']}**. {causal_sentence} These findings distinguish where evidence is decodable from where it is spatially concentrated and causally used. [Add only the frozen Phase 8 held-out estimate here before submission.]

## Claim discipline

- Probe performance supports accessibility, not use.
- Attention and dense similarity are localization diagnostics, not explanations.
- Phase 7 supports use only when top-evidence replacement exceeds matched controls with a CI excluding zero.
- {frozen_note}
- Treat null-compatible effects as results; do not retune the transition on the official test set.

## Paper assembly order

1. Use `table_phase3_accessibility.csv` for the trajectory figure.
2. Use `table_phase4_localization.csv` for discriminative-versus-semantic localization.
3. Use `table_phase5_vqa.csv` and `table_phase6_joint_inference.csv` for behavior and transition selection.
4. Include `table_phase7_causal.csv` only when generated.
5. Resolve every sentence against `claim_ledger.csv` before writing the abstract or conclusion.
"""
    destination = args.output_dir / "PAPER_FINDINGS.md"
    destination.write_text(markdown, encoding="utf-8")
    artifact_paths = sorted(path for path in args.output_dir.iterdir() if path.is_file())
    manifest = {
        "schema_version": 1,
        "status": "PASS",
        "selected_transition": transition["selected_transition"],
        "phase7_included": p7 is not None,
        "phase8_frozen": args.frozen_protocol is not None and args.frozen_protocol.is_file(),
        "official_test_images_used": 0,
        "artifacts": {
            path.name: {
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in artifact_paths
        },
    }
    (args.output_dir / "paper_package_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
