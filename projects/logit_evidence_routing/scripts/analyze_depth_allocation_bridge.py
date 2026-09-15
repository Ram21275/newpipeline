#!/usr/bin/env python3
"""Analyze complete paired development outputs; never edits existing result tables."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from lger.depth_allocation import cluster_interval


def analyze(rows: list[dict], draws: int = 10000) -> dict:
    if not rows:
        raise ValueError("empty outcomes")
    keyed = {}
    for row in rows:
        key = (row["decision_id"], tuple(row["pair"]), row["reverse_order"], row["arm"])
        if key in keyed:
            raise ValueError("duplicate outcome identity")
        keyed[key] = row
    results = {}
    variants = sorted({(tuple(r["pair"]), r["reverse_order"]) for r in rows})
    comparisons = [("spectral", "random"), ("spectral", "uniform_total"),
                   ("spectral", "uniform_answer"), ("spectral", "mean"),
                   ("spectral", "contrast"), ("attention_spectral", "native"),
                   ("attention_oracle_landmark", "native"),
                   ("spectral_multi", "random_multi"), ("spectral_multi", "spectral"),
                   ("replace_spectral", "native"), ("replace_random_0", "native")]
    for pair, reverse in variants:
        subset = [r for r in rows if tuple(r["pair"]) == pair and r["reverse_order"] == reverse]
        ids = {r["decision_id"] for r in subset if r["arm"] == "native"}
        arms = {r["arm"] for r in subset}
        by_arm = {arm: {r["decision_id"]: dict(r) for r in subset if r["arm"] == arm} for arm in arms}
        for arm, values in by_arm.items():
            if "oracle" not in arm and set(values) != ids:
                raise ValueError(f"incomplete arm: {arm}")
            for key, value in values.items():
                baseline = by_arm["native"][key]
                if (value["image_id"], value["attribute_id"], value["target"], value["image_sha256"]) != (
                        baseline["image_id"], baseline["attribute_id"], baseline["target"], baseline["image_sha256"]):
                    raise ValueError("paired outcome identity/target/image mismatch")
        for random_name in ("random", "random_multi"):
            if all(f"{random_name}_{s}" in by_arm for s in (0, 1, 2)):
                by_arm[random_name] = {}
                for key in ids:
                    seed_rows = [by_arm[f"{random_name}_{s}"][key] for s in (0, 1, 2)]
                    averaged = dict(seed_rows[0])
                    for field in ("correct", "correct_margin", "margin", "positive_prediction"):
                        averaged[field] = statistics.mean(r[field] for r in seed_rows)
                    by_arm[random_name][key] = averaged
        summary = {}
        for arm, values in sorted(by_arm.items()):
            items = list(values.values())
            summary[arm] = {"decisions": len(items), "images": len({r["image_id"] for r in items}),
                            "accuracy": statistics.mean(r["correct"] for r in items),
                            "mean_correct_margin": statistics.mean(r["correct_margin"] for r in items),
                            "positive_rate": statistics.mean(r["positive_prediction"] for r in items),
                            "ties": sum(r["tie"] for r in items),
                            "mean_total_new_encoded_tokens": statistics.mean(r["total_new_encoded_tokens"] for r in items),
                            "mean_total_forward_seconds_with_diagnostics": statistics.mean(r["total_forward_seconds"] for r in items),
                            "by_target": {str(y): {"n": sum(r["target"]==y for r in items),
                                "accuracy": statistics.mean(r["correct"] for r in items if r["target"]==y)
                                if any(r["target"]==y for r in items) else None} for y in (0, 1)}}
            localization = [r for r in items if r.get("local_landmark_fraction") is not None]
            summary[arm]["localization"] = {
                "evaluable": len(localization),
                "mean_landmark_fraction": statistics.mean(r["local_landmark_fraction"] for r in localization) if localization else None,
                "complete_visible_landmark_set": statistics.mean(r["local_complete_landmark_set"] for r in localization) if localization else None}
            lens_rows = [r for r in items if r.get("lens_margin_by_layer") is not None]
            if lens_rows:
                layers = len(lens_rows[0]["lens_margin_by_layer"])
                summary[arm]["answer_lens_accuracy_by_layer"] = [statistics.mean(
                    r["lens_margin_by_layer"][layer] * (2*r["target"]-1) > 0 for r in lens_rows)
                    for layer in range(layers)]
            probe_rows = [r for r in items if r.get("frozen_probe_scores") is not None]
            summary[arm]["fixed_probe_status"] = "available" if probe_rows else "N/A"
            if probe_rows:
                layers = len(probe_rows[0]["frozen_probe_scores"])
                # Within-attribute AUROC controls incompatible score offsets across concepts.
                macro_auc = []
                for layer in range(layers):
                    per_attribute = []
                    for aid in sorted({r["attribute_id"] for r in probe_rows}):
                        pos = [r["frozen_probe_scores"][layer] for r in probe_rows if r["attribute_id"]==aid and r["target"]==1]
                        neg = [r["frozen_probe_scores"][layer] for r in probe_rows if r["attribute_id"]==aid and r["target"]==0]
                        if pos and neg:
                            per_attribute.append(statistics.mean(float(p>n) + 0.5*float(p==n) for p in pos for n in neg))
                    macro_auc.append(statistics.mean(per_attribute) if per_attribute else None)
                summary[arm]["fixed_probe_macro_auroc_by_layer"] = macro_auc
                summary[arm]["probe_warning"] = "tiny per-attribute support; high AUROC requires ROI-matched and shuffled controls"
        contrasts = {}
        for left, right in comparisons:
            if left not in by_arm or right not in by_arm:
                continue
            common = sorted(set(by_arm[left]) & set(by_arm[right]))
            images = [str(by_arm[left][key]["image_id"]) for key in common]
            if len(set(images)) < 2:
                contrasts[f"{left}-{right}"] = {"status": "insufficient_image_clusters"}
                continue
            fields = {}
            for field in ("correct", "correct_margin", "margin", "positive_prediction"):
                differences = [by_arm[left][key][field] - by_arm[right][key][field] for key in common]
                fields[field] = cluster_interval(images, differences, draws=draws,
                                                 alpha=0.05 / (len(comparisons) * 4 * len(variants)))
            fields["token_sum_match_within_2pct"] = all(abs(
                by_arm[left][key]["total_new_encoded_tokens"] - by_arm[right][key]["total_new_encoded_tokens"])
                <= 0.02 * max(by_arm[left][key]["total_new_encoded_tokens"], by_arm[right][key]["total_new_encoded_tokens"])
                for key in common)
            fields["compute_match_verified"] = False
            contrasts[f"{left}-{right}"] = fields
        interactions = {}
        if "spectral" in by_arm and "uniform_total" in by_arm:
            for radius in ("0.005", "0.01", "0.02"):
                for threshold in (0.15, 0.25):
                    groups = {"below": [], "above": []}
                    for key in ids:
                        extent = by_arm["native"][key].get("proxy_tokens", {}).get(radius, [])
                        if extent:
                            groups["below" if min(extent)<threshold else "above"].append(key)
                    record = {name: {"decisions": len(keys), "images": len({by_arm["native"][k]["image_id"] for k in keys})}
                              for name, keys in groups.items()}
                    for name, keys in groups.items():
                        if len({by_arm["native"][k]["image_id"] for k in keys}) >= 2:
                            record[name]["spectral_minus_uniform_accuracy"] = cluster_interval(
                                [str(by_arm["native"][k]["image_id"]) for k in keys],
                                [by_arm["spectral"][k]["correct"] - by_arm["uniform_total"][k]["correct"] for k in keys],
                                draws=draws, alpha=0.05/(6*2*len(variants)))
                    record["interpretation"] = "descriptive proxy strata, not a validated part-area cliff; empty strata are non-identifiable"
                    interactions[f"radius={radius};reference={threshold}"] = record
        results[f"{'/'.join(pair)};reverse={reverse}"] = {"arms": summary, "paired_contrasts": contrasts,
                                                            "proxy_threshold_strata": interactions}
    # No automatic promotion: benchmark-quality efficiency needs a separately timed
    # deployment path, and landmark squares cannot validate a real part-area cliff.
    return {"status": "DESCRIPTIVE_DEVELOPMENT_ONLY", "variants": results,
            "intervals": "paired image-cluster bootstrap; Bonferroni across configured contrasts, metrics, variants",
            "promotion": "BLOCKED_pending_compatible_fixed_probes_and_deployment_compute_matching",
            "threshold_status": "landmark_proxy_only; true attribute-area threshold not identified",
            "official_test_images_used": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing to overwrite an existing analysis")
    report = json.loads((args.run_dir / "report.json").read_text())
    path = args.run_dir / "outcomes.jsonl"
    if report["status"] != "PASS" or report["outcome_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest():
        raise ValueError("run is incomplete or its outcome hash changed")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if len(rows) != report["expected_outcomes"] or len({r["decision_id"] for r in rows}) != report["decisions"]:
        raise ValueError("outcome counts differ from the completed execution ledger")
    required = {"native", "spectral", "uniform_total", "random_0", "random_1", "random_2"}
    if report["mode"] != "robustness":
        required |= {"native_default", "uniform_answer", "mean", "contrast", "fixed_1", "fixed_2", "fixed_3",
                     "spectral_multi", "logit_concept", "attention_spectral", "replace_spectral", "attention_random_0", "replace_random_0",
                     "random_multi_0", "random_multi_1", "random_multi_2"}
    for decision in {r["decision_id"] for r in rows}:
        variants = {(tuple(r["pair"]), r["reverse_order"]) for r in rows if r["decision_id"] == decision}
        expected_variants = 6 if report["mode"] == "robustness" else 1
        if len(variants) != expected_variants:
            raise ValueError("incomplete answer-vocabulary/order factorial")
        for pair, reverse in variants:
            present = {r["arm"] for r in rows if r["decision_id"] == decision and tuple(r["pair"])==pair and r["reverse_order"]==reverse}
            if not required <= present:
                raise ValueError("required factorial arms are missing")
    result = analyze(rows)
    result["analyzer_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result["run_report_sha256"] = hashlib.sha256((args.run_dir / "report.json").read_bytes()).hexdigest()
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
