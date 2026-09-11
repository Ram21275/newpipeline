"""Synthetic tests for the model-independent Phase 7 intervention track."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import torch

from lger.phase7 import (
    PRIMARY_METRIC,
    REPLACEMENT,
    Phase7ValidationError,
    aggregate_intervention_outcomes,
    apply_planned_intervention,
    build_intervention_plan,
    clustered_paired_bootstrap,
    compute_train_stage_means,
    normalize_outcomes,
    replace_tokens_with_stage_mean,
    validate_stage_pair,
)


PROJECT = Path(__file__).resolve().parents[1]


def load_cli():
    spec = importlib.util.spec_from_file_location(
        "run_phase7_interventions", PROJECT / "scripts" / "run_phase7_interventions.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CLI = load_cli()
STAGES = ("vision.late", "projector.output", "llm.early")


def token_rows(decisions=("d1",), *, include_matching=True):
    rows = []
    for decision_number, decision_id in enumerate(decisions, 1):
        for stage in STAGES[:2]:
            for token_index in range(12):
                row = {
                    "decision_id": decision_id,
                    "image_id": decision_number,
                    "stage": stage,
                    "token_index": token_index,
                    "evidence_score": 100.0 - token_index,
                }
                if include_matching:
                    row.update(
                        bird_box_membership="inside" if token_index % 2 == 0 else "outside",
                        hidden_state_norm=1.0 + token_index,
                    )
                rows.append(row)
    return rows


def make_plan(decisions=("d1", "d2"), random_replicates=2):
    return build_intervention_plan(
        token_rows(decisions),
        selected_stage=STAGES[0],
        neighbor_stage=STAGES[1],
        stage_order=STAGES,
        k=2,
        seed=17,
        random_replicates=random_replicates,
        norm_quantiles=2,
    )


def outcomes_for_plan(plan, top_drops=None, include_secondary=True, minimal=False):
    top_drops = top_drops or {}
    rows = []
    for record in plan["records"]:
        key = (record["decision_id"], record["stage"])
        if record["intervention_type"] == "top_evidence":
            drop = top_drops.get(key, 2.0)
        elif record["intervention_type"] == "matched_random":
            drop = 1.0
        else:
            drop = 0.5
        row = {
            "decision_id": record["decision_id"],
            "intervention_id": record["intervention_id"],
            "correct_answer_teacher_forced_margin_before": 5.0,
            "correct_answer_teacher_forced_margin_after": 5.0 - drop,
        }
        if not minimal:
            row.update({key: record[key] for key in (
                "image_id", "stage", "stage_role", "intervention_type"
            )})
        if include_secondary:
            row.update(
                generated_correct_before=True,
                generated_correct_after=record["intervention_type"] != "top_evidence",
                unrelated_attribute_margins_before={"a": 1.0, "b": 2.0},
                unrelated_attribute_margins_after={"a": 0.9, "b": 1.9},
            )
        rows.append(row)
    return rows


class Phase7PlanningTests(unittest.TestCase):
    def test_stage_pair_must_be_adjacent(self):
        validate_stage_pair(STAGES[0], STAGES[1], STAGES)
        with self.assertRaisesRegex(Phase7ValidationError, "adjacent"):
            validate_stage_pair(STAGES[0], STAGES[2], STAGES)

    def test_plan_is_deterministic_stratified_and_keyed(self):
        first = make_plan(("d1",), random_replicates=3)
        second = make_plan(("d1",), random_replicates=3)
        self.assertEqual(first, second)
        self.assertEqual(first["intervention_count"], 10)
        self.assertEqual(
            len({(row["decision_id"], row["intervention_id"]) for row in first["records"]}),
            10,
        )
        for stage in STAGES[:2]:
            records = [row for row in first["records"] if row["stage"] == stage]
            top = next(row for row in records if row["intervention_type"] == "top_evidence")
            low = next(row for row in records if row["intervention_type"] == "low_evidence")
            self.assertEqual(top["token_indices"], [0, 1])
            self.assertEqual(low["token_indices"], [11, 10])
            for random_record in [row for row in records if row["intervention_type"] == "matched_random"]:
                self.assertEqual(
                    random_record["matching_fields"],
                    ["bird_box_membership", "hidden_state_norm_quantile"],
                )
                self.assertEqual(len(random_record["token_indices"]), 2)
                self.assertEqual(
                    {index % 2 for index in random_record["token_indices"]},
                    {0, 1},
                )
                self.assertTrue(all(index < 6 for index in random_record["token_indices"]))
                self.assertFalse(set(random_record["token_indices"]) & {0, 1, 10, 11})

    def test_missing_metadata_is_allowed_but_partial_metadata_fails_closed(self):
        plan = build_intervention_plan(
            token_rows(include_matching=False), selected_stage=STAGES[0],
            neighbor_stage=STAGES[1], stage_order=STAGES, k=2, random_replicates=1,
        )
        self.assertTrue(all(row["matching_fields"] == [] for row in plan["records"]))
        rows = token_rows()
        rows[0].pop("hidden_state_norm")
        with self.assertRaisesRegex(Phase7ValidationError, "partially"):
            build_intervention_plan(
                rows, selected_stage=STAGES[0], neighbor_stage=STAGES[1],
                stage_order=STAGES, k=2, random_replicates=1, norm_quantiles=2,
            )

    def test_insufficient_exact_match_and_missing_neighbor_fail_closed(self):
        rows = token_rows(include_matching=False)
        for row in rows:
            row["bird_box_stratum"] = "rare" if row["token_index"] < 2 else "common"
        with self.assertRaisesRegex(Phase7ValidationError, "insufficient matched-random"):
            build_intervention_plan(
                rows, selected_stage=STAGES[0], neighbor_stage=STAGES[1],
                stage_order=STAGES, k=2, random_replicates=1,
            )
        one_stage = [row for row in token_rows() if row["stage"] == STAGES[0]]
        with self.assertRaisesRegex(Phase7ValidationError, "does not cover"):
            build_intervention_plan(
                one_stage, selected_stage=STAGES[0], neighbor_stage=STAGES[1],
                stage_order=STAGES, k=2,
            )


class Phase7ReplacementTests(unittest.TestCase):
    def test_training_means_and_replacement_do_not_mutate_inputs(self):
        means = compute_train_stage_means([
            {"development_split": "train", "stage": STAGES[0], "hidden_state": torch.tensor([1.0, 3.0])},
            {"development_split": "train", "stage": STAGES[0], "hidden_state": torch.tensor([3.0, 5.0])},
        ])
        self.assertTrue(torch.equal(means[STAGES[0]], torch.tensor([2.0, 4.0])))
        hidden = torch.arange(16, dtype=torch.float32).reshape(2, 4, 2)
        original = hidden.clone()
        replaced = replace_tokens_with_stage_mean(hidden, [1, 3], means[STAGES[0]])
        self.assertTrue(torch.equal(hidden, original))
        self.assertTrue(torch.equal(replaced[:, 1], torch.tensor([[2.0, 4.0], [2.0, 4.0]])))
        self.assertTrue(torch.equal(replaced[:, 3], torch.tensor([[2.0, 4.0], [2.0, 4.0]])))
        self.assertTrue(torch.equal(replaced[:, 0], original[:, 0]))

    def test_apply_changes_only_planned_stage_and_rejects_bad_mean(self):
        states = {STAGES[0]: torch.zeros(1, 4, 2), STAGES[1]: torch.ones(1, 4, 2)}
        record = {"stage": STAGES[0], "token_indices": [0], "replacement": REPLACEMENT}
        result = apply_planned_intervention(states, {STAGES[0]: torch.tensor([2.0, 3.0])}, record)
        self.assertTrue(torch.equal(result[STAGES[1]], states[STAGES[1]]))
        self.assertTrue(torch.equal(result[STAGES[0]][0, 0], torch.tensor([2.0, 3.0])))
        with self.assertRaisesRegex(Phase7ValidationError, "shape"):
            replace_tokens_with_stage_mean(states[STAGES[0]], [0], torch.zeros(3))
        with self.assertRaisesRegex(Phase7ValidationError, "training rows only"):
            compute_train_stage_means([
                {"development_split": "val", "stage": STAGES[0], "hidden_state": torch.ones(2)}
            ])


class Phase7AggregationTests(unittest.TestCase):
    def test_primary_secondary_contrasts_and_cluster_bootstrap(self):
        plan = make_plan(("d1", "d2", "d3"), random_replicates=2)
        # d1 and d2 share image only after this controlled identity edit.
        for record in plan["records"]:
            if record["decision_id"] == "d2":
                record["image_id"] = "1"
        top = {}
        for stage in STAGES[:2]:
            top.update({("d1", stage): 2.0, ("d2", stage): 2.0, ("d3", stage): 4.0})
        result = aggregate_intervention_outcomes(
            outcomes_for_plan(plan, top), plan=plan, bootstrap_samples=300, seed=9
        )
        self.assertEqual(result["primary_metric"], PRIMARY_METRIC)
        self.assertEqual(result["image_count"], 2)
        self.assertTrue(result["null_result_valid"])
        contrast = next(row for row in result["paired_contrasts"]
                        if row["stage"] == STAGES[0] and row["reference"] == "matched_random"
                        and row["metric"] == PRIMARY_METRIC)
        self.assertEqual(contrast["estimate"], 2.0)  # image means: (1+1)/2 and 3
        self.assertEqual(contrast["n_images"], 2)
        self.assertEqual(contrast["n_paired_rows"], 3)
        generated = next(row for row in result["paired_contrasts"]
                         if row["metric"] == "generated_correctness_drop"
                         and row["stage"] == STAGES[0] and row["reference"] == "matched_random")
        self.assertEqual(generated["estimate"], 1.0)
        self.assertIn("unrelated_attribute_margin_drop", result["secondary_metrics"])

    def test_null_result_is_valid_and_bootstrap_is_deterministic(self):
        first = clustered_paired_bootstrap([1, 1, 2], [0.0, 0.0, 0.0], samples=100, seed=4)
        second = clustered_paired_bootstrap([1, 1, 2], [0.0, 0.0, 0.0], samples=100, seed=4)
        self.assertEqual(first, second)
        self.assertEqual((first["estimate"], first["ci_low"], first["ci_high"]), (0.0, 0.0, 0.0))
        plan = make_plan(("d1",), random_replicates=1)
        rows = outcomes_for_plan(plan, include_secondary=False)
        for row in rows:
            row["correct_answer_teacher_forced_margin_after"] = row[
                "correct_answer_teacher_forced_margin_before"
            ]
        result = aggregate_intervention_outcomes(rows, plan=plan, bootstrap_samples=50)
        self.assertTrue(all(row["interpretation"] == "null_compatible"
                            for row in result["paired_contrasts"]))

    def test_plan_join_supports_minimal_outcomes_and_rejects_failures(self):
        plan = make_plan(("d1",), random_replicates=1)
        rows = outcomes_for_plan(plan, minimal=True)
        normalized = normalize_outcomes(rows, plan=plan)
        self.assertEqual(len(normalized), plan["intervention_count"])
        duplicated = rows + [dict(rows[0])]
        with self.assertRaisesRegex(Phase7ValidationError, "cover every|duplicate"):
            normalize_outcomes(duplicated, plan=plan)
        partial = outcomes_for_plan(plan)
        partial[0].pop("generated_correct_after")
        with self.assertRaisesRegex(Phase7ValidationError, "supplied together"):
            normalize_outcomes(partial, plan=plan)
        inconsistent = outcomes_for_plan(plan)
        inconsistent[1]["correct_answer_teacher_forced_margin_before"] = 6.0
        with self.assertRaisesRegex(Phase7ValidationError, "baseline outcomes differ"):
            normalize_outcomes(inconsistent, plan=plan)
        incomplete = [row for row in outcomes_for_plan(plan)
                      if row["intervention_type"] != "low_evidence"]
        with self.assertRaisesRegex(Phase7ValidationError, "incomplete intervention design"):
            aggregate_intervention_outcomes(incomplete, bootstrap_samples=10)


class Phase7CliTests(unittest.TestCase):
    def test_cli_plans_and_aggregates_precomputed_outcomes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata_path = root / "tokens.jsonl"
            metadata_path.write_text(
                "\n".join(json.dumps(row) for row in token_rows()) + "\n", encoding="utf-8"
            )
            plan_path = root / "plan.json"
            CLI.main([
                "plan", "--token-metadata", str(metadata_path), "--output", str(plan_path),
                "--selected-stage", STAGES[0], "--neighbor-stage", STAGES[1],
                "--stage-order", *STAGES, "--k", "2", "--seed", "5",
                "--random-seeds", "0", "--norm-quantiles", "2",
            ])
            plan = json.loads(plan_path.read_text())
            self.assertEqual(plan["intervention_count"], 6)
            outcomes_path = root / "outcomes.json"
            outcomes_path.write_text(json.dumps(outcomes_for_plan(plan, minimal=True)), encoding="utf-8")
            summary_path = root / "summary.json"
            CLI.main([
                "aggregate", "--outcomes", str(outcomes_path), "--plan", str(plan_path),
                "--output", str(summary_path), "--bootstrap-samples", "50", "--seed", "6",
            ])
            summary = json.loads(summary_path.read_text())
            self.assertEqual(summary["outcome_count"], 6)
            self.assertEqual(summary["bootstrap_samples"], 50)


if __name__ == "__main__":
    unittest.main()
