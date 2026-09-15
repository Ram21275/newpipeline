from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from lger.allocation_manifest import select_pilot, validate_decisions
from lger.depth_allocation import (choose_crops, cluster_interval, crop_at, normalized_maps,
                                   proxy_token_coverage, random_crops, select_depth_map)
from lger.hf_allocation import attention_bias_hook, budget_grid, cached_visual_intervention


def test_signed_disagreement_survives_a_cancelled_mean_and_layer_permutation():
    a = torch.tensor([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]], dtype=torch.float64)
    null = torch.tensor([[0.1, 0.8, 0.1], [0.1, 0.8, 0.1]], dtype=torch.float64)
    result = select_depth_map(a, null)
    assert result.status == "spectral"
    assert result.scores[0] > result.scores[1]
    assert torch.allclose(result.scores, select_depth_map(a.flip(0), null.flip(0)).scores)
    assert torch.allclose(result.scores, select_depth_map(a * 7, null * 11).scores)
    assert result.scores[1] < 0


def test_repeated_eigenvalue_projection_is_basis_invariant():
    a = torch.eye(3, dtype=torch.float64) + 0.2
    b = a.clone()
    b[:, 0] += 0.3
    original = select_depth_map(a, b)
    assert original.top_rank == 2
    permutation = torch.tensor([2, 0, 1])
    permuted = select_depth_map(a[:, permutation], b[:, permutation])
    assert torch.allclose(original.scores[permutation], permuted.scores, atol=1e-10)


def test_unidentified_query_does_not_orient_arbitrary_component():
    a = torch.tensor([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]])
    output = select_depth_map(a, a)
    assert output.status == "fallback_unidentified"
    assert torch.allclose(output.scores, normalized_maps(a).mean(0))
    for bad in (torch.zeros(2, 3), torch.full((2, 3), float("nan")), -a):
        with pytest.raises(ValueError):
            normalized_maps(bad)


def test_spatial_coverage_and_random_reproducibility():
    assert crop_at(0, 1, 0.25) == (0, 0.75, 0.25, 1.0)
    scores = torch.zeros(100)
    scores[22], scores[77] = 1, 1
    boxes = choose_crops(scores, (10, 10), width=0.25, count=2)
    assert boxes[0] != boxes[1]
    kwargs = dict(grid=(10, 10), width=0.25, count=2, image_id="5", attribute_id="2", seed=0)
    assert random_crops(**kwargs) == random_crops(**kwargs)
    assert random_crops(**kwargs) != random_crops(**{**kwargs, "seed": 1})
    native = proxy_token_coverage(torch.tensor([[0.5, 0.5]]), [(0, 0, 1, 1)], [(10, 10)], 0.01)
    crop = proxy_token_coverage(torch.tensor([[0.5, 0.5]]), [(0.25, 0.25, 0.75, 0.75)], [(10, 10)], 0.01)
    assert native[0] == pytest.approx(0.04)
    assert crop[0] == pytest.approx(4 * native[0])


@pytest.mark.parametrize("size", [(500, 333), (333, 500), (499, 497)])
@pytest.mark.parametrize("budget", [144, 288, 432, 720])
def test_realizable_budget_grid(size, budget):
    h, w = budget_grid(size, budget)
    assert abs(h * w - budget) <= 0.02 * budget


def test_official_test_rejected_before_any_image_file_is_opened(tmp_path):
    (tmp_path / "images.txt").write_text("1 first.jpg\n2 forbidden.jpg\n")
    (tmp_path / "train_test_split.txt").write_text("1 1\n2 0\n")
    dev = [{"image_id": 1, "relative_path": "first.jpg", "split": "val"},
           {"image_id": 2, "relative_path": "forbidden.jpg", "split": "val"}]
    row = {"decision_id": "d", "image_id": 2, "relative_path": "forbidden.jpg", "attribute_id": 2, "target": 0}
    with pytest.raises(ValueError, match="official-test"):
        validate_decisions([row], dev, tmp_path, {2})
    valid = {**row, "image_id": 1, "relative_path": "first.jpg"}
    assert validate_decisions([valid], dev, tmp_path, {2}) == [valid]
    # No image exists: metadata validation must not have needed it.
    assert not (tmp_path / "images").exists()


def test_cluster_resampling_preserves_repeated_decisions():
    one = cluster_interval(["a", "b"], [1, -1], draws=1000)
    duplicated = cluster_interval(["a", "a", "b", "b"], [1, 1, -1, -1], draws=1000)
    assert one["interval"] == duplicated["interval"]
    assert one["estimate"] == duplicated["estimate"] == 0
    with pytest.raises(ValueError):
        cluster_interval(["a", "a"], [0, 1])


def test_attention_intervention_changes_only_intended_query_keys():
    original = torch.zeros(1, 1, 5, 5)
    original[:, :, 0, 1:] = -1e9
    audit = {}
    hook = attention_bias_hook(torch.tensor([1, 2, 3]), torch.tensor([False, True, False]), 4, 2, audit)
    _, kwargs = hook(None, (), {"attention_mask": original})
    difference = kwargs["attention_mask"] - original
    assert torch.count_nonzero(difference) == 1
    assert difference[0, 0, 4, 2] == pytest.approx(0.693147, abs=1e-6)
    assert audit["bias_calls"] == 1
    assert original[0, 0, 4, 2] == 0
    with pytest.raises(RuntimeError):
        hook(None, (), {"attention_mask": torch.ones(5)})


def test_cached_internal_arm_bypasses_encoder_and_restores_on_failure():
    class Vision(torch.nn.Module):
        def forward(self, *args):
            raise AssertionError("vision encoding should be bypassed")
    model = SimpleNamespace(visual=Vision())
    cached = torch.tensor([[1., 2.], [3., 4.], [7., 8.]])
    selected = torch.tensor([True, False, False])
    with pytest.raises(ValueError, match="intentional"):
        with cached_visual_intervention(model, cached, selected=selected, replace=True):
            assert torch.equal(model.visual()[0], torch.tensor([5., 6.]))
            assert torch.equal(model.visual()[1:], cached[1:])
            raise ValueError("intentional failure")
    with pytest.raises(AssertionError, match="bypassed"):
        model.visual()
    assert torch.equal(cached[0], torch.tensor([1., 2.]))


def test_pilot_selection_ignores_outcomes_and_input_order():
    rows = [{"image_id": i, "attribute_id": a, "target": y, "margin": i*y}
            for a in (2, 5) for y in (0, 1) for i in (1, 2, 3)]
    first = select_pilot(rows, {2, 5})
    second = select_pilot(list(reversed(rows)), {2, 5})
    assert first == second and len(first) == 4


def test_analyzer_rejects_incomplete_and_mismatched_pairing():
    path = Path(__file__).parents[1] / "scripts/analyze_depth_allocation_bridge.py"
    spec = importlib.util.spec_from_file_location("bridge_analysis", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    base = {"pair": ["yes", "no"], "reverse_order": False, "arm": "native",
            "decision_id": "a", "image_id": 1, "attribute_id": 2, "target": 1, "image_sha256": "x"}
    rows = [base, {**base, "decision_id": "b", "image_id": 2}, {**base, "arm": "spectral"}]
    with pytest.raises(ValueError, match="incomplete arm"):
        module.analyze(rows)
    with pytest.raises(ValueError, match="duplicate"):
        module.analyze([base, base])


def test_end_to_end_analyzer_keeps_random_seeds_within_decision():
    path = Path(__file__).parents[1] / "scripts/analyze_depth_allocation_bridge.py"
    spec = importlib.util.spec_from_file_location("bridge_analysis_complete", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = []
    for image_id, target in ((1, 0), (2, 1)):
        for arm in ("native", "spectral", "uniform_total", "random_0", "random_1", "random_2"):
            rows.append({"decision_id": str(image_id), "image_id": image_id, "attribute_id": 2,
                         "image_sha256": str(image_id), "target": target, "arm": arm,
                         "pair": ["yes", "no"], "reverse_order": False,
                         "correct": int(arm != "random_0"), "correct_margin": 1., "margin": 2*target-1,
                         "positive_prediction": target, "tie": False,
                         "total_new_encoded_tokens": 720, "total_forward_seconds": 1.0})
    result = module.analyze(rows, draws=100)
    variant = result["variants"]["yes/no;reverse=False"]
    assert variant["arms"]["random"]["decisions"] == 2
    assert variant["arms"]["random"]["accuracy"] == pytest.approx(2/3)
    assert variant["paired_contrasts"]["spectral-random"]["correct"]["estimate"] == pytest.approx(1/3)
    assert not variant["paired_contrasts"]["spectral-uniform_total"]["compute_match_verified"]


def test_measure_uses_final_logits_and_audits_single_normalization(monkeypatch):
    from lger.hf_allocation import measure
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    class Tokenizer:
        def encode(self, word, **kwargs):
            return [3 if word == "bird" else 4]
        def decode(self, ids):
            return "bird" if ids == [3] else "birds"
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.visual = torch.nn.Identity()
            self.model = torch.nn.Module()
            self.model.norm = torch.nn.Linear(3, 3, bias=False)
            self.model.norm.weight.data.copy_(2 * torch.eye(3))
            self.head = torch.nn.Linear(3, 5, bias=False)
            self.head.weight.data.copy_(torch.arange(15).reshape(5, 3).float()/10)
            self.config = SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=2))
        def get_output_embeddings(self):
            return self.head
        def forward(self, input_ids, image_grid_thw, **kwargs):
            visual = self.visual(torch.tensor([[1., 2., 3.], [3., 2., 1.]]))
            hidden = torch.cat((visual, torch.ones(1, 3))).unsqueeze(0)
            final = self.model.norm(hidden)
            attention = torch.full((1, 2, 3, 3), 1/3)
            return SimpleNamespace(logits=self.head(final), hidden_states=(hidden, hidden, final),
                                   attentions=(attention, attention))
    model = Model()
    runner = SimpleNamespace(model=model, image_token_id=9, positive_token_ids=(0,),
                             negative_token_ids=(1,), processor=SimpleNamespace(tokenizer=Tokenizer()))
    record, aux = measure(runner, {"input_ids": torch.tensor([[9, 9, 1]]),
                                  "image_grid_thw": torch.tensor([[1, 2, 4]])}, capture=True)
    assert record["final_projection_max_error"] == 0
    assert record["double_norm_max_error"] > 1
    assert record["lens_margin_by_layer"][-1] == record["margin"]
    assert record["visual_tokens"] == 2
    assert aux["maps"].shape == (2, 2)


def test_metadata_preparation_cli_opens_no_images(tmp_path):
    import json
    import os
    import subprocess
    import sys
    cub = tmp_path / "cub"
    (cub / "attributes").mkdir(parents=True)
    (cub / "parts").mkdir()
    (cub / "images.txt").write_text("1 missing_a.jpg\n2 missing_b.jpg\n")
    (cub / "train_test_split.txt").write_text("1 1\n2 1\n")
    (cub / "attributes/attributes.txt").write_text("2 has_bill_shape::dagger\n")
    (cub / "attributes/certainties.txt").write_text("3 probably\n4 definitely\n")
    (cub / "attributes/image_attribute_labels.txt").write_text("1 2 0 3 1.0\n2 2 1 4 1.0\n")
    (cub / "parts/part_locs.txt").write_text("1 1 10 10 1\n2 1 20 20 1\n")
    (cub / "parts/parts.txt").write_text("1 beak\n")
    dev = tmp_path / "dev.csv"
    dev.write_text("image_id,relative_path,split\n1,missing_a.jpg,val\n2,missing_b.jpg,val\n")
    plan = tmp_path / "plan.json"
    script = Path(__file__).parents[1] / "scripts/run_depth_allocation_bridge.py"
    result = subprocess.run([sys.executable, str(script), "--mode", "prepare", "--cub-root", str(cub),
                             "--development-manifest", str(dev), "--plan", str(plan)],
                            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    payload = json.loads(plan.read_text())
    assert payload["pixels_opened"] == 0 and len(payload["decisions"]) == 2
    assert len(payload["missing_strata"]) == 50
    assert not (cub / "images").exists()
