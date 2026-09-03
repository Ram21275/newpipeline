import inspect
import unittest

import torch

from lger import EvidenceRouter, RouterConfig


def candidates() -> tuple[torch.Tensor, ...]:
    hidden = torch.arange(3 * 6 * 4, dtype=torch.float32).reshape(3, 6, 4)
    logits = torch.zeros(3, 6, 4)
    logits[:, 2, 0] = torch.tensor([3.0, 4.0, 5.0])
    logits[:, 4, 1] = torch.tensor([2.0, 3.0, 4.0])
    coords = torch.tensor(
        [[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1]]
    ).float()
    layers = torch.tensor([10, 11, 12])
    attention = torch.tensor([0.0, 7.0, 1.0, 0.0, 2.0, 0.0])
    return hidden, logits, coords, layers, attention


class RoutingTests(unittest.TestCase):
    def test_router_interface_has_no_label_input(self) -> None:
        parameters = inspect.signature(EvidenceRouter.__call__).parameters
        self.assertNotIn("label", parameters)
        self.assertNotIn("class_name", parameters)

    def test_lger_selects_unique_patches_and_final_layer_states(self) -> None:
        hidden, logits, coords, layers, attention = candidates()
        selected = EvidenceRouter(
            RouterConfig(selector="lger", score="margin", aggregate="mean_layers", k=2)
        )(hidden, logits, coords, layers, attention_scores=attention)
        self.assertEqual(selected.patch_indices.tolist(), [2, 4])
        self.assertEqual(selected.patch_indices.unique().numel(), 2)
        torch.testing.assert_close(
            selected.hidden_states, hidden[-1].index_select(0, selected.patch_indices)
        )
        self.assertEqual(selected.layer_ids.tolist(), [12, 12])
        self.assertEqual(selected.evidence_count, 2)

    def test_random_selector_is_seed_deterministic(self) -> None:
        hidden, logits, coords, layers, attention = candidates()
        router = EvidenceRouter(
            RouterConfig(
                selector="random", score="maxprob", aggregate="last", k=3, seed=7
            )
        )
        first = router(hidden, logits, coords, layers, attention_scores=attention)
        second = router(hidden, logits, coords, layers, attention_scores=attention)
        self.assertEqual(first.patch_indices.tolist(), second.patch_indices.tolist())

    def test_random_context_never_duplicates_evidence(self) -> None:
        hidden, logits, coords, layers, attention = candidates()
        selected = EvidenceRouter(
            RouterConfig(
                selector="lger",
                score="margin",
                aggregate="persistent",
                k=2,
                context="random",
                context_k=2,
                seed=4,
            )
        )(hidden, logits, coords, layers, attention_scores=attention)
        self.assertEqual(selected.patch_indices.unique().numel(), 4)
        self.assertEqual(selected.evidence_count, 2)
        self.assertEqual(selected.context_count, 2)

    def test_attention_selector_requires_attention_scores(self) -> None:
        hidden, logits, coords, layers, _ = candidates()
        router = EvidenceRouter(
            RouterConfig(
                selector="attention", score="maxprob", aggregate="last", k=2
            )
        )
        with self.assertRaisesRegex(ValueError, "requires attention_scores"):
            router(hidden, logits, coords, layers)


if __name__ == "__main__":
    unittest.main()
