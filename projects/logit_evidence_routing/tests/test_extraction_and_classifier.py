import unittest

import torch
from torch import nn

from lger import EvidenceClassifier, FrozenLogitProjector, gather_visual_hidden_states


class ExtractionAndClassifierTests(unittest.TestCase):
    def test_visual_token_gather_excludes_prompt_positions(self) -> None:
        states = [
            torch.arange(6 * 3, dtype=torch.float32).reshape(1, 6, 3) + 100 * i
            for i in range(3)
        ]
        indices = torch.tensor([1, 3, 4])
        gathered = gather_visual_hidden_states(states, indices, [1, 2])
        self.assertEqual(gathered.shape, (2, 3, 3))
        torch.testing.assert_close(gathered[0], states[1][0, indices])

    def test_logit_projector_stays_frozen(self) -> None:
        projector = FrozenLogitProjector(
            nn.LayerNorm(4), nn.Linear(4, 7, bias=False)
        )
        hidden = torch.randn(2, 3, 4, requires_grad=True)
        logits = projector(hidden)
        self.assertEqual(logits.shape, (2, 3, 7))
        self.assertFalse(logits.requires_grad)
        self.assertTrue(
            all(not parameter.requires_grad for parameter in projector.parameters())
        )
        projector.train()
        self.assertFalse(projector.training)

    def test_classifier_is_trainable(self) -> None:
        classifier = EvidenceClassifier(
            hidden_dim=4,
            num_classes=3,
            model_dim=16,
            num_layers=1,
            num_heads=4,
            dropout=0.0,
        )
        hidden = torch.randn(5, 4)
        coords = torch.tensor(
            [[0, 0], [1, 0], [2, 0], [0, 1], [1, 1]]
        ).float()
        layers = torch.full((5,), 12)
        output = classifier(hidden, coords, layers)
        self.assertEqual(output.shape, (3,))
        output.sum().backward()
        self.assertTrue(
            any(parameter.grad is not None for parameter in classifier.parameters())
        )


if __name__ == "__main__":
    unittest.main()
