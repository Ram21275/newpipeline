import unittest

import torch

from lger.probe import classification_metrics, jaccard, run_linear_probe


class ProbeTests(unittest.TestCase):
    def test_metrics_include_macro_f1(self) -> None:
        predictions = torch.tensor([0, 0, 1, 1])
        targets = torch.tensor([0, 1, 1, 1])
        accuracy, macro_f1 = classification_metrics(predictions, targets, 2)
        self.assertAlmostEqual(accuracy, 0.75)
        self.assertAlmostEqual(macro_f1, (2 / 3 + 0.8) / 2)

    def test_linear_probe_learns_separable_features(self) -> None:
        train_features = torch.tensor(
            [[-2.0, 0.0], [-1.0, 0.1], [-1.5, -0.1], [2.0, 0.0], [1.0, 0.1], [1.5, -0.1]]
        )
        train_targets = torch.tensor([0, 0, 0, 1, 1, 1])
        val_features = torch.tensor([[-1.2, 0.0], [-2.2, 0.1], [1.2, 0.0], [2.2, -0.1]])
        val_targets = torch.tensor([0, 0, 1, 1])
        result = run_linear_probe(
            train_features,
            train_targets,
            val_features,
            val_targets,
            num_classes=2,
            seed=0,
            epochs=100,
            learning_rate=0.05,
            weight_decay=0.0,
            device=torch.device("cpu"),
        )
        self.assertEqual(result.accuracy, 1.0)
        self.assertEqual(result.macro_f1, 1.0)

    def test_jaccard(self) -> None:
        self.assertAlmostEqual(
            jaccard(torch.tensor([1, 2, 3]), torch.tensor([2, 3, 4])), 0.5
        )


if __name__ == "__main__":
    unittest.main()
