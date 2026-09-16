from __future__ import annotations

import unittest

import torch

from cub_final.models import FrozenVlmRunner


class _AddOne(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + 1


class CubFinalModelProjectionTests(unittest.TestCase):
    @staticmethod
    def _runner(architecture: str) -> FrozenVlmRunner:
        runner = object.__new__(FrozenVlmRunner)
        runner.architecture = architecture
        runner.final_norm = _AddOne()
        runner.lm_head = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            runner.lm_head.weight.copy_(torch.eye(2))
        return runner

    def test_qwen3_final_capture_requires_final_norm(self) -> None:
        runner = self._runner("qwen3")
        self.assertTrue(runner.final_state_requires_norm)
        result = runner._project_state(
            torch.tensor([2.0, 4.0]),
            apply_final_norm=runner.final_state_requires_norm,
        )
        torch.testing.assert_close(result, torch.tensor([3.0, 5.0]))

    def test_llava_final_capture_is_projected_directly(self) -> None:
        runner = self._runner("llava")
        self.assertFalse(runner.final_state_requires_norm)
        result = runner._project_state(
            torch.tensor([2.0, 4.0]),
            apply_final_norm=runner.final_state_requires_norm,
        )
        torch.testing.assert_close(result, torch.tensor([2.0, 4.0]))


if __name__ == "__main__":
    unittest.main()
