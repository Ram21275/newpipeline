from __future__ import annotations

import unittest

import torch

from cub_final.scoring import assert_final_logit_agreement


class _DeviceTaggedLogits:
    """Tensor-like input that only permits an explicit host comparison copy."""

    def __init__(self, values: list[float], logical_device: str) -> None:
        self._tensor = torch.tensor(values)
        self.logical_device = logical_device
        self.host_copy_requested = False

    @property
    def shape(self) -> torch.Size:
        return self._tensor.shape

    def detach(self) -> _DeviceTaggedLogits:
        return self

    def to(self, *, device: str, dtype: torch.dtype) -> torch.Tensor:
        if device != "cpu" or dtype != torch.float32:
            raise AssertionError("agreement validation must request an fp32 host copy")
        self.host_copy_requested = True
        return self._tensor.to(dtype=dtype)


class CubFinalScoringTests(unittest.TestCase):
    def test_final_logit_agreement_normalizes_sharded_devices_on_host(self) -> None:
        reconstructed = _DeviceTaggedLogits([1.0, 2.0, 3.0], "cuda:1")
        model = _DeviceTaggedLogits([1.0, 2.0, 3.0], "cuda:0")

        result = assert_final_logit_agreement(
            reconstructed,
            model,
            atol=1e-5,
            rtol=1e-5,
        )

        self.assertTrue(reconstructed.host_copy_requested)
        self.assertTrue(model.host_copy_requested)
        self.assertEqual(result["max_abs_error"], 0.0)
        self.assertEqual(result["mean_abs_error"], 0.0)

    def test_final_logit_agreement_still_fails_closed(self) -> None:
        with self.assertRaisesRegex(AssertionError, "final logit reconstruction"):
            assert_final_logit_agreement(
                torch.tensor([0.0, 1.0]),
                torch.tensor([0.0, 2.0]),
                atol=1e-5,
                rtol=1e-5,
            )


if __name__ == "__main__":
    unittest.main()
