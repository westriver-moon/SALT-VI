import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from salt_vi_global_cal.token_dropout import (  # noqa: E402
    ConfidenceTokenDropout,
    TokenDropoutConfig,
    effective_support,
)


class TokenDropoutTests(unittest.TestCase):
    def test_final_formula_and_hard_guard(self):
        tokens = 210
        pose = torch.linspace(0.05, 0.95, tokens).repeat(3, 1)
        ellipse = torch.full_like(pose, 0.55)
        reliability = torch.tensor([0.8, 0.2, 0.9])
        tta = torch.tensor([6, 6, 3])
        module = ConfidenceTokenDropout()
        result = module(pose, ellipse, reliability, tta)
        expected = reliability[:, None] * pose + (1.0 - reliability[:, None]) * ellipse
        torch.testing.assert_close(result.effective_support, expected)
        expected_probability = torch.clamp(
            result.rho * (1.0 - expected).pow(2.0), max=0.95
        )
        expected_probability[1:] = 0.0
        torch.testing.assert_close(result.probabilities, expected_probability)
        self.assertEqual(int(result.drop_mask[0].sum()), 32)
        self.assertEqual(int(result.drop_mask[1].sum()), 0)
        self.assertEqual(int(result.drop_mask[2].sum()), 0)

    def test_364_token_rate_is_55(self):
        pose = torch.linspace(0.01, 0.99, 364)[None]
        ellipse = torch.full_like(pose, 0.5)
        result = ConfidenceTokenDropout()(pose, ellipse, 0.8, 6)
        self.assertEqual(int(result.drop_mask.sum()), 55)
        self.assertEqual(result.keep_mask.shape, (1, 364))

    def test_layerwise_rho_and_rates(self):
        pose = torch.linspace(0.01, 0.99, 100)[None]
        ellipse = torch.full_like(pose, 0.5)
        result = ConfidenceTokenDropout().forward_layers(
            pose,
            ellipse,
            0.8,
            6,
            layer_count=3,
            target_fractions=(0.10, 0.15, 0.20),
        )
        self.assertEqual(result.keep_masks.shape, (3, 1, 100))
        self.assertEqual(result.drop_masks.sum(dim=(1, 2)).tolist(), [10, 15, 20])
        self.assertLess(result.rho_by_layer[0], result.rho_by_layer[2])

    def test_effective_support_validates_shape(self):
        with self.assertRaises(ValueError):
            effective_support(torch.zeros(2, 3), torch.zeros(2, 4), 0.5)

    def test_bernoulli_mode_obeys_guard(self):
        cfg = TokenDropoutConfig(selection="bernoulli")
        module = ConfidenceTokenDropout(cfg)
        pose = torch.zeros(2, 50)
        ellipse = torch.zeros_like(pose)
        generator = torch.Generator().manual_seed(3)
        result = module(
            pose, ellipse, torch.tensor([0.9, 0.1]), 6, generator=generator
        )
        self.assertFalse(bool(result.drop_mask[1].any()))


if __name__ == "__main__":
    unittest.main()
