import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from salt_vi_global_cal.losses import (  # noqa: E402
    SoftSuperellipseAttentionLoss,
    SuperellipseLossConfig,
)


class SuperellipseLossTests(unittest.TestCase):
    def test_validated_p2_mask_matches_exact_equation(self):
        module = SoftSuperellipseAttentionLoss()
        mask = module.mask((4, 2), torch.device("cpu"))
        y = (torch.arange(4, dtype=torch.float32) + 0.5) / 4
        x = (torch.arange(2, dtype=torch.float32) + 0.5) / 2
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        distance = ((xx - 0.5) / 0.45).square() + ((yy - 0.5) / 0.50).square()
        expected = torch.sigmoid((1.0 - distance) / 0.12).flatten()
        torch.testing.assert_close(mask, expected)

    def test_three_epoch_warmup(self):
        module = SoftSuperellipseAttentionLoss()
        self.assertAlmostEqual(module.epoch_weight(0), 0.1 / 3.0)
        self.assertAlmostEqual(module.epoch_weight(1), 0.2 / 3.0)
        self.assertAlmostEqual(module.epoch_weight(2), 0.1)
        attention = torch.full((2, 3, 8), 1.0 / 8.0)
        loss, outside = module(attention, (4, 2), current_epoch=2)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(outside))

    def test_other_exponent_is_explicit_ablation(self):
        config = SuperellipseLossConfig(exponent=4.0)
        module = SoftSuperellipseAttentionLoss(config)
        self.assertEqual(module.config.exponent, 4.0)


if __name__ == "__main__":
    unittest.main()
