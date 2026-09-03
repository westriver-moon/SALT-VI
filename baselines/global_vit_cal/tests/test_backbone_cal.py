import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from salt_vi_global_cal.backbone import (  # noqa: E402
    SALTGlobalVisionTransformer,
    VisionTransformerConfig,
)
from salt_vi_global_cal.losses import CenterAggregationLoss  # noqa: E402
from salt_vi_global_cal.recipe import SALTGlobalObjective  # noqa: E402


class BackboneAndCALTests(unittest.TestCase):
    def tiny_model(self):
        return SALTGlobalVisionTransformer(
            VisionTransformerConfig(
                image_size_hw=(32, 16),
                patch_size=4,
                patch_stride=4,
                embed_dim=16,
                depth=3,
                heads=4,
                ellipse_attention_layer=2,
            )
        )

    def test_single_global_branch_and_fixed_token_mask(self):
        model = self.tiny_model().train()
        images = torch.randn(4, 3, 32, 16)
        keep = torch.ones(4, 32, dtype=torch.bool)
        keep[:, 7] = False
        output = model(images, keep)
        self.assertEqual(set(output), {
            "tokens", "features", "ellipse_attention", "grid_size"
        })
        self.assertEqual(output["tokens"].shape, (4, 33, 16))
        self.assertEqual(output["features"].shape, (4, 16))
        self.assertEqual(output["ellipse_attention"].shape, (4, 4, 32))
        torch.testing.assert_close(
            output["ellipse_attention"][:, :, 7], torch.zeros(4, 4)
        )

    def test_layerwise_masks_keep_positions(self):
        model = self.tiny_model().train()
        masks = torch.ones(3, 2, 32, dtype=torch.bool)
        masks[0, :, 1] = False
        masks[1, :, 2] = False
        masks[2, :, 3] = False
        output = model(torch.randn(2, 3, 32, 16), masks)
        self.assertEqual(output["tokens"].shape, (2, 33, 16))
        torch.testing.assert_close(
            output["ellipse_attention"][:, :, 2], torch.zeros(2, 4)
        )

    def test_cal_exact_contract_and_label_guard(self):
        labels_half = torch.tensor([0, 0, 1, 1])
        labels = torch.cat((labels_half, labels_half))
        features = torch.tensor(
            [
                [0.0, 0.0], [0.1, 0.0], [2.0, 0.0], [2.1, 0.0],
                [0.0, 0.2], [0.1, 0.2], [2.0, 0.2], [2.1, 0.2],
            ],
            requires_grad=True,
        )
        value = CenterAggregationLoss()(features, labels)
        self.assertTrue(torch.isfinite(value))
        self.assertGreater(float(value), 0.0)
        value.backward()
        self.assertIsNotNone(features.grad)
        bad = labels.clone()
        bad[-1] = 0
        with self.assertRaises(ValueError):
            CenterAggregationLoss()(features.detach(), bad)

    def test_objective_consumes_only_global_backbone_outputs(self):
        model = self.tiny_model().train()
        visual = model(torch.randn(4, 3, 32, 16))
        labels = torch.tensor([0, 1, 0, 1])
        losses = SALTGlobalObjective(embed_dim=16, num_classes=2)(
            visual, labels, current_epoch=0
        )
        self.assertEqual(
            set(losses),
            {
                "global_identity_triplet_loss",
                "cal_loss",
                "superellipse_attention_loss",
                "superellipse_outside_mass",
                "total",
            },
        )
        self.assertTrue(torch.isfinite(losses["total"]))
        losses["total"].backward()


if __name__ == "__main__":
    unittest.main()
