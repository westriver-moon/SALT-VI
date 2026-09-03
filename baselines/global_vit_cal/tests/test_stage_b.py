import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from salt_vi_global_cal.qwen.offline_text import retained_world_texts, weighted_world_feature_mean  # noqa: E402
from salt_vi_global_cal.stage_b import stage_b_hard_weight, weighted_cross_modal_hard_loss  # noqa: E402


class StageBProfileTests(unittest.TestCase):
    def test_historical_ramp(self):
        self.assertEqual(stage_b_hard_weight(3), 0.0)
        self.assertAlmostEqual(stage_b_hard_weight(4), 0.3125)
        self.assertAlmostEqual(stage_b_hard_weight(7), 1.25)

    def test_recorded_active_pairs_only(self):
        labels = torch.tensor([0, 0, 1, 1])
        base = torch.tensor([[0.0, 0.0], [0.1, 0.0], [3.0, 0.0], [3.1, 0.0]])
        modalities = {"RGB": base, "IR": base + 0.02, "Fusion": base + 5.0, "Text": base + 0.04}
        loss, pairs = weighted_cross_modal_hard_loss(modalities, labels)
        active = (pairs["RGB-IR"] + pairs["RGB-Text"] + pairs["IR-Text"]) / 3.0
        self.assertTrue(torch.allclose(loss, active))

    def test_qwen_worlds_become_weighted_text_features(self):
        record = {"worlds": {"worlds": [
            {"world_id": "w00", "selected_weight": 0.75, "assignments": [{"category": "headwear", "state": "cap", "description": "ATOM | headwear | cap | dark cap | head"}]},
            {"world_id": "w01", "selected_weight": 0.25, "assignments": [{"category": "headwear", "state": "hood", "description": "ATOM | headwear | hood | light hood | head"}]},
        ]}}
        worlds = retained_world_texts(record)
        features = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        weights = torch.tensor([[worlds[0].weight, worlds[1].weight]])
        self.assertTrue(torch.allclose(weighted_world_feature_mean(features, weights), torch.tensor([[0.75, 0.25]])))


if __name__ == "__main__": unittest.main()
