import sys
import unittest
from collections import defaultdict
from pathlib import Path

from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from salt_vi_global_cal.qwen.hypotheses import parse_atom  # noqa: E402
from salt_vi_global_cal.qwen.pipeline import QwenRegionalPipeline  # noqa: E402
from salt_vi_global_cal.qwen.roi import (  # noqa: E402
    pose_regions,
    score_region_uncertainty,
    select_uncertain_regions,
)
from salt_vi_global_cal.qwen.schema import Region  # noqa: E402
from salt_vi_global_cal.pose_posterior import PoseObservation  # noqa: E402
from salt_vi_global_cal.qwen.track_anchor import (  # noqa: E402
    TrackFrame,
    select_shared_region_ids,
    select_track_anchor,
)
from salt_vi_global_cal.qwen.zoom import expanded_bbox, swin_roi_board  # noqa: E402


class FakeBackend:
    model_id = "fake-qwen"

    def __init__(self):
        self.calls = defaultdict(int)

    def sample_atomic(
        self,
        full_swin,
        roi_board,
        region,
        *,
        modality,
        observed,
        system_prompt,
        seed,
        temperature,
        thinking,
    ):
        self.assertions(full_swin, roi_board, temperature, thinking, system_prompt)
        index = self.calls[region.region_id]
        self.calls[region.region_id] += 1
        if region.category == "headwear":
            if index < 6:
                return "ATOM | headwear | cap | dark cap | head"
            return "ATOM | headwear | hood | light hood | head"
        if region.category == "carried_object" and index < 5:
            return "ATOM | carried_object | backpack | small bag | right hand"
        if region.category == "carried_object":
            return "ATOM | carried_object | other_carried_object | long package | right hand"
        if region.category == "wrist_accessory":
            return "ATOM | wrist_accessory | watch | dark watch | wrist"
        return f"ATOM | {region.category} | no_additional_detail | none | none"

    @staticmethod
    def assertions(full_swin, board, temperature, thinking, prompt):
        if full_swin.size != (256, 512) or board.size != (512, 512):
            raise RuntimeError("incorrect visual context")
        if temperature != 0.75 or thinking or "Never return prose" not in prompt:
            raise RuntimeError("incorrect atomic decoding contract")


class QwenSamplingTests(unittest.TestCase):
    def regions(self):
        return [
            Region("head", "headwear", (80, 20, 150, 100), ("cap", "hood", "absent")),
            Region(
                "right_hand",
                "carried_object",
                (140, 180, 220, 310),
                ("backpack", "other_carried_object", "absent"),
            ),
        ]

    def test_zoom_expands_and_clips(self):
        self.assertEqual(expanded_bbox((0, 0, 20, 20), (100, 100)), (0, 0, 28, 28))
        board = swin_roi_board(Image.new("RGB", (256, 512)), self.regions()[0])
        self.assertEqual(board.size, (512, 512))

    def test_wholebody_pose_builds_audited_roi_taxonomy(self):
        points = torch.zeros(133, 3)
        body_xy = torch.tensor([
            [128, 45], [115, 42], [141, 42], [105, 48], [151, 48],
            [92, 120], [164, 120], [75, 205], [181, 205],
            [65, 285], [191, 285], [105, 285], [151, 285],
            [100, 390], [156, 390], [95, 485], [161, 485],
        ], dtype=torch.float32)
        points[:17, :2] = body_xy
        points[:17, 2] = 0.9
        observation = PoseObservation(
            points, torch.tensor([40.0, 10.0, 216.0, 505.0])
        )
        regions = pose_regions(observation)
        self.assertEqual(len(regions), 13)
        self.assertTrue(all(region.validate() is region for region in regions))
        self.assertTrue(all(region.mask.any() for region in regions))
        self.assertIn("eyes", {region.region_id for region in regions})
        self.assertIn("right_foot", {region.region_id for region in regions})
        scored = score_region_uncertainty(
            Image.new("RGB", (128, 256), (100, 100, 100)),
            Image.new("RGB", (256, 512), (110, 110, 110)),
            regions,
        )
        self.assertTrue(all(0.0 <= region.uncertainty <= 1.0 for region in scored))
        selected = select_uncertain_regions(scored, selected_count=3)
        self.assertEqual(
            [region.category for region in selected],
            ["carried_object", "carried_object", "wrist_accessory"],
        )
        integrated = QwenRegionalPipeline(FakeBackend()).run_from_pose(
            Image.new("RGB", (128, 256), (100, 100, 100)),
            Image.new("RGB", (256, 512), (110, 110, 110)),
            observation,
            modality="ir",
            observed="person observation",
            source_key="cam1/pose-driven.jpg",
            seed=31,
        )
        self.assertEqual(len(integrated["roi_selection"]), 3)
        self.assertEqual(len(integrated["regions"]), 3)
        self.assertEqual(integrated["worlds"]["sample_count"], 64)

    def test_repeated_regional_hypotheses_and_joint_worlds(self):
        pipeline = QwenRegionalPipeline(FakeBackend())
        result = pipeline.run(
            Image.new("RGB", (256, 512), (120, 120, 120)),
            self.regions(),
            modality="ir",
            observed="person observation",
            source_key="cam1/0001.jpg",
            seed=17,
        )
        self.assertEqual(result["version"], "empirical-atomic-v1")
        self.assertEqual(len(result["regions"]), 2)
        self.assertEqual(result["regions"][0]["scheduled_sample_count"], 8)
        self.assertEqual(len(result["regions"][0]["clusters"]), 2)
        weights = [item["weight"] for item in result["regions"][0]["clusters"]]
        self.assertEqual(sorted(weights), [0.25, 0.75])
        worlds = result["worlds"]
        self.assertEqual(worlds["sample_count"], 64)
        self.assertLessEqual(worlds["retained_world_count"], 8)
        self.assertAlmostEqual(
            sum(world["selected_weight"] for world in worlds["worlds"]), 1.0
        )

    def test_self_reported_probability_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_atom(
                "ATOM | headwear | cap | dark cap probability 0.8 | head",
                expected_category="headwear",
                allowed_states=("cap", "absent"),
            )

    def test_track_anchor_contract(self):
        region_a = Region(
            "head", "headwear", (0, 0, 10, 10), ("cap",), 0.8, 0.4
        )
        region_b = Region(
            "hand", "carried_object", (10, 10, 20, 20), ("backpack",), 0.6, 0.2
        )
        frames = [
            TrackFrame("a", 0.2, (region_a, region_b)),
            TrackFrame("b", 0.4, (region_a, region_b)),
        ]
        selected = select_shared_region_ids(frames, 2)
        self.assertEqual(selected, ["head", "hand"])
        self.assertEqual(select_track_anchor(frames, selected).source_key, "b")


if __name__ == "__main__":
    unittest.main()
