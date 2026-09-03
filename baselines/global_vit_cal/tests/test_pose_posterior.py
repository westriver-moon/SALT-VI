import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from salt_vi_global_cal.pose_posterior import (  # noqa: E402
    PoseObservation,
    TTASpec,
    affine_tta_matrix,
    build_pose_posterior,
    transform_points,
)
from salt_vi_global_cal.token_dropout import ConfidenceTokenDropout  # noqa: E402


class PosePosteriorTests(unittest.TestCase):
    def observation(self, jitter=0.0):
        points = torch.zeros(133, 3)
        body_xy = torch.tensor([
            [16, 8], [14, 9], [18, 9], [12, 10], [20, 10],
            [10, 15], [22, 15], [8, 25], [24, 25], [6, 35], [26, 35],
            [12, 32], [20, 32], [11, 45], [21, 45], [10, 58], [22, 58],
        ], dtype=torch.float32)
        points[:17, :2] = body_xy + jitter
        points[:17, 2] = 0.9
        bbox = torch.tensor([4, 4, 28, 60], dtype=torch.float32) + jitter
        return PoseObservation(points, bbox, structural_reliability=0.8)

    def test_tta_inverse_restores_points(self):
        points = torch.tensor([[3.0, 7.0], [11.0, 13.0]])
        matrix = affine_tta_matrix(
            32, 64, TTASpec("flip_shift", tx=0.04, flip=True)
        )
        recovered = transform_points(
            transform_points(points, matrix), torch.linalg.inv(matrix)
        )
        torch.testing.assert_close(recovered, points, atol=1e-5, rtol=1e-5)

    def test_dense_support_to_fixed_vit_tokens(self):
        observations = [
            self.observation(value) for value in (0.0, 0.2, -0.2, 0.1)
        ]
        posterior = build_pose_posterior(
            observations,
            source_hw=(64, 32),
            output_hw=(32, 16),
            patch_hw=(4, 4),
            stride_hw=(4, 4),
            total_tta=6,
        )
        self.assertEqual(posterior.token_grid, (8, 4))
        self.assertEqual(posterior.pose_support.shape, (32,))
        self.assertEqual(posterior.ellipse_support.shape, (32,))
        self.assertEqual(posterior.successful_tta, 4)
        self.assertGreater(posterior.image_reliability, 0.3)
        result = ConfidenceTokenDropout()(
            posterior.pose_support[None],
            posterior.ellipse_support[None],
            posterior.image_reliability,
            posterior.successful_tta,
        )
        self.assertEqual(int(result.drop_mask.sum()), 5)


if __name__ == "__main__":
    unittest.main()
