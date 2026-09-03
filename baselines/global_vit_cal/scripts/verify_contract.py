from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from salt_vi_global_cal.losses import SuperellipseLossConfig  # noqa: E402
from salt_vi_global_cal.backbone import VisionTransformerConfig  # noqa: E402
from salt_vi_global_cal.pose_posterior import DEFAULT_TTA, PoseSupportConfig  # noqa: E402
from salt_vi_global_cal.qwen.pipeline import QwenSamplingConfig  # noqa: E402
from salt_vi_global_cal.token_dropout import TokenDropoutConfig  # noqa: E402
from salt_vi_global_cal.stage_b import StageBHardMiningConfig, stage_b_hard_weight  # noqa: E402


def main() -> None:
    document = yaml.safe_load((ROOT / "configs" / "salt_vi_global_cal.yaml").read_text())
    model = VisionTransformerConfig()
    dropout = TokenDropoutConfig()
    pose = PoseSupportConfig()
    ellipse = SuperellipseLossConfig()
    qwen = QwenSamplingConfig()
    stage_b = StageBHardMiningConfig()
    stage_b_profile = yaml.safe_load(
        (ROOT / "configs" / "stage_b_hard_mining.yaml").read_text()
    )

    assert model.grid_size == (28, 13) and model.patch_token_count == 364
    assert set(document["losses"]) == {
        "identity_weight", "triplet_weight", "cal_weight"
    }
    assert (dropout.target_fraction, dropout.gamma, dropout.p_max) == (0.15, 2.0, 0.95)
    assert (dropout.min_image_reliability, dropout.min_successful_tta) == (0.30, 4)
    assert len(DEFAULT_TTA) == 6 and pose.tau_aug == 0.020
    assert (
        ellipse.radius_x, ellipse.radius_y, ellipse.exponent,
        ellipse.temperature, ellipse.tolerance, ellipse.weight,
        ellipse.warmup_epochs,
    ) == (0.45, 0.50, 2.0, 0.12, 0.08, 0.10, 3)
    assert (
        qwen.atomic_sample_count, qwen.world_sample_count, qwen.max_worlds,
        qwen.similarity_threshold, qwen.max_attempts,
    ) == (8, 64, 8, 0.85, 4)
    assert document["qwen_roi"]["main_vit_uses_super_resolution"] is False
    assert document["qwen_roi"]["pose"]["detector"] == "yolo26x-pose"
    assert stage_b_profile["stage_b"]["cross_modal_hard"]["margin"] == stage_b.triplet_margin
    assert [stage_b_hard_weight(epoch) for epoch in (3, 4, 7)] == [0.0, 0.3125, 1.25]
    report = {
        "vit_grid": list(model.grid_size),
        "patch_tokens": model.patch_token_count,
        "baseline_objectives": ["ID", "triplet", "CAL"],
        "token_dropout": {
            "fraction": dropout.target_fraction,
            "gamma": dropout.gamma,
            "p_max": dropout.p_max,
            "hard_guard": [dropout.min_image_reliability, dropout.min_successful_tta],
            "tokenwise_reliability_power": False,
        },
        "pose_tta": len(DEFAULT_TTA),
        "superellipse_validated_exponent": ellipse.exponent,
        "qwen_sampling": [
            qwen.atomic_sample_count, qwen.world_sample_count, qwen.max_worlds,
        ],
        "stage_b_code_profile": {
            "hard_ramp": [stage_b.hard_start_epoch, stage_b.hard_ramp_epochs],
            "qwen_mode": "offline_optional",
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
