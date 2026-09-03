from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from salt_vi_global_cal.backbone import (  # noqa: E402
    SALTGlobalVisionTransformer,
    VisionTransformerConfig,
)
from salt_vi_global_cal.recipe import SALTGlobalObjective  # noqa: E402
from salt_vi_global_cal.token_dropout import ConfidenceTokenDropout  # noqa: E402


def main() -> None:
    torch.manual_seed(7)
    config = VisionTransformerConfig(
        image_size_hw=(32, 16),
        patch_size=4,
        patch_stride=4,
        embed_dim=16,
        depth=3,
        heads=4,
        ellipse_attention_layer=2,
    )
    images = torch.randn(4, 3, 32, 16)
    pose = torch.rand(4, 32)
    ellipse = torch.full_like(pose, 0.5)
    dropout = ConfidenceTokenDropout().forward_layers(
        pose, ellipse, 0.8, 6, layer_count=config.depth
    )
    model = SALTGlobalVisionTransformer(config).train()
    visual = model(images, dropout.keep_masks)
    labels = torch.tensor([0, 1, 0, 1])
    losses = SALTGlobalObjective(16, 2)(visual, labels, current_epoch=0)
    losses["total"].backward()
    report = {
        "patch_tokens": config.patch_token_count,
        "dropped_per_layer_and_image": dropout.drop_masks.sum(dim=2).tolist(),
        "rho_by_layer": list(dropout.rho_by_layer),
        "global_feature_shape": list(visual["features"].shape),
        "finite_total_loss": bool(torch.isfinite(losses["total"])),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
