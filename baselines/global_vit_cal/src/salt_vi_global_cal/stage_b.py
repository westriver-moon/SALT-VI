"""Reusable Stage-B cross-modal hard-mining components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as F
from torch import Tensor


STAGE_B_PAIR_NAMES = (
    "RGB-IR", "RGB-Fusion", "RGB-Text", "IR-Fusion", "IR-Text", "Fusion-Text",
)


@dataclass(frozen=True)
class StageBHardMiningConfig:
    """Stage-B cross-modal hard-mining settings independent of a runner."""

    hard_weight: float = 1.25
    hard_start_epoch: int = 3
    hard_ramp_epochs: int = 5
    triplet_margin: float = 0.10
    rgb_ir_weight: float = 1.0
    rgb_fusion_weight: float = 0.0
    rgb_text_weight: float = 1.0
    ir_fusion_weight: float = 0.0
    ir_text_weight: float = 1.0
    fusion_text_weight: float = 0.0

    def validate(self) -> "StageBHardMiningConfig":
        if self.hard_weight < 0.0 or self.triplet_margin < 0.0:
            raise ValueError("Stage-B weights and margin must be non-negative")
        if self.hard_start_epoch < 0 or self.hard_ramp_epochs < 0:
            raise ValueError("Stage-B epochs must be non-negative")
        if any(value < 0.0 for value in self.pair_weights().values()):
            raise ValueError("Stage-B pair weights must be non-negative")
        if sum(self.pair_weights().values()) <= 0.0:
            raise ValueError("at least one Stage-B pair weight must be positive")
        return self

    def pair_weights(self) -> dict[str, float]:
        return {
            "RGB-IR": self.rgb_ir_weight,
            "RGB-Fusion": self.rgb_fusion_weight,
            "RGB-Text": self.rgb_text_weight,
            "IR-Fusion": self.ir_fusion_weight,
            "IR-Text": self.ir_text_weight,
            "Fusion-Text": self.fusion_text_weight,
        }


def stage_b_hard_weight(current_epoch: int | None, config: StageBHardMiningConfig | None = None) -> float:
    """Return the historical Stage-B hard-mining ramp coefficient."""

    cfg = (config or StageBHardMiningConfig()).validate()
    if current_epoch is None:
        return cfg.hard_weight
    epoch = int(current_epoch)
    if cfg.hard_ramp_epochs <= 1:
        return cfg.hard_weight if epoch >= cfg.hard_start_epoch else 0.0
    progress = (epoch - cfg.hard_start_epoch) / float(cfg.hard_ramp_epochs - 1)
    return cfg.hard_weight * min(1.0, max(0.0, progress))


def _pairwise_distance(left: Tensor, right: Tensor) -> Tensor:
    """Match the historical PMT Euclidean-distance calculation."""

    rows, columns = left.shape[0], right.shape[0]
    left_sq = left.square().sum(dim=1, keepdim=True).expand(rows, columns)
    right_sq = right.square().sum(dim=1, keepdim=True).expand(columns, rows).t()
    return (left_sq + right_sq - 2.0 * left @ right.t()).clamp_min(1.0e-12).sqrt()


def _directional_hard_triplet(distance: Tensor, labels: Tensor, margin: float) -> Tensor:
    positive = labels[:, None].eq(labels[None, :])
    negative = ~positive
    valid = positive.any(dim=1) & negative.any(dim=1)
    if not bool(valid.all()):
        invalid = torch.where(~valid)[0].detach().cpu().tolist()
        raise ValueError(
            "cross-modal hard mining needs a positive and a negative for every "
            f"anchor; invalid indices={invalid[:20]}"
        )
    positive_distance = distance.masked_fill(~positive, -torch.inf).max(dim=1).values
    negative_distance = distance.masked_fill(~negative, torch.inf).min(dim=1).values
    return F.margin_ranking_loss(
        negative_distance[valid], positive_distance[valid],
        torch.ones_like(negative_distance[valid]), margin=float(margin),
    )


def cross_modal_hard_triplet(left: Tensor, right: Tensor, labels: Tensor, *, margin: float = 0.10) -> Tensor:
    """Bidirectional hard triplet loss used by the recorded Stage-B run."""

    if left.ndim != 2 or right.ndim != 2 or left.shape != right.shape:
        raise ValueError("cross-modal hard triplet expects matching [B,D] features")
    labels = labels.view(-1).to(device=left.device, dtype=torch.long)
    if labels.numel() != left.shape[0]:
        raise ValueError("labels must have one identity per feature row")
    if left.shape[0] < 2 or not torch.isfinite(left).all() or not torch.isfinite(right).all():
        raise ValueError("cross-modal hard triplet needs finite features from two identities")
    distances = _pairwise_distance(left, right)
    return (
        _directional_hard_triplet(distances, labels, margin)
        + _directional_hard_triplet(distances.t(), labels, margin)
    ) * 0.5


def weighted_cross_modal_hard_loss(
    modalities: Mapping[str, Tensor], labels: Tensor, *, config: StageBHardMiningConfig | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Evaluate the six Stage-B pairs and apply the recorded pair weights."""

    cfg = (config or StageBHardMiningConfig()).validate()
    expected = {"RGB", "IR", "Fusion", "Text"}
    if set(modalities) != expected:
        raise ValueError(f"Stage-B modalities must be exactly {sorted(expected)}")
    shapes = {name: tuple(value.shape) for name, value in modalities.items()}
    if len(set(shapes.values())) != 1:
        raise ValueError(f"Stage-B modality features are not aligned: {shapes}")
    pair_losses: dict[str, Tensor] = {}
    names = ("RGB", "IR", "Fusion", "Text")
    for index, left_name in enumerate(names):
        for right_name in names[index + 1 :]:
            pair_name = f"{left_name}-{right_name}"
            pair_losses[pair_name] = cross_modal_hard_triplet(
                modalities[left_name], modalities[right_name], labels, margin=cfg.triplet_margin,
            )
    weights = cfg.pair_weights()
    values = torch.stack([pair_losses[name] for name in STAGE_B_PAIR_NAMES])
    weight_tensor = values.new_tensor([weights[name] for name in STAGE_B_PAIR_NAMES])
    return (values * weight_tensor).sum() / weight_tensor.sum(), pair_losses
