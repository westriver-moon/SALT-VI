from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .losses import CenterAggregationLoss, SoftSuperellipseAttentionLoss


def batch_hard_triplet(features: Tensor, labels: Tensor, margin: float = 0.3) -> Tensor:
    distances = torch.cdist(features.float(), features.float())
    same = labels[:, None].eq(labels[None, :])
    same.fill_diagonal_(False)
    different = ~labels[:, None].eq(labels[None, :])
    if not same.any(dim=1).all() or not different.any(dim=1).all():
        raise ValueError("batch-hard triplet needs a positive and negative per sample")
    hardest_positive = distances.masked_fill(~same, -torch.inf).max(dim=1).values
    hardest_negative = distances.masked_fill(~different, torch.inf).min(dim=1).values
    return F.relu(hardest_positive - hardest_negative + margin).mean()


@dataclass(frozen=True)
class ObjectiveWeights:
    identity: float = 1.0
    triplet: float = 1.0
    cal: float = 1.0


class SALTGlobalObjective(nn.Module):
    """Global identity/triplet objective with CAL and the geometry loss."""

    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        *,
        weights: ObjectiveWeights | None = None,
        attention_loss: SoftSuperellipseAttentionLoss | None = None,
    ):
        super().__init__()
        self.classifier = nn.Linear(embed_dim, num_classes, bias=False)
        self.cal = CenterAggregationLoss()
        self.attention_loss = attention_loss or SoftSuperellipseAttentionLoss()
        self.weights = weights or ObjectiveWeights()

    def _id_triplet(self, features: Tensor, labels: Tensor) -> Tensor:
        return (
            F.cross_entropy(self.classifier(features), labels) * self.weights.identity
            + batch_hard_triplet(features, labels) * self.weights.triplet
        )

    def forward(
        self,
        visual: dict[str, Tensor | tuple[int, int]],
        labels: Tensor,
        *,
        current_epoch: int | None = None,
    ) -> dict[str, Tensor]:
        required = {"features", "ellipse_attention", "grid_size"}
        missing = required - set(visual)
        if missing:
            raise ValueError(f"training output omits {sorted(missing)}")
        global_features = visual["features"]
        if not isinstance(global_features, Tensor):
            raise TypeError("global visual features must be a tensor")
        identity_triplet = self._id_triplet(global_features, labels)
        cal_loss = self.cal(global_features, labels) * self.weights.cal
        attention = visual["ellipse_attention"]
        grid_size = visual["grid_size"]
        if not isinstance(attention, Tensor) or not isinstance(grid_size, tuple):
            raise TypeError("invalid attention payload")
        ellipse, outside = self.attention_loss(
            attention, grid_size, current_epoch=current_epoch
        )
        return {
            "global_identity_triplet_loss": identity_triplet,
            "cal_loss": cal_loss,
            "superellipse_attention_loss": ellipse,
            "superellipse_outside_mass": outside,
            "total": identity_triplet + cal_loss + ellipse,
        }
