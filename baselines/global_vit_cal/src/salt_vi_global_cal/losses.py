from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class CenterAggregationLoss(nn.Module):
    """Exact CAL reduction from the audited released implementation."""

    def __init__(self, eps: float = 1.0e-12):
        super().__init__()
        self.eps = float(eps)
        if self.eps <= 0.0:
            raise ValueError("CAL epsilon must be positive")

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        if inputs.ndim != 2:
            raise ValueError("CAL expects a [2B,D] feature matrix")
        targets = targets.view(-1)
        if targets.numel() != inputs.shape[0] or inputs.shape[0] % 2:
            raise ValueError("CAL expects matching even-sized features and labels")
        half = inputs.shape[0] // 2
        rgb_labels, ir_labels = targets[:half], targets[half:]
        if not torch.equal(rgb_labels, ir_labels):
            raise ValueError("CAL requires aligned RGB and IR label order")
        features = inputs.float()
        rgb_features, ir_features = features[:half], features[half:]
        identities, inverse = torch.unique(rgb_labels, sorted=True, return_inverse=True)
        if identities.numel() < 2:
            raise ValueError("CAL requires at least two identities per batch")
        rgb_centers = torch.stack(
            [rgb_features[rgb_labels == identity].mean(0) for identity in identities]
        )
        ir_centers = torch.stack(
            [ir_features[ir_labels == identity].mean(0) for identity in identities]
        )
        rgb_sample_centers = rgb_centers[inverse]
        ir_sample_centers = ir_centers[inverse]
        paired = torch.linalg.vector_norm(rgb_sample_centers - ir_sample_centers, dim=1)
        repeated_rgb = torch.cat((rgb_sample_centers, rgb_sample_centers), dim=0)
        repeated_ir = torch.cat((ir_sample_centers, ir_sample_centers), dim=0)
        negative = targets[:, None].ne(targets[None, :])
        negative_count = negative.sum(dim=1).clamp_min(1)
        rgb_negative_mean = (
            torch.cdist(repeated_rgb, features).masked_fill(~negative, 0.0).sum(dim=1)
            / negative_count
        )
        ir_negative_mean = (
            torch.cdist(repeated_ir, features).masked_fill(~negative, 0.0).sum(dim=1)
            / negative_count
        )
        denominator = rgb_negative_mean.sum() + ir_negative_mean.sum() - paired.sum()
        return paired.sum() / denominator.clamp_min(self.eps)


@dataclass(frozen=True)
class SuperellipseLossConfig:
    radius_x: float = 0.45
    radius_y: float = 0.50
    exponent: float = 2.0
    temperature: float = 0.12
    tolerance: float = 0.08
    weight: float = 0.10
    warmup_epochs: int = 3


class SoftSuperellipseAttentionLoss(nn.Module):
    """Soft superellipse-family prior; p=2 is the validated experiment."""

    def __init__(self, config: SuperellipseLossConfig | None = None):
        super().__init__()
        self.config = config or SuperellipseLossConfig()
        cfg = self.config
        if min(cfg.radius_x, cfg.radius_y, cfg.exponent, cfg.temperature) <= 0.0:
            raise ValueError("radii, exponent and temperature must be positive")
        if cfg.tolerance < 0.0 or cfg.weight < 0.0 or cfg.warmup_epochs < 0:
            raise ValueError("tolerance, weight and warmup_epochs must be non-negative")

    def mask(self, grid_size: tuple[int, int], device: torch.device) -> Tensor:
        height, width = (int(value) for value in grid_size)
        y = (torch.arange(height, device=device, dtype=torch.float32) + 0.5) / height
        x = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) / width
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        cfg = self.config
        distance = ((xx - 0.5).abs() / cfg.radius_x).pow(cfg.exponent)
        distance = distance + ((yy - 0.5).abs() / cfg.radius_y).pow(cfg.exponent)
        return torch.sigmoid((1.0 - distance) / cfg.temperature).flatten()

    def epoch_weight(self, current_epoch: int | None) -> float:
        cfg = self.config
        if current_epoch is None or cfg.warmup_epochs <= 1:
            return cfg.weight
        progress = min(1.0, max(0.0, (int(current_epoch) + 1) / cfg.warmup_epochs))
        return cfg.weight * progress

    def forward(
        self,
        cls_patch_attention: Tensor,
        grid_size: tuple[int, int],
        *,
        current_epoch: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        attention = cls_patch_attention
        if attention.ndim == 3:
            attention = attention.mean(dim=1)
        if attention.ndim != 2:
            raise ValueError("attention must have shape [B,N] or [B,H,N]")
        expected = int(grid_size[0]) * int(grid_size[1])
        if attention.shape[1] != expected:
            raise ValueError("attention token count does not match grid_size")
        mask = self.mask(grid_size, attention.device).to(attention.dtype)
        outside_mass = (attention * (1.0 - mask)).sum(dim=-1)
        loss = F.relu(outside_mass - self.config.tolerance).mean()
        return loss * self.epoch_weight(current_epoch), outside_mass.detach().mean()
