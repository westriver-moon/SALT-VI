from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class TokenDropoutConfig:
    """Normative final confidence-dropout settings."""

    target_fraction: float = 0.15
    gamma: float = 2.0
    p_max: float = 0.95
    min_image_reliability: float = 0.30
    min_successful_tta: int = 4
    selection: str = "exact_topk"

    def validate(self) -> "TokenDropoutConfig":
        if not 0.0 <= self.target_fraction < 1.0:
            raise ValueError("target_fraction must be in [0, 1)")
        if self.gamma <= 0.0:
            raise ValueError("gamma must be positive")
        if not 0.0 < self.p_max <= 1.0:
            raise ValueError("p_max must be in (0, 1]")
        if not 0.0 <= self.min_image_reliability <= 1.0:
            raise ValueError("min_image_reliability must be in [0, 1]")
        if self.min_successful_tta < 0:
            raise ValueError("min_successful_tta must be non-negative")
        if self.selection not in {"exact_topk", "bernoulli"}:
            raise ValueError("selection must be exact_topk or bernoulli")
        return self


@dataclass
class TokenDropoutResult:
    keep_mask: Tensor
    drop_mask: Tensor
    probabilities: Tensor
    effective_support: Tensor
    eligible_images: Tensor
    rho: float


@dataclass
class LayerwiseTokenDropoutResult:
    keep_masks: Tensor
    drop_masks: Tensor
    probabilities: Tensor
    effective_support: Tensor
    eligible_images: Tensor
    rho_by_layer: tuple[float, ...]


def _batch_scalar(value: Tensor | float, batch: int, device: torch.device) -> Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
    if tensor.ndim == 0:
        tensor = tensor.repeat(batch)
    if tensor.shape != (batch,):
        raise ValueError(f"expected scalar or [B], got {tuple(tensor.shape)}")
    return tensor


def effective_support(
    pose_support: Tensor,
    ellipse_support: Tensor,
    image_reliability: Tensor | float,
) -> Tensor:
    """P_eff = R_img * P_pose + (1-R_img) * P_ellipse."""

    if pose_support.shape != ellipse_support.shape or pose_support.ndim != 2:
        raise ValueError("pose_support and ellipse_support must share shape [B,N]")
    reliability = _batch_scalar(
        image_reliability, pose_support.shape[0], pose_support.device
    ).clamp(0.0, 1.0)
    pose = pose_support.float().clamp(0.0, 1.0)
    ellipse = ellipse_support.float().clamp(0.0, 1.0)
    return reliability[:, None] * pose + (1.0 - reliability[:, None]) * ellipse


def calibrate_rho(
    base_scores: Tensor,
    *,
    target_fraction: float,
    p_max: float,
    eligible_images: Tensor,
) -> float:
    """Calibrate rho so mean min(p_max, rho*score) matches the target."""

    selected = base_scores[eligible_images]
    if selected.numel() == 0 or target_fraction <= 0.0:
        return 0.0
    target = float(target_fraction)

    def mean_at(rho: float) -> float:
        return float(torch.clamp(selected * rho, max=p_max).mean().item())

    lower, upper = 0.0, 1.0
    while mean_at(upper) < target and upper < 1.0e8:
        upper *= 2.0
    if mean_at(upper) < target - 1.0e-7:
        raise ValueError("target dropout rate is unattainable for these supports")
    for _ in range(64):
        middle = (lower + upper) / 2.0
        if mean_at(middle) < target:
            lower = middle
        else:
            upper = middle
    return upper


def exact_topk_drop_mask(
    probabilities: Tensor,
    eligible_images: Tensor,
    target_fraction: float,
) -> Tensor:
    """Drop the highest-probability fixed-position patch tokens per image."""

    if probabilities.ndim != 2:
        raise ValueError("probabilities must have shape [B,N]")
    batch, tokens = probabilities.shape
    drop_count = int(tokens * float(target_fraction) + 0.5)
    result = torch.zeros_like(probabilities, dtype=torch.bool)
    if drop_count == 0:
        return result
    for index in range(batch):
        if bool(eligible_images[index]):
            chosen = torch.topk(
                probabilities[index], k=drop_count, largest=True, sorted=False
            ).indices
            result[index, chosen] = True
    return result


class ConfidenceTokenDropout(nn.Module):
    """Final confidence dropout; it returns masks and never removes positions."""

    def __init__(self, config: TokenDropoutConfig | None = None):
        super().__init__()
        self.config = (config or TokenDropoutConfig()).validate()

    def forward(
        self,
        pose_support: Tensor,
        ellipse_support: Tensor,
        image_reliability: Tensor | float,
        successful_tta: Tensor | int,
        *,
        generator: torch.Generator | None = None,
    ) -> TokenDropoutResult:
        cfg = self.config
        p_eff = effective_support(pose_support, ellipse_support, image_reliability)
        reliability = _batch_scalar(
            image_reliability, p_eff.shape[0], p_eff.device
        )
        tta = _batch_scalar(successful_tta, p_eff.shape[0], p_eff.device)
        eligible = (reliability >= cfg.min_image_reliability) & (
            tta >= cfg.min_successful_tta
        )
        # Final formula: no tokenwise reliability power is present.
        base_scores = (1.0 - p_eff).pow(cfg.gamma)
        rho = calibrate_rho(
            base_scores,
            target_fraction=cfg.target_fraction,
            p_max=cfg.p_max,
            eligible_images=eligible,
        )
        probabilities = torch.clamp(base_scores * rho, max=cfg.p_max)
        probabilities = probabilities * eligible[:, None]
        if cfg.selection == "exact_topk":
            drop_mask = exact_topk_drop_mask(
                probabilities, eligible, cfg.target_fraction
            )
        else:
            draws = torch.rand(
                probabilities.shape,
                device=probabilities.device,
                generator=generator,
            )
            drop_mask = draws < probabilities
        return TokenDropoutResult(
            keep_mask=~drop_mask,
            drop_mask=drop_mask,
            probabilities=probabilities,
            effective_support=p_eff,
            eligible_images=eligible,
            rho=rho,
        )

    def forward_layers(
        self,
        pose_support: Tensor,
        ellipse_support: Tensor,
        image_reliability: Tensor | float,
        successful_tta: Tensor | int,
        *,
        layer_count: int,
        target_fractions: list[float] | tuple[float, ...] | None = None,
        generator: torch.Generator | None = None,
    ) -> LayerwiseTokenDropoutResult:
        """Evaluate rho_l and P_drop_i,l for every requested transformer layer."""

        if layer_count < 1:
            raise ValueError("layer_count must be positive")
        fractions = (
            tuple(float(value) for value in target_fractions)
            if target_fractions is not None
            else (self.config.target_fraction,) * layer_count
        )
        if len(fractions) != layer_count or any(not 0.0 <= value < 1.0 for value in fractions):
            raise ValueError("target_fractions must provide one valid rate per layer")
        p_eff = effective_support(pose_support, ellipse_support, image_reliability)
        reliability = _batch_scalar(
            image_reliability, p_eff.shape[0], p_eff.device
        )
        tta = _batch_scalar(successful_tta, p_eff.shape[0], p_eff.device)
        eligible = (reliability >= self.config.min_image_reliability) & (
            tta >= self.config.min_successful_tta
        )
        base_scores = (1.0 - p_eff).pow(self.config.gamma)
        probabilities, masks, rhos = [], [], []
        for target in fractions:
            rho = calibrate_rho(
                base_scores,
                target_fraction=target,
                p_max=self.config.p_max,
                eligible_images=eligible,
            )
            probability = torch.clamp(base_scores * rho, max=self.config.p_max)
            probability = probability * eligible[:, None]
            if self.config.selection == "exact_topk":
                mask = exact_topk_drop_mask(probability, eligible, target)
            else:
                mask = torch.rand(
                    probability.shape,
                    device=probability.device,
                    generator=generator,
                ) < probability
            probabilities.append(probability)
            masks.append(mask)
            rhos.append(rho)
        drop_masks = torch.stack(masks)
        return LayerwiseTokenDropoutResult(
            keep_masks=~drop_masks,
            drop_masks=drop_masks,
            probabilities=torch.stack(probabilities),
            effective_support=p_eff,
            eligible_images=eligible,
            rho_by_layer=tuple(rhos),
        )
