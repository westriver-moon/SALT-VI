from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor, nn


BODY_EDGES = (
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
)


@dataclass(frozen=True)
class TTASpec:
    name: str
    scale: float = 1.0
    tx: float = 0.0
    ty: float = 0.0
    flip: bool = False


DEFAULT_TTA = (
    TTASpec("identity"),
    TTASpec("hflip", flip=True),
    TTASpec("scale_in", scale=0.92),
    TTASpec("scale_out", scale=1.08),
    TTASpec("shift_left", tx=-0.04),
    TTASpec("shift_right", tx=0.04),
)


@dataclass(frozen=True)
class PoseObservation:
    keypoints: Tensor
    bbox_xyxy: Tensor
    structural_reliability: float = 1.0

    def validate(self) -> "PoseObservation":
        if self.keypoints.shape != (133, 3):
            raise ValueError("whole-body pose must have shape [133,3]")
        if self.bbox_xyxy.shape != (4,):
            raise ValueError("person bbox must have shape [4]")
        return self


class WholeBodyPoseBackend(Protocol):
    def infer(self, image: Image.Image) -> PoseObservation | None: ...


@dataclass(frozen=True)
class PoseSupportConfig:
    keypoint_confidence: float = 0.30
    joint_sigma_over_height: dict[str, float] = field(
        default_factory=lambda: {
            "body": 0.025, "foot": 0.015, "face": 0.008, "hand": 0.010,
        }
    )
    limb_width_over_height: dict[str, float] = field(
        default_factory=lambda: {
            "torso": 0.075, "upper_limb": 0.040,
            "lower_limb": 0.035, "leg": 0.050,
        }
    )
    tau_aug: float = 0.020
    ellipse_rx_over_width: float = 0.42
    ellipse_ry_over_height: float = 0.48
    token_pool_mean_weight: float = 0.5
    token_pool_top_fraction: float = 0.20


@dataclass
class PosePosteriorResult:
    pose_support: Tensor
    ellipse_support: Tensor
    dense_pose_posterior: Tensor
    dense_ellipse_prior: Tensor
    local_reliability: Tensor
    image_reliability: float
    structural_reliability: float
    association_reliability: float
    successful_tta: int
    token_grid: tuple[int, int]


def affine_tta_matrix(width: int, height: int, spec: TTASpec) -> Tensor:
    scale = float(spec.scale)
    tx, ty = float(spec.tx) * width, float(spec.ty) * height
    cx, cy = (width - 1) * 0.5, (height - 1) * 0.5
    transform = torch.tensor(
        [[scale, 0.0, cx + tx - scale * cx],
         [0.0, scale, cy + ty - scale * cy],
         [0.0, 0.0, 1.0]], dtype=torch.float32,
    )
    if spec.flip:
        flip = torch.tensor(
            [[-1.0, 0.0, width - 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        )
        transform = flip @ transform
    return transform


def transform_points(points: Tensor, matrix: Tensor) -> Tensor:
    output = points.clone().float()
    homogeneous = torch.cat(
        (output[:, :2], torch.ones(len(output), 1, device=output.device)), dim=1
    )
    mapped = homogeneous @ matrix.to(output.device).T
    output[:, :2] = mapped[:, :2] / mapped[:, 2:3].clamp_min(1.0e-8)
    return output


def transform_bbox(bbox: Tensor, matrix: Tensor) -> Tensor:
    left, top, right, bottom = bbox.float()
    corners = torch.stack((
        torch.stack((left, top)), torch.stack((right, top)),
        torch.stack((right, bottom)), torch.stack((left, bottom)),
    ))
    mapped = transform_points(corners, matrix)
    return torch.stack((
        mapped[:, 0].min(), mapped[:, 1].min(),
        mapped[:, 0].max(), mapped[:, 1].max(),
    ))


def warp_image(image: Image.Image, forward: Tensor) -> Image.Image:
    inverse = torch.linalg.inv(forward).cpu().numpy()
    coefficients = tuple(float(value) for value in inverse[:2].reshape(-1))
    transform_mode = getattr(Image, "Transform", Image).AFFINE
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    return image.convert("RGB").transform(
        image.size, transform_mode, coefficients, resample=resampling
    )


def collect_tta_observations(
    image: Image.Image,
    backend: WholeBodyPoseBackend,
    *,
    specs: Sequence[TTASpec] = DEFAULT_TTA,
    structural_scorer: "StructuralScorer | None" = None,
) -> list[PoseObservation]:
    observations = []
    for spec in specs:
        forward = affine_tta_matrix(image.width, image.height, spec)
        result = backend.infer(warp_image(image, forward))
        if result is None:
            continue
        result.validate()
        inverse = torch.linalg.inv(forward)
        points = transform_points(result.keypoints, inverse)
        bbox = transform_bbox(result.bbox_xyxy, inverse)
        structural = (
            structural_scorer.reliability(points, bbox)
            if structural_scorer is not None
            else float(result.structural_reliability)
        )
        observations.append(PoseObservation(points, bbox, structural))
    return observations


def _joint_group(index: int) -> str:
    if index < 17:
        return "body"
    if index < 23:
        return "foot"
    if index < 91:
        return "face"
    return "hand"


def _grid(shape: tuple[int, int], device: torch.device) -> tuple[Tensor, Tensor]:
    y = torch.arange(shape[0], device=device, dtype=torch.float32)
    x = torch.arange(shape[1], device=device, dtype=torch.float32)
    return torch.meshgrid(y, x, indexing="ij")


def _gaussian(
    shape: tuple[int, int], point: Tensor, sigma: float, weight: float
) -> Tensor:
    yy, xx = _grid(shape, point.device)
    distance2 = (xx - point[0]).square() + (yy - point[1]).square()
    return float(weight) * torch.exp(-distance2 / (2.0 * float(sigma) ** 2))


def _capsule(
    shape: tuple[int, int], start: Tensor, end: Tensor, radius: float, weight: float
) -> Tensor:
    yy, xx = _grid(shape, start.device)
    vector = end.float() - start.float()
    length2 = vector.square().sum()
    if float(length2) < 1.0e-6:
        return _gaussian(shape, start, max(radius, 1.0), weight)
    t = ((xx - start[0]) * vector[0] + (yy - start[1]) * vector[1]) / length2
    t = t.clamp(0.0, 1.0)
    dx = xx - (start[0] + t * vector[0])
    dy = yy - (start[1] + t * vector[1])
    distance = torch.sqrt(dx.square() + dy.square())
    return float(weight) * torch.exp(-0.5 * (distance / max(radius, 1.0)).square())


def _convex_polygon(shape: tuple[int, int], polygon: Tensor, weight: float) -> Tensor:
    yy, xx = _grid(shape, polygon.device)
    signs = []
    for index in range(len(polygon)):
        start, end = polygon[index], polygon[(index + 1) % len(polygon)]
        signs.append(
            (end[0] - start[0]) * (yy - start[1])
            - (end[1] - start[1]) * (xx - start[0])
        )
    stacked = torch.stack(signs)
    inside = (stacked >= 0).all(dim=0) | (stacked <= 0).all(dim=0)
    return inside.float() * float(weight)


def _gaussian_blur(field: Tensor, sigma: float) -> Tensor:
    sigma = max(float(sigma), 1.0)
    radius = max(1, int(round(3.0 * sigma)))
    coordinates = torch.arange(-radius, radius + 1, device=field.device).float()
    kernel = torch.exp(-0.5 * (coordinates / sigma).square())
    kernel = kernel / kernel.sum()
    value = F.conv2d(
        field[None, None], kernel[None, None, :, None], padding=(radius, 0)
    )
    value = F.conv2d(value, kernel[None, None, None, :], padding=(0, radius))
    return value[0, 0]


def render_pose_support(
    observation: PoseObservation,
    *,
    output_hw: tuple[int, int],
    source_hw: tuple[int, int],
    config: PoseSupportConfig | None = None,
) -> Tensor:
    """Render joint Gaussians, limb capsules, torso and head support."""

    cfg = config or PoseSupportConfig()
    observation.validate()
    out_h, out_w = output_hw
    src_h, src_w = source_hw
    points = observation.keypoints.float().clone()
    points[:, 0] *= out_w / src_w
    points[:, 1] *= out_h / src_h
    bbox = observation.bbox_xyxy.float() * torch.tensor(
        [out_w / src_w, out_h / src_h, out_w / src_w, out_h / src_h],
        device=observation.bbox_xyxy.device,
    )
    channels = []
    for index, point in enumerate(points):
        confidence = float(point[2])
        if not bool(torch.isfinite(point).all()) or confidence < cfg.keypoint_confidence:
            continue
        sigma = cfg.joint_sigma_over_height[_joint_group(index)] * out_h
        channels.append(_gaussian(output_hw, point[:2], sigma, confidence))

    body = points[:17]
    widths = cfg.limb_width_over_height
    for start, end in BODY_EDGES:
        confidence = float(torch.minimum(body[start, 2], body[end, 2]))
        if confidence < cfg.keypoint_confidence:
            continue
        if (start, end) in {(5, 11), (6, 12), (5, 6), (11, 12)}:
            width = widths["torso"]
        elif max(start, end) <= 10:
            width = (
                widths["upper_limb"]
                if min(start, end) in {5, 6, 7, 8}
                else widths["lower_limb"]
            )
        else:
            width = widths["leg"]
        channels.append(
            _capsule(
                output_hw, body[start, :2], body[end, :2],
                width * out_h, confidence,
            )
        )

    torso_indices = torch.tensor([5, 6, 12, 11], device=body.device)
    if bool((body[torso_indices, 2] >= cfg.keypoint_confidence).all()):
        torso = _convex_polygon(
            output_hw,
            body[torso_indices, :2],
            float(body[torso_indices, 2].mean()),
        )
        channels.append(_gaussian_blur(torso, widths["torso"] * out_h * 0.35))

    left, top, right, bottom = bbox
    cx, cy = (left + right) * 0.5, top + 0.11 * (bottom - top)
    rx = max(2.0, float(0.23 * (right - left)))
    ry = max(2.0, float(0.12 * (bottom - top)))
    yy, xx = _grid(output_hw, points.device)
    head = (
        (((xx - cx) / rx).square() + ((yy - cy) / ry).square()) <= 1.0
    ).float() * 0.65
    channels.append(_gaussian_blur(head, 0.01 * out_h))
    if not channels:
        return torch.zeros(output_hw, dtype=torch.float32)
    stack = torch.stack(channels).clamp(0.0, 1.0)
    return 1.0 - torch.prod(1.0 - stack, dim=0)


def ellipse_prior(
    bbox_xyxy: Tensor,
    *,
    output_hw: tuple[int, int],
    source_hw: tuple[int, int],
    config: PoseSupportConfig | None = None,
) -> Tensor:
    cfg = config or PoseSupportConfig()
    out_h, out_w = output_hw
    src_h, src_w = source_hw
    bbox = bbox_xyxy.float() * torch.tensor(
        [out_w / src_w, out_h / src_h, out_w / src_w, out_h / src_h],
        device=bbox_xyxy.device,
    )
    left, top, right, bottom = bbox
    cx, cy = (left + right) * 0.5, (top + bottom) * 0.5
    rx = max(1.0, float(cfg.ellipse_rx_over_width * (right - left)))
    ry = max(1.0, float(cfg.ellipse_ry_over_height * (bottom - top)))
    yy, xx = _grid(output_hw, bbox.device)
    distance = ((xx - cx) / rx).square() + ((yy - cy) / ry).square()
    return torch.exp(-0.5 * distance)


def token_pool(
    field: Tensor,
    *,
    patch_hw: tuple[int, int],
    stride_hw: tuple[int, int],
    mean_weight: float = 0.5,
    top_fraction: float = 0.20,
) -> Tensor:
    patch_h, patch_w = patch_hw
    patches = F.unfold(
        field.float()[None, None], kernel_size=patch_hw, stride=stride_hw
    )[0]
    top_count = max(1, int(round(patch_h * patch_w * float(top_fraction))))
    top_mean = patches.topk(top_count, dim=0).values.mean(dim=0)
    pooled = (
        float(mean_weight) * patches.mean(dim=0)
        + (1.0 - float(mean_weight)) * top_mean
    )
    rows = (field.shape[0] - patch_h) // stride_hw[0] + 1
    cols = (field.shape[1] - patch_w) // stride_hw[1] + 1
    return pooled.reshape(rows, cols)


def bbox_association_reliability(
    boxes: Sequence[Tensor], source_hw: tuple[int, int]
) -> float:
    if len(boxes) < 2:
        return 0.0
    height, width = source_hw
    features = []
    for left, top, right, bottom in boxes:
        features.append(torch.stack((
            (left + right) / (2 * width),
            (top + bottom) / (2 * height),
            (right - left) / width,
            (bottom - top) / height,
        )))
    dispersion = torch.stack(features).var(dim=0, unbiased=False).mean()
    return float(torch.exp(-dispersion / 0.0025))


def build_pose_posterior(
    observations: Sequence[PoseObservation],
    *,
    source_hw: tuple[int, int] = (512, 256),
    output_hw: tuple[int, int] = (288, 144),
    patch_hw: tuple[int, int] = (16, 16),
    stride_hw: tuple[int, int] = (10, 10),
    total_tta: int = 6,
    config: PoseSupportConfig | None = None,
) -> PosePosteriorResult:
    """Fuse successful TTA supports and map pose/fallback evidence to ViT tokens."""

    if not observations:
        raise ValueError("pose posterior requires at least one successful TTA")
    cfg = config or PoseSupportConfig()
    supports = torch.stack([
        render_pose_support(
            observation, output_hw=output_hw, source_hw=source_hw, config=cfg
        )
        for observation in observations
    ])
    posterior = supports.mean(dim=0)
    local_reliability = torch.exp(
        -supports.var(dim=0, unbiased=False) / cfg.tau_aug
    )
    structural = float(np.mean([
        item.structural_reliability for item in observations
    ]))
    association = bbox_association_reliability(
        [item.bbox_xyxy for item in observations], source_hw
    )
    availability = len(observations) / max(1, int(total_tta))
    image_reliability = float(
        max(0.0, structural * association * availability) ** (1.0 / 3.0)
    )
    fallback = ellipse_prior(
        observations[0].bbox_xyxy,
        output_hw=output_hw,
        source_hw=source_hw,
        config=cfg,
    )
    pool = {
        "patch_hw": patch_hw,
        "stride_hw": stride_hw,
        "mean_weight": cfg.token_pool_mean_weight,
        "top_fraction": cfg.token_pool_top_fraction,
    }
    pose_tokens = token_pool(posterior, **pool)
    ellipse_tokens = token_pool(fallback, **pool)
    return PosePosteriorResult(
        pose_support=pose_tokens.flatten(),
        ellipse_support=ellipse_tokens.flatten(),
        dense_pose_posterior=posterior,
        dense_ellipse_prior=fallback,
        local_reliability=local_reliability,
        image_reliability=image_reliability,
        structural_reliability=structural,
        association_reliability=association,
        successful_tta=len(observations),
        token_grid=tuple(pose_tokens.shape),
    )


def pose_feature(points: Tensor, bbox: Tensor) -> Tensor:
    body = points.float()[:17]
    left, top, right, bottom = bbox.float()
    width = (right - left).clamp_min(1.0)
    height = (bottom - top).clamp_min(1.0)
    xy = body[:, :2].clone()
    xy[:, 0] = (xy[:, 0] - left) / width
    xy[:, 1] = (xy[:, 1] - top) / height
    score = body[:, 2:3].clamp(0.0, 1.0)
    return torch.nan_to_num(
        torch.cat((xy, score), dim=1).flatten(),
        nan=0.0, posinf=2.0, neginf=-1.0,
    )


class PoseAutoencoder(nn.Module):
    def __init__(self, latent_dim: int = 8):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(51, 32), nn.ReLU(), nn.Linear(32, latent_dim), nn.ReLU(),
            nn.Linear(latent_dim, 32), nn.ReLU(), nn.Linear(32, 51),
        )

    def forward(self, value: Tensor) -> Tensor:
        return self.network(value)


class StructuralScorer:
    def __init__(self, artifact: str):
        payload = torch.load(artifact, map_location="cpu", weights_only=False)
        self.model = PoseAutoencoder(int(payload["latent_dim"]))
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()
        self.mean = torch.as_tensor(payload["mean"], dtype=torch.float32)
        self.std = torch.as_tensor(payload["std"], dtype=torch.float32)
        self.tau = float(payload["tau"])

    def reliability(self, points: Tensor, bbox: Tensor) -> float:
        value = ((pose_feature(points, bbox) - self.mean) / self.std)[None]
        with torch.no_grad():
            error = torch.mean((self.model(value) - value).square())
        return float(torch.exp(-error / self.tau))
