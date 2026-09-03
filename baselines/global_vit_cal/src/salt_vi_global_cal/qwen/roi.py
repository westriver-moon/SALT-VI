from __future__ import annotations

from dataclasses import replace
from typing import Protocol, Sequence

import numpy as np
import torch
from PIL import Image, ImageFilter

from ..pose_posterior import PoseObservation
from .schema import CATEGORY_STATES, Region


LIP_PARTS = {
    "hat": 1, "hair": 2, "upper_clothes": 5, "dress": 6, "coat": 7,
    "socks": 8, "jumpsuit": 10, "left_arm": 14, "right_arm": 15,
    "left_shoe": 18, "right_shoe": 19,
}


class MaskRefiner(Protocol):
    def refine(
        self,
        image: Image.Image,
        bbox_xyxy: tuple[int, int, int, int],
        seed_mask: np.ndarray,
    ) -> np.ndarray: ...


PARTS_BY_REGION = {
    "head": ("hat", "hair"),
    "left_arm": ("left_arm",),
    "right_arm": ("right_arm",),
    "upper_torso": ("upper_clothes", "dress", "coat", "jumpsuit"),
    "left_foot": ("left_shoe", "socks"),
    "right_foot": ("right_shoe", "socks"),
}


def _clip(
    bbox: Sequence[float], image_size_wh: tuple[int, int]
) -> tuple[int, int, int, int]:
    width, height = image_size_wh
    left, top, right, bottom = bbox
    left = max(0, min(width - 1, int(round(float(left)))))
    top = max(0, min(height - 1, int(round(float(top)))))
    right = max(left + 1, min(width, int(round(float(right)))))
    bottom = max(top + 1, min(height, int(round(float(bottom)))))
    return left, top, right, bottom


def _point_box(
    point: tuple[float, float] | None,
    size: tuple[float, float],
    image_size_wh: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    if point is None:
        return None
    x, y = point
    width, height = size
    return _clip(
        (x - width / 2, y - height / 2, x + width / 2, y + height / 2),
        image_size_wh,
    )


def _group_box(
    points: torch.Tensor,
    indices: slice | Sequence[int],
    confidence: float,
    image_size_wh: tuple[int, int],
    *,
    min_size: tuple[float, float],
    padding: tuple[float, float] = (0.2, 0.2),
) -> tuple[int, int, int, int] | None:
    group = points[indices if isinstance(indices, slice) else list(indices)]
    valid = torch.isfinite(group).all(dim=1) & (group[:, 2] >= confidence)
    group = group[valid, :2]
    if not len(group):
        return None
    low, high = group.min(dim=0).values, group.max(dim=0).values
    center = (low + high) * 0.5
    span = torch.maximum(
        high - low, torch.tensor(min_size, device=group.device)
    )
    span *= 1.0 + 2.0 * torch.tensor(padding, device=group.device)
    return _point_box(
        (float(center[0]), float(center[1])),
        (float(span[0]), float(span[1])),
        image_size_wh,
    )


def pose_regions(
    observation: PoseObservation,
    image_size_wh: tuple[int, int] = (256, 512),
    *,
    keypoint_confidence: float = 0.30,
    allow_geometric_eye_fallback: bool = False,
    image: Image.Image | None = None,
    parsing: np.ndarray | None = None,
    mask_refiner: MaskRefiner | None = None,
) -> list[Region]:
    """Build the audited Qwen ROI taxonomy directly from WholeBody pose."""

    observation.validate()
    points = observation.keypoints.float()
    person = _clip(observation.bbox_xyxy.tolist(), image_size_wh)
    left, top, right, bottom = person
    person_w, person_h = right - left, bottom - top

    def point(index: int) -> tuple[float, float] | None:
        value = points[index]
        if not bool(torch.isfinite(value).all()) or float(value[2]) < keypoint_confidence:
            return None
        return float(value[0]), float(value[1])

    eye_box = _group_box(
        points, slice(59, 71), keypoint_confidence, image_size_wh,
        min_size=(max(12, person_w * 0.25), max(8, person_h * 0.035)),
        padding=(0.25, 0.65),
    )
    if eye_box is None:
        eyes = [value for value in (point(1), point(2)) if value is not None]
        if eyes:
            center = (
                sum(value[0] for value in eyes) / len(eyes),
                sum(value[1] for value in eyes) / len(eyes),
            )
            eye_box = _point_box(
                center, (max(12, person_w * 0.55), max(8, person_h * 0.055)),
                image_size_wh,
            )
        elif allow_geometric_eye_fallback:
            eye_box = _point_box(
                (left + person_w * 0.5, top + person_h * 0.09),
                (max(12, person_w * 0.55), max(8, person_h * 0.055)),
                image_size_wh,
            )

    head_box = _clip(
        (left + person_w * 0.25, top, right - person_w * 0.25, top + person_h * 0.22),
        image_size_wh,
    )
    dense_face = _group_box(
        points, slice(23, 91), keypoint_confidence, image_size_wh,
        min_size=(max(14, person_w * 0.25), max(18, person_h * 0.10)),
        padding=(0.25, 0.20),
    )
    if dense_face is not None:
        fl, ft, fr, fb = dense_face
        head_box = _clip(
            (fl - person_w * 0.08, ft - person_h * 0.08,
             fr + person_w * 0.08, fb + person_h * 0.03),
            image_size_wh,
        )

    wrist_size = (max(12, person_w * 0.24), max(12, person_h * 0.10))
    shoe_size = (max(16, person_w * 0.36), max(14, person_h * 0.12))
    pocket_size = (max(18, person_w * 0.35), max(20, person_h * 0.18))
    left_hand = _group_box(
        points, slice(91, 112), keypoint_confidence, image_size_wh,
        min_size=wrist_size, padding=(0.15, 0.15),
    ) or _point_box(point(9), wrist_size, image_size_wh)
    right_hand = _group_box(
        points, slice(112, 133), keypoint_confidence, image_size_wh,
        min_size=wrist_size, padding=(0.15, 0.15),
    ) or _point_box(point(10), wrist_size, image_size_wh)
    left_foot = _group_box(
        points, slice(17, 20), keypoint_confidence, image_size_wh,
        min_size=shoe_size,
    ) or _point_box(point(15), shoe_size, image_size_wh)
    right_foot = _group_box(
        points, slice(20, 23), keypoint_confidence, image_size_wh,
        min_size=shoe_size,
    ) or _point_box(point(16), shoe_size, image_size_wh)

    arm_width = max(12, person_w * 0.28)
    arm_height = max(24, person_h * 0.35)
    torso_box = _group_box(
        points, (5, 6, 11, 12), keypoint_confidence, image_size_wh,
        min_size=(max(20, person_w * 0.65), max(24, person_h * 0.32)),
        padding=(0.08, 0.08),
    )
    left_arm = _group_box(
        points, (5, 7, 9), keypoint_confidence, image_size_wh,
        min_size=(arm_width, arm_height), padding=(0.12, 0.08),
    )
    right_arm = _group_box(
        points, (6, 8, 10), keypoint_confidence, image_size_wh,
        min_size=(arm_width, arm_height), padding=(0.12, 0.08),
    )
    specs = (
        ("eyes", "eyewear", eye_box),
        ("head", "headwear", head_box),
        ("left_wrist", "wrist_accessory", left_hand),
        ("right_wrist", "wrist_accessory", right_hand),
        ("left_arm", "body_marking", left_arm),
        ("right_arm", "body_marking", right_arm),
        ("upper_torso", "clothing_detail", torso_box),
        ("left_pocket", "pocket_item", _point_box(point(11), pocket_size, image_size_wh)),
        ("right_pocket", "pocket_item", _point_box(point(12), pocket_size, image_size_wh)),
        (
            "left_carried", "carried_object",
            _clip(
                (left - person_w * 0.35, top + person_h * 0.18,
                 left + person_w * 0.30, top + person_h * 0.78),
                image_size_wh,
            ),
        ),
        (
            "right_carried", "carried_object",
            _clip(
                (right - person_w * 0.30, top + person_h * 0.18,
                 right + person_w * 0.35, top + person_h * 0.78),
                image_size_wh,
            ),
        ),
        ("left_foot", "footwear_detail", left_foot),
        ("right_foot", "footwear_detail", right_foot),
    )
    width, height = image_size_wh
    parsing_array = None
    if parsing is not None:
        parsing_array = np.asarray(parsing, dtype=np.uint8)
        if parsing_array.shape != (height, width):
            parsing_array = np.asarray(
                Image.fromarray(parsing_array).resize(
                    image_size_wh, getattr(Image, "Resampling", Image).NEAREST
                )
            )
    if mask_refiner is not None and image is None:
        raise ValueError("SAM-style mask refinement requires the SwinIR image")
    regions = []
    for region_id, category, bbox in specs:
        if bbox is None:
            continue
        base = np.zeros((height, width), dtype=bool)
        box_left, box_top, box_right, box_bottom = bbox
        base[box_top:box_bottom, box_left:box_right] = True
        part_names = PARTS_BY_REGION.get(region_id, ())
        if parsing_array is not None and part_names:
            part = np.isin(parsing_array, [LIP_PARTS[name] for name in part_names])
            if part.any():
                base &= part
        if not base.any():
            base[box_top:box_bottom, box_left:box_right] = True
        if mask_refiner is not None and category != "eyewear":
            refined = np.asarray(
                mask_refiner.refine(image.convert("RGB"), bbox, base), dtype=bool
            )
            if refined.shape != base.shape:
                raise ValueError("SAM-style refined mask is not image-aligned")
            box_mask = np.zeros_like(base)
            box_mask[box_top:box_bottom, box_left:box_right] = True
            refined &= box_mask
            if refined.any():
                base = refined
        regions.append(
            Region(
                region_id,
                category,
                bbox,
                CATEGORY_STATES[category],
                mask=base,
            )
        )
    return regions


def _gray(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("L"), dtype=np.float32) / 255.0


def _high_pass(image: Image.Image) -> np.ndarray:
    gray = _gray(image)
    blurred = np.asarray(
        image.convert("L").filter(ImageFilter.GaussianBlur(1.0)),
        dtype=np.float32,
    ) / 255.0
    return gray - blurred


def _robust_normalize(value: float, median: float, iqr: float) -> float:
    z = (float(value) - float(median)) / max(float(iqr), 1.0e-8)
    return float(np.clip(0.5 + 0.25 * z, 0.0, 1.0))


def score_region_uncertainty(
    lr: Image.Image,
    swin: Image.Image,
    regions: Sequence[Region],
    *,
    category_stats: dict[str, dict[str, float]] | None = None,
) -> list[Region]:
    """Reproduce the final fast residual/blur ROI pre-scan used by track-anchor."""

    if not regions:
        raise ValueError("ROI uncertainty scoring requires pose-derived regions")
    bicubic = lr.convert("RGB").resize(
        swin.size, getattr(Image, "Resampling", Image).BICUBIC
    )
    residual = np.abs(_high_pass(swin) - _high_pass(bicubic))
    raw_residual, raw_blur = [], []
    for region in regions:
        selected = np.asarray(region.mask, dtype=bool)
        if selected.shape != residual.shape or not selected.any():
            selected = np.zeros_like(residual, dtype=bool)
            left, top, right, bottom = region.bbox_xyxy
            selected[top:bottom, left:right] = True
        values = residual[selected]
        center = float(np.median(values))
        mad = float(np.median(np.abs(values - center)))
        raw_residual.append(center + 1.4826 * mad)
        high = np.abs(_high_pass(swin))[selected]
        contrast = np.std(_gray(swin)[selected])
        information = float(np.mean(high) + contrast)
        raw_blur.append(float(1.0 / (1.0 + 20.0 * information)))
    residual_median = float(np.median(raw_residual))
    residual_iqr = float(np.subtract(*np.percentile(raw_residual, [75, 25])))
    blur_median = float(np.median(raw_blur))
    blur_iqr = float(np.subtract(*np.percentile(raw_blur, [75, 25])))
    stats_by_category = category_stats or {}
    scored = []
    for region, residual_value, blur_value in zip(regions, raw_residual, raw_blur):
        stats = stats_by_category.get(region.category, {})
        residual_score = _robust_normalize(
            residual_value,
            float(stats.get("fast_residual_median", residual_median)),
            float(stats.get("fast_residual_iqr", residual_iqr)),
        )
        blur_score = _robust_normalize(
            blur_value,
            float(stats.get("blur_median", blur_median)),
            float(stats.get("blur_iqr", blur_iqr)),
        )
        scored.append(
            replace(
                region,
                uncertainty=0.65 * residual_score + 0.35 * blur_score,
                blur_uncertainty=blur_value,
            )
        )
    return scored


def select_uncertain_regions(
    regions: Sequence[Region],
    *,
    selected_count: int = 3,
    max_selected_count: int = 8,
    category_priority_boosts: dict[str, float] | None = None,
) -> list[Region]:
    if not 1 <= selected_count <= max_selected_count:
        raise ValueError("selected_count must be within the configured maximum")
    if len(regions) < selected_count:
        raise ValueError("too few pose-derived regions for Qwen selection")
    boosts = category_priority_boosts or {
        "carried_object": 0.25,
        "wrist_accessory": 0.10,
        "headwear": 0.05,
    }
    ranked = sorted(
        regions,
        key=lambda item: (
            -(float(item.uncertainty) + float(boosts.get(item.category, 0.0))),
            item.region_id,
        ),
    )
    return ranked[:selected_count]
