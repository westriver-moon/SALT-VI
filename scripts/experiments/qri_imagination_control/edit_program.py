#!/usr/bin/env python3
"""Generic VLM edit-program validation and spatial control-map rasterization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


SUPPORTED_PRIMITIVES = {"ellipse", "rectangle", "polygon", "polyline"}


def load_edit_program(path: Path) -> dict[str, Any]:
    program = json.loads(path.read_text(encoding="utf-8"))
    if int(program.get("schema_version", 0)) != 1:
        raise ValueError("edit program requires schema_version=1")
    layout = program.get("layout")
    if not isinstance(layout, dict):
        raise ValueError("edit program requires a layout object")
    _validate_layout(layout, "layout")
    for optional_name in ("edit_region", "preservation_layout"):
        optional_layout = program.get(optional_name)
        if optional_layout is not None:
            if not isinstance(optional_layout, dict):
                raise ValueError(f"{optional_name} must be an object")
            _validate_layout(optional_layout, optional_name)
    return program


def _validate_layout(layout: dict[str, Any], label: str) -> None:
    primitives = layout.get("primitives")
    if not isinstance(primitives, list) or not primitives:
        raise ValueError(f"{label}.primitives must be a non-empty list")
    coordinate_space = str(layout.get("coordinate_space", "pixel"))
    if coordinate_space not in {"pixel", "normalized"}:
        raise ValueError(f"{label}.coordinate_space must be pixel or normalized")
    if coordinate_space == "pixel":
        canvas = layout.get("canvas_size")
        if (
            not isinstance(canvas, list)
            or len(canvas) != 2
            or min(int(canvas[0]), int(canvas[1])) <= 0
        ):
            raise ValueError(f"pixel {label} requires positive canvas_size [width, height]")
    for index, primitive in enumerate(primitives):
        if not isinstance(primitive, dict):
            raise ValueError(f"layout primitive {index} must be an object")
        kind = str(primitive.get("type", ""))
        if kind not in SUPPORTED_PRIMITIVES:
            raise ValueError(f"unsupported layout primitive: {kind}")
        if kind in {"ellipse", "rectangle"} and not _is_bbox(primitive.get("bbox")):
            raise ValueError(f"{kind} primitive requires bbox [x0,y0,x1,y1]")
        if kind in {"polygon", "polyline"} and not _is_points(primitive.get("points")):
            raise ValueError(f"{kind} primitive requires at least two points")
        if float(primitive.get("stroke_width", 1.0)) <= 0:
            raise ValueError("stroke_width must be positive")


def _is_bbox(value: object) -> bool:
    return isinstance(value, list) and len(value) == 4


def _is_points(value: object) -> bool:
    return (
        isinstance(value, list)
        and len(value) >= 2
        and all(isinstance(point, list) and len(point) == 2 for point in value)
    )


def _coordinate_scale(layout: dict[str, Any], size: tuple[int, int]) -> tuple[float, float]:
    if str(layout.get("coordinate_space", "pixel")) == "normalized":
        return float(size[0]), float(size[1])
    canvas = layout["canvas_size"]
    return float(size[0]) / float(canvas[0]), float(size[1]) / float(canvas[1])


def _point(value: list[float], scale: tuple[float, float]) -> tuple[int, int]:
    return (int(round(float(value[0]) * scale[0])), int(round(float(value[1]) * scale[1])))


def _bbox(value: list[float], scale: tuple[float, float]) -> tuple[int, int, int, int]:
    x0, y0 = _point(value[:2], scale)
    x1, y1 = _point(value[2:], scale)
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def rasterize_layout(layout: dict[str, Any], size: tuple[int, int]) -> Image.Image:
    """Rasterize category-agnostic layout primitives into a binary map."""
    scale = _coordinate_scale(layout, size)
    width_scale = max(1e-6, (scale[0] + scale[1]) * 0.5)
    image = Image.new("L", size, 0)
    draw = ImageDraw.Draw(image)
    for primitive in layout["primitives"]:
        kind = str(primitive["type"])
        stroke_width = max(1, int(round(float(primitive.get("stroke_width", 1.0)) * width_scale)))
        fill_shape = bool(primitive.get("fill", False))
        if kind == "ellipse":
            box = _bbox(primitive["bbox"], scale)
            draw.ellipse(box, fill=255 if fill_shape else None, outline=255, width=stroke_width)
        elif kind == "rectangle":
            box = _bbox(primitive["bbox"], scale)
            draw.rectangle(box, fill=255 if fill_shape else None, outline=255, width=stroke_width)
        else:
            points = [_point(point, scale) for point in primitive["points"]]
            if kind == "polygon" and fill_shape:
                draw.polygon(points, fill=255)
            elif kind == "polygon":
                draw.line(points + [points[0]], fill=255, width=stroke_width, joint="curve")
            else:
                draw.line(points, fill=255, width=stroke_width, joint="curve")
    return image.point(lambda value: 255 if value > 0 else 0)


def rasterize_creation_map(program: dict[str, Any], size: tuple[int, int]) -> Image.Image:
    """Rasterize the semantic target layout into a binary anchor map."""
    return rasterize_layout(program["layout"], size)


def rasterize_preservation_hint(
    program: dict[str, Any], size: tuple[int, int]
) -> Image.Image:
    """Rasterize optional structures that write-back must preserve exactly."""
    layout = program.get("preservation_layout")
    if layout is None:
        return Image.new("L", size, 0)
    return rasterize_layout(layout, size)


def rasterize_edit_region(program: dict[str, Any], size: tuple[int, int]) -> Image.Image:
    """Rasterize the semantic canvas where coherent target content may be written."""
    layout = program.get("edit_region")
    if layout is None:
        return rasterize_creation_map(program, size)
    return rasterize_layout(layout, size)


def feathered_region_map(region: Image.Image, feather_px: float) -> Image.Image:
    """Create a coherent soft write-back alpha from a VLM-proposed semantic region."""
    alpha = region.convert("L")
    if float(feather_px) > 0.0:
        alpha = alpha.filter(ImageFilter.GaussianBlur(float(feather_px)))
    return alpha


def harmonize_proposal(
    reference: Image.Image,
    proposal: Image.Image,
    statistics_mask: Image.Image,
    *,
    strength: float = 0.7,
    scale_min: float = 0.8,
    scale_max: float = 1.2,
) -> Image.Image:
    """Match local color statistics while retaining generated semantic detail."""
    if reference.size != proposal.size or reference.size != statistics_mask.size:
        raise ValueError("harmonization inputs must share a size")
    amount = float(strength)
    if not 0.0 <= amount <= 1.0:
        raise ValueError("harmonization strength must be in [0, 1]")
    source = np.asarray(reference.convert("RGB"), dtype=np.float32)
    target = np.asarray(proposal.convert("RGB"), dtype=np.float32)
    selected = np.asarray(statistics_mask.convert("L"), dtype=np.float32) > 127.0
    if int(selected.sum()) < 2:
        return proposal.convert("RGB")
    source_pixels = source[selected]
    target_pixels = target[selected]
    source_mean = source_pixels.mean(axis=0)
    target_mean = target_pixels.mean(axis=0)
    source_std = source_pixels.std(axis=0)
    target_std = np.maximum(target_pixels.std(axis=0), 1e-6)
    scale = np.clip(source_std / target_std, float(scale_min), float(scale_max))
    matched = (target - target_mean[None, None, :]) * scale[None, None, :] + source_mean[
        None, None, :
    ]
    output = target * (1.0 - amount) + matched * amount
    return Image.fromarray(np.round(np.clip(output, 0.0, 255.0)).astype(np.uint8), mode="RGB")


def _dilate(mask: Image.Image, radius: int) -> Image.Image:
    radius = max(0, int(radius))
    if radius == 0:
        return mask.convert("L")
    return mask.convert("L").filter(ImageFilter.MaxFilter(2 * radius + 1))


def _gradient_energy(array: np.ndarray) -> np.ndarray:
    gray = array.astype(np.float32).mean(axis=2)
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:] = np.abs(gray[:, 1:] - gray[:, :-1])
    gy[1:, :] = np.abs(gray[1:, :] - gray[:-1, :])
    return gx + gy


def adaptive_writeback_map(
    reference: Image.Image,
    proposal: Image.Image,
    layout_anchor: Image.Image,
    preservation_hint: Image.Image | None = None,
    *,
    support_dilation_px: int = 10,
    anchor_dilation_px: int = 3,
    delta_quantile: float = 55.0,
    detail_quantile: float = 55.0,
    high_delta_quantile: float = 88.0,
    source_detail_ratio: float = 0.45,
    seed_dilation_px: int = 1,
    feather_px: float = 0.7,
) -> dict[str, Image.Image]:
    """Extract generated semantic evidence instead of using layout as a stencil.

    The layout is only a broad spatial anchor. Candidate pixels are selected from
    source-relative change and newly created detail, while optional preservation
    hints are excluded exactly. This keeps the mechanism category-agnostic and
    avoids forcing a generated object back through a hand-drawn outline.
    """
    if reference.size != proposal.size or reference.size != layout_anchor.size:
        raise ValueError("reference, proposal and layout_anchor must share a size")
    if preservation_hint is not None and preservation_hint.size != reference.size:
        raise ValueError("preservation_hint must share the reference size")
    for name, value in {
        "delta_quantile": delta_quantile,
        "detail_quantile": detail_quantile,
        "high_delta_quantile": high_delta_quantile,
    }.items():
        if not 0.0 <= float(value) <= 100.0:
            raise ValueError(f"{name} must be in [0, 100]")

    reference_array = np.asarray(reference.convert("RGB"), dtype=np.float32)
    proposal_array = np.asarray(proposal.convert("RGB"), dtype=np.float32)
    delta = np.abs(proposal_array - reference_array).mean(axis=2)
    proposal_detail = _gradient_energy(proposal_array)
    reference_detail = _gradient_energy(reference_array)
    new_detail = np.maximum(
        proposal_detail - float(source_detail_ratio) * reference_detail, 0.0
    )

    support_image = _dilate(layout_anchor, support_dilation_px)
    anchor_image = _dilate(layout_anchor, anchor_dilation_px)
    support = np.asarray(support_image, dtype=np.float32) > 0.0
    anchor = np.asarray(anchor_image, dtype=np.float32) > 0.0
    protected = np.zeros_like(support)
    if preservation_hint is not None:
        protected = np.asarray(preservation_hint.convert("L"), dtype=np.float32) > 0.0
    valid = support & ~protected
    values = delta[valid]
    details = new_detail[valid]
    if values.size == 0:
        raise ValueError("adaptive write-back support is empty")
    delta_threshold = float(np.percentile(values, float(delta_quantile)))
    detail_threshold = float(np.percentile(details, float(detail_quantile)))
    high_delta_threshold = float(np.percentile(values, float(high_delta_quantile)))

    detail_seed = (delta >= delta_threshold) & (new_detail >= detail_threshold)
    anchor_seed = (delta >= high_delta_threshold) & anchor
    seed = valid & (detail_seed | anchor_seed)
    seed_image = Image.fromarray(np.where(seed, 255, 0).astype(np.uint8), mode="L")
    alpha = _dilate(seed_image, seed_dilation_px)
    if float(feather_px) > 0.0:
        alpha = alpha.filter(ImageFilter.GaussianBlur(float(feather_px)))
    alpha_array = np.asarray(alpha, dtype=np.float32) / 255.0
    alpha_array *= valid.astype(np.float32)
    alpha = Image.fromarray(
        np.round(np.clip(alpha_array, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L"
    )

    delta_scale = max(high_delta_threshold, delta_threshold, 1e-6)
    detail_scale = max(detail_threshold, 1e-6)
    evidence = np.clip(delta / delta_scale, 0.0, 1.0) * np.clip(
        new_detail / detail_scale, 0.0, 1.0
    )
    evidence *= valid.astype(np.float32)
    return {
        "alpha": alpha,
        "support": support_image,
        "seed": seed_image,
        "evidence": Image.fromarray(
            np.round(evidence * 255.0).astype(np.uint8), mode="L"
        ),
    }


def build_trimap(
    creation_map: Image.Image,
    transition_dilation_px: int,
    transition_feather_px: float,
) -> dict[str, Image.Image]:
    """Build creation, transition, preservation and binary inpaint maps."""
    creation = creation_map.convert("L").point(lambda value: 255 if value > 0 else 0)
    dilation = max(0, int(transition_dilation_px))
    outer = creation
    if dilation > 0:
        outer = creation.filter(ImageFilter.MaxFilter(2 * dilation + 1))
    outer_array = np.asarray(outer, dtype=np.float32) / 255.0
    creation_array = np.asarray(creation, dtype=np.float32) / 255.0
    transition_array = np.clip(outer_array - creation_array, 0.0, 1.0)
    transition = Image.fromarray(np.round(transition_array * 255.0).astype(np.uint8), mode="L")
    if float(transition_feather_px) > 0:
        transition = transition.filter(ImageFilter.GaussianBlur(float(transition_feather_px)))
        transition_array = np.asarray(transition, dtype=np.float32) / 255.0
        transition_array *= 1.0 - creation_array
    preservation_array = np.clip(1.0 - creation_array - transition_array, 0.0, 1.0)
    inpaint_array = np.maximum(creation_array, transition_array)
    return {
        "creation": creation,
        "transition": Image.fromarray(
            np.round(np.clip(transition_array, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L"
        ),
        "preservation": Image.fromarray(
            np.round(preservation_array * 255.0).astype(np.uint8), mode="L"
        ),
        "inpaint": Image.fromarray(
            np.where(inpaint_array > 1e-4, 255, 0).astype(np.uint8), mode="L"
        ),
    }


def weighted_map(
    trimap: dict[str, Image.Image], transition_weight: float
) -> Image.Image:
    creation = np.asarray(trimap["creation"], dtype=np.float32) / 255.0
    transition = np.asarray(trimap["transition"], dtype=np.float32) / 255.0
    weight = np.clip(creation + float(transition_weight) * transition, 0.0, 1.0)
    return Image.fromarray(np.round(weight * 255.0).astype(np.uint8), mode="L")
