from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


BICUBIC = getattr(Image, "Resampling", Image).BICUBIC
LANCZOS = getattr(Image, "Resampling", Image).LANCZOS


def union_mask(masks: list[np.ndarray], size: tuple[int, int]) -> np.ndarray:
    height, width = int(size[0]), int(size[1])
    result = np.zeros((height, width), dtype=bool)
    for mask in masks:
        candidate = np.asarray(mask, dtype=bool)
        if candidate.shape != result.shape:
            raise ValueError(f"ROI mask shape {candidate.shape} != {(height, width)}")
        result |= candidate
    return result


def soft_mask(
    mask: np.ndarray, dilation_px: int = 4, feather_px: float = 3.0
) -> Image.Image:
    binary = Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255, mode="L")
    if dilation_px:
        binary = binary.filter(ImageFilter.MaxFilter(2 * int(dilation_px) + 1))
    if feather_px:
        binary = binary.filter(ImageFilter.GaussianBlur(float(feather_px)))
    return binary


def masked_composite(
    swin: Image.Image,
    pasd: Image.Image,
    mask: Image.Image,
) -> Image.Image:
    swin = swin.convert("RGB")
    pasd = pasd.convert("RGB")
    if pasd.size != swin.size or mask.size != swin.size:
        raise ValueError("SwinIR, PASD, and regional mask must share one geometry")
    return Image.composite(pasd, swin, mask.convert("L"))


def roi_crop_box(
    bbox_xyxy: tuple[int, int, int, int],
    image_size: tuple[int, int],
    *,
    context_scale: float,
    target_size: tuple[int, int] = (256, 512),
) -> tuple[int, int, int, int]:
    """Return an in-bounds, target-aspect crop containing one semantic ROI."""
    image_width, image_height = (int(value) for value in image_size)
    left, top, right, bottom = (int(value) for value in bbox_xyxy)
    if not (0 <= left < right <= image_width and 0 <= top < bottom <= image_height):
        raise ValueError(f"invalid ROI bbox {bbox_xyxy} for image {image_size}")
    if float(context_scale) < 1.0:
        raise ValueError("ROI context scale must be at least 1")
    target_width, target_height = (int(value) for value in target_size)
    if target_width <= 0 or target_height <= 0:
        raise ValueError("ROI PASD target size must be positive")

    target_aspect = target_width / target_height
    roi_width = (right - left) * float(context_scale)
    roi_height = (bottom - top) * float(context_scale)
    crop_height = max(roi_height, roi_width / target_aspect)
    crop_width = crop_height * target_aspect
    if crop_width > image_width:
        crop_width = float(image_width)
        crop_height = crop_width / target_aspect
    if crop_height > image_height:
        crop_height = float(image_height)
        crop_width = crop_height * target_aspect

    crop_width = max(right - left, min(image_width, int(round(crop_width))))
    crop_height = max(bottom - top, min(image_height, int(round(crop_height))))
    center_x = 0.5 * (left + right)
    center_y = 0.5 * (top + bottom)
    crop_left = int(round(center_x - crop_width / 2))
    crop_top = int(round(center_y - crop_height / 2))
    crop_left = max(0, min(crop_left, image_width - crop_width))
    crop_top = max(0, min(crop_top, image_height - crop_height))
    crop_right = crop_left + crop_width
    crop_bottom = crop_top + crop_height
    if not (
        crop_left <= left
        and crop_top <= top
        and crop_right >= right
        and crop_bottom >= bottom
    ):
        raise ValueError("computed ROI crop does not contain the semantic bbox")
    return crop_left, crop_top, crop_right, crop_bottom


def roi_control_image(
    image: Image.Image,
    crop_box: tuple[int, int, int, int],
    target_size: tuple[int, int] = (256, 512),
) -> Image.Image:
    return image.convert("RGB").crop(crop_box).resize(target_size, BICUBIC)


def paste_roi_realization(
    base: Image.Image,
    generated_crop: Image.Image,
    crop_box: tuple[int, int, int, int],
    mask: Image.Image,
) -> Image.Image:
    """Map a canonical PASD crop back without changing the full-image geometry."""
    base = base.convert("RGB")
    if mask.size != base.size:
        raise ValueError("regional PASD mask must match the full SwinIR canvas")
    left, top, right, bottom = crop_box
    raw = base.copy()
    rewritten = generated_crop.convert("RGB").resize(
        (right - left, bottom - top), LANCZOS
    )
    raw.paste(rewritten, (left, top))
    return masked_composite(base, raw, mask)


def lr_cycle_energy(composite: Image.Image, lr: Image.Image, modality: str) -> float:
    degraded = composite.convert("RGB").resize(lr.size, BICUBIC)
    reference = lr.convert("RGB")
    if modality.lower() == "ir":
        degraded = degraded.convert("L")
        reference = reference.convert("L")
    left = np.asarray(degraded, dtype=np.float32) / 255.0
    right = np.asarray(reference, dtype=np.float32) / 255.0
    return float(np.mean(np.abs(left - right)))


def edit_energy(pasd: Image.Image, swin: Image.Image, mask: Image.Image) -> float:
    generated = np.asarray(pasd.convert("RGB"), dtype=np.float32) / 255.0
    baseline = np.asarray(swin.convert("RGB"), dtype=np.float32) / 255.0
    alpha = np.asarray(mask.convert("L"), dtype=np.float32) / 255.0
    outside = 1.0 - alpha
    outside_error = float(
        np.sum(np.mean(np.abs(generated - baseline), axis=2) * outside)
        / max(float(np.sum(outside)), 1.0)
    )
    edit_fraction = float(np.mean(alpha))
    return outside_error + 0.1 * edit_fraction


def atomic_png(path: str | Path, image: Image.Image, compress_level: int = 4) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.png")
    try:
        image.convert("RGB").save(
            temporary, format="PNG", compress_level=compress_level
        )
        with Image.open(temporary) as check:
            check.verify()
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def atomic_mask(path: str | Path, mask: Image.Image, compress_level: int = 4) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.png")
    try:
        mask.convert("L").save(temporary, format="PNG", compress_level=compress_level)
        with Image.open(temporary) as check:
            check.verify()
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path
