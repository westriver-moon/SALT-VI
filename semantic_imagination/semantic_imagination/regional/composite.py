from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


BICUBIC = getattr(Image, "Resampling", Image).BICUBIC


def union_mask(masks: list[np.ndarray], size: tuple[int, int]) -> np.ndarray:
    height, width = int(size[0]), int(size[1])
    result = np.zeros((height, width), dtype=bool)
    for mask in masks:
        candidate = np.asarray(mask, dtype=bool)
        if candidate.shape != result.shape:
            raise ValueError(f"ROI mask shape {candidate.shape} != {(height, width)}")
        result |= candidate
    return result


def soft_mask(mask: np.ndarray, dilation_px: int = 4, feather_px: float = 3.0) -> Image.Image:
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
        image.convert("RGB").save(temporary, format="PNG", compress_level=compress_level)
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
