from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Protocol

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


BICUBIC = getattr(Image, "Resampling", Image).BICUBIC


class SwinBackend(Protocol):
    def restore(self, image: Image.Image, modality: str) -> Image.Image: ...


@dataclass(frozen=True)
class TTASpec:
    name: str
    value: float | int | tuple[int, int]


def qri_tta_specs() -> tuple[TTASpec, ...]:
    return (
        TTASpec("brightness", 0.98),
        TTASpec("brightness", 1.02),
        TTASpec("contrast", 0.98),
        TTASpec("contrast", 1.02),
        TTASpec("jpeg_quality", 95),
        TTASpec("shift", (-1, 0)),
        TTASpec("shift", (1, 0)),
        TTASpec("shift", (0, -1)),
        TTASpec("shift", (0, 1)),
        TTASpec("gaussian_blur", 0.3),
        TTASpec("resize_roundtrip", 0.98),
        TTASpec("resize_roundtrip", 1.02),
    )


def _shift(image: Image.Image, dx: int, dy: int) -> Image.Image:
    array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    shifted = np.empty_like(array)
    shifted[:] = array
    if dx > 0:
        shifted[:, dx:] = array[:, :-dx]
        shifted[:, :dx] = array[:, :1]
    elif dx < 0:
        amount = -dx
        shifted[:, :-amount] = array[:, amount:]
        shifted[:, -amount:] = array[:, -1:]
    if dy > 0:
        shifted[dy:] = shifted[:-dy]
        shifted[:dy] = shifted[:1]
    elif dy < 0:
        amount = -dy
        shifted[:-amount] = shifted[amount:]
        shifted[-amount:] = shifted[-1:]
    return Image.fromarray(shifted, mode="RGB")


def perturb(image: Image.Image, spec: TTASpec) -> Image.Image:
    image = image.convert("RGB")
    if spec.name == "brightness":
        return ImageEnhance.Brightness(image).enhance(float(spec.value))
    if spec.name == "contrast":
        return ImageEnhance.Contrast(image).enhance(float(spec.value))
    if spec.name == "jpeg_quality":
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=int(spec.value), subsampling=0)
        buffer.seek(0)
        with Image.open(buffer) as encoded:
            return encoded.convert("RGB")
    if spec.name == "shift":
        dx, dy = spec.value
        return _shift(image, int(dx), int(dy))
    if spec.name == "gaussian_blur":
        return image.filter(ImageFilter.GaussianBlur(float(spec.value)))
    if spec.name == "resize_roundtrip":
        scale = float(spec.value)
        width = max(2, int(round(image.width * scale)))
        height = max(2, int(round(image.height * scale)))
        return image.resize((width, height), BICUBIC).resize(image.size, BICUBIC)
    raise ValueError(f"unsupported QRI TTA transform: {spec.name}")


def inverse_align(
    restored: Image.Image, spec: TTASpec, lr_size: tuple[int, int]
) -> Image.Image:
    if spec.name != "shift":
        return restored.convert("RGB")
    dx, dy = spec.value
    scale_x = restored.width / float(lr_size[0])
    scale_y = restored.height / float(lr_size[1])
    return _shift(
        restored, -int(round(int(dx) * scale_x)), -int(round(int(dy) * scale_y))
    )


def restore_tta_set(
    backend: SwinBackend,
    lr_image: Image.Image,
    modality: str,
    reference: Image.Image | None = None,
) -> tuple[Image.Image, list[Image.Image]]:
    specs = qri_tta_specs()
    reference = (
        backend.restore(lr_image.convert("RGB"), modality).convert("RGB")
        if reference is None
        else reference.convert("RGB")
    )
    variants = [
        inverse_align(
            backend.restore(perturb(lr_image, spec), modality).convert("RGB"),
            spec,
            lr_image.size,
        )
        for spec in specs
    ]
    if any(image.size != reference.size for image in variants):
        raise ValueError("SwinIR TTA outputs do not share one geometry")
    return reference, variants


def _gray(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("L"), dtype=np.float32) / 255.0


def _high_pass(image: Image.Image) -> np.ndarray:
    base = _gray(image)
    blurred = (
        np.asarray(
            image.convert("L").filter(ImageFilter.GaussianBlur(1.0)), dtype=np.float32
        )
        / 255.0
    )
    return base - blurred


def swin_instability(
    reference: Image.Image,
    variants: list[Image.Image],
    mask: np.ndarray,
) -> float:
    if len(variants) != 12:
        raise ValueError(f"QRI requires 12 SwinIR TTA outputs, got {len(variants)}")
    selected = np.asarray(mask, dtype=bool)
    if selected.shape != (reference.height, reference.width) or not selected.any():
        raise ValueError("regional instability requires a non-empty aligned mask")
    reference_hf = _high_pass(reference)
    residuals = np.stack(
        [np.abs(_high_pass(image) - reference_hf) for image in variants]
    )
    pixel_median = np.median(residuals, axis=0)
    values = pixel_median[selected]
    center = float(np.median(values))
    mad = float(np.median(np.abs(values - center)))
    return center + 1.4826 * mad


def blur_information(image: Image.Image, mask: np.ndarray) -> float:
    selected = np.asarray(mask, dtype=bool)
    high = np.abs(_high_pass(image))[selected]
    contrast = np.std(_gray(image)[selected])
    information = float(np.mean(high) + contrast)
    return float(1.0 / (1.0 + 20.0 * information))


def robust_category_normalize(value: float, median: float, iqr: float) -> float:
    scale = max(float(iqr), 1e-8)
    z = (float(value) - float(median)) / scale
    return float(np.clip(0.5 + 0.25 * z, 0.0, 1.0))
