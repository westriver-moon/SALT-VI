from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageFilter


RESAMPLING = getattr(Image, "Resampling", Image)


@dataclass(frozen=True)
class PersonDetection:
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float
    source: str


class PersonDetector:
    """Small YOLO wrapper with a full-frame, identity-safe fallback."""

    def __init__(self, model_path: str | Path | None, confidence: float = 0.25):
        self.model_path = Path(model_path).expanduser().resolve() if model_path else None
        self.confidence = float(confidence)
        self._model = None

    def _load(self):
        if self.model_path is None:
            return None
        if not self.model_path.is_file():
            raise FileNotFoundError(f"missing person detector weight: {self.model_path}")
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(str(self.model_path))
        return self._model

    def detect(self, image: Image.Image) -> PersonDetection:
        width, height = image.size
        model = self._load()
        if model is None:
            return PersonDetection((0.0, 0.0, float(width), float(height)), 0.0, "full_frame")
        try:
            results = model.predict(image, imgsz=640, device="cpu", verbose=False, save=False)
            candidates: list[tuple[float, Sequence[float]]] = []
            for result in results[:1]:
                for cls, conf, box in zip(
                    result.boxes.cls.cpu().tolist(),
                    result.boxes.conf.cpu().tolist(),
                    result.boxes.xyxy.cpu().tolist(),
                ):
                    if int(cls) == 0 and float(conf) >= self.confidence:
                        candidates.append((float(conf), box))
            if candidates:
                confidence, box = max(candidates, key=lambda item: item[0])
                x1, y1, x2, y2 = (float(value) for value in box)
                return PersonDetection((x1, y1, x2, y2), confidence, "yolov8n")
        except Exception:
            # Tiny infrared crops can be outside COCO detector coverage.  The
            # conservative fallback preserves the full image instead of
            # guessing a crop.
            pass
        return PersonDetection((0.0, 0.0, float(width), float(height)), 0.0, "full_frame")


def _expanded_bbox(
    bbox: Sequence[float], width: int, height: int, margin: float
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = (float(value) for value in bbox)
    dx = max(0.0, x2 - x1) * margin
    dy = max(0.0, y2 - y1) * margin
    return (
        max(0.0, x1 - dx),
        max(0.0, y1 - dy),
        min(float(width), x2 + dx),
        min(float(height), y2 + dy),
    )


def prepare_direct_rewrite(
    source: Image.Image,
    target_size: tuple[int, int] = (256, 512),
) -> tuple[Image.Image, dict]:
    """Preserve an already canonical control canvas without crop or padding."""

    source = source.convert("RGB")
    target_width, target_height = (int(value) for value in target_size)
    if source.size != (target_width, target_height):
        raise ValueError(
            f"direct_rewrite requires source size {target_size}, got {source.size}"
        )
    size = [target_width, target_height]
    return source, {
        "mode": "direct_rewrite",
        "source_size": size,
        "target_size": size,
        "resized_size": size,
        "padding": [0, 0, 0, 0],
        "transform": {
            "scale_x": 1.0,
            "scale_y": 1.0,
            "offset_x": 0.0,
            "offset_y": 0.0,
        },
        "background_restoration": False,
    }


def prepare_control_image(
    source: Image.Image,
    detection: PersonDetection,
    target_size: tuple[int, int] = (256, 512),
    margin: float = 0.05,
    background_blur_radius: float = 24.0,
    foreground_feather_radius: float = 2.0,
) -> tuple[Image.Image, dict]:
    """Fit the complete source over a blurred same-image background.

    ``target_size`` follows PIL's ``(width, height)`` convention.
    """

    source = source.convert("RGB")
    width, height = source.size
    target_width, target_height = (int(value) for value in target_size)
    bbox = _expanded_bbox(detection.bbox_xyxy, width, height, float(margin))
    fit_scale = min(target_width / width, target_height / height)
    fit_width = min(target_width, max(1, round(width * fit_scale)))
    fit_height = min(target_height, max(1, round(height * fit_scale)))
    left = (target_width - fit_width) // 2
    top = (target_height - fit_height) // 2
    right = target_width - fit_width - left
    bottom = target_height - fit_height - top

    cover_scale = max(target_width / width, target_height / height)
    cover_width = max(target_width, round(width * cover_scale))
    cover_height = max(target_height, round(height * cover_scale))
    cover_left = (cover_width - target_width) // 2
    cover_top = (cover_height - target_height) // 2
    background = source.resize((cover_width, cover_height), RESAMPLING.LANCZOS).crop(
        (cover_left, cover_top, cover_left + target_width, cover_top + target_height)
    )
    background = background.filter(ImageFilter.GaussianBlur(background_blur_radius))
    foreground = source.resize((fit_width, fit_height), RESAMPLING.LANCZOS)
    foreground_layer = background.copy()
    foreground_layer.paste(foreground, (left, top))
    mask = Image.new("L", (target_width, target_height), 0)
    mask.paste(255, (left, top, left + fit_width, top + fit_height))
    if foreground_feather_radius > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(foreground_feather_radius))
    control = Image.composite(foreground_layer, background, mask)
    geometry = {
        "mode": "person_fit_blurred_background",
        "scale": fit_scale,
        "resized_size": [fit_width, fit_height],
        "crop_box": None,
        "padding": [left, top, right, bottom],
        "foreground_box": [left, top, left + fit_width, top + fit_height],
        "background_cover_scale": cover_scale,
        "background_resized_size": [cover_width, cover_height],
        "background_crop_box": [
            cover_left,
            cover_top,
            cover_left + target_width,
            cover_top + target_height,
        ],
        "background_blur_radius": float(background_blur_radius),
        "foreground_feather_radius": float(foreground_feather_radius),
    }

    if control.size != (target_width, target_height):
        raise AssertionError(f"adaptive geometry produced {control.size}, expected {target_size}")
    geometry.update(
        {
            "source_size": [width, height],
            "target_size": [target_width, target_height],
            "person_detection": asdict(detection),
            "expanded_person_bbox": list(bbox),
            "person_margin": float(margin),
            "source_was_larger_than_target": width > target_width or height > target_height,
        }
    )
    return control, geometry


def restore_blurred_background(
    output: Image.Image,
    geometry: dict,
    source_background: Image.Image | None = None,
) -> Image.Image:
    """Keep PASD detail only in the aspect-preserved foreground frame.

    When provided, ``source_background`` is the 256×512 control canvas made
    from the same source image.  Consequently the final padded region cannot
    inherit PASD-hallucinated background content.
    """

    output = output.convert("RGB")
    if source_background is None:
        background = output.filter(
            ImageFilter.GaussianBlur(float(geometry["background_blur_radius"]))
        )
    else:
        background = source_background.convert("RGB")
        if background.size != output.size:
            raise ValueError("source background size does not match PASD output")
    box = tuple(int(value) for value in geometry["foreground_box"])
    foreground = output.crop(box)
    foreground_layer = background.copy()
    foreground_layer.paste(foreground, box[:2])
    mask = Image.new("L", output.size, 0)
    mask.paste(255, box)
    feather = float(geometry["foreground_feather_radius"])
    if feather > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(feather))
    return Image.composite(foreground_layer, background, mask)
