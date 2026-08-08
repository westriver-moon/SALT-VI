from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image


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


def _clamp(value: int, lower: int, upper: int) -> int:
    return min(max(value, lower), upper)


def prepare_control_image(
    source: Image.Image,
    detection: PersonDetection,
    target_size: tuple[int, int] = (256, 512),
    margin: float = 0.05,
) -> tuple[Image.Image, dict]:
    """Return a fixed canvas using person-safe crop, otherwise edge padding.

    ``target_size`` follows PIL's ``(width, height)`` convention.
    """

    source = source.convert("RGB")
    width, height = source.size
    target_width, target_height = (int(value) for value in target_size)
    bbox = _expanded_bbox(detection.bbox_xyxy, width, height, float(margin))

    cover_scale = max(target_width / width, target_height / height)
    cover_width = max(target_width, int(np.ceil(width * cover_scale)))
    cover_height = max(target_height, int(np.ceil(height * cover_scale)))
    sx, sy = cover_width / width, cover_height / height
    scaled_bbox = (bbox[0] * sx, bbox[1] * sy, bbox[2] * sx, bbox[3] * sy)
    center_x = (scaled_bbox[0] + scaled_bbox[2]) / 2.0
    center_y = (scaled_bbox[1] + scaled_bbox[3]) / 2.0
    crop_left = _clamp(round(center_x - target_width / 2), 0, cover_width - target_width)
    crop_top = _clamp(round(center_y - target_height / 2), 0, cover_height - target_height)
    crop_box = (
        crop_left,
        crop_top,
        crop_left + target_width,
        crop_top + target_height,
    )
    safe_crop = (
        scaled_bbox[0] >= crop_box[0]
        and scaled_bbox[1] >= crop_box[1]
        and scaled_bbox[2] <= crop_box[2]
        and scaled_bbox[3] <= crop_box[3]
    )

    if safe_crop:
        resized = source.resize((cover_width, cover_height), RESAMPLING.LANCZOS)
        control = resized.crop(crop_box)
        geometry = {
            "mode": "person_safe_cover_crop",
            "scale": cover_scale,
            "resized_size": [cover_width, cover_height],
            "crop_box": list(crop_box),
            "padding": [0, 0, 0, 0],
        }
    else:
        fit_scale = min(target_width / width, target_height / height)
        fit_width = min(target_width, max(1, round(width * fit_scale)))
        fit_height = min(target_height, max(1, round(height * fit_scale)))
        fx, fy = fit_width / width, fit_height / height
        bbox_center_x = (bbox[0] + bbox[2]) * 0.5 * fx
        bbox_center_y = (bbox[1] + bbox[3]) * 0.5 * fy
        left = _clamp(round(target_width / 2 - bbox_center_x), 0, target_width - fit_width)
        top = _clamp(round(target_height / 2 - bbox_center_y), 0, target_height - fit_height)
        right = target_width - fit_width - left
        bottom = target_height - fit_height - top
        resized = source.resize((fit_width, fit_height), RESAMPLING.LANCZOS)
        pixels = np.asarray(resized)
        padded = np.pad(pixels, ((top, bottom), (left, right), (0, 0)), mode="edge")
        control = Image.fromarray(padded, mode="RGB")
        geometry = {
            "mode": "person_fit_edge_pad",
            "scale": fit_scale,
            "resized_size": [fit_width, fit_height],
            "crop_box": None,
            "padding": [left, top, right, bottom],
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
