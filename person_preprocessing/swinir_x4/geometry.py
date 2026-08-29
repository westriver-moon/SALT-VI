"""Optional post-SR geometry, adapted from SALT-VI/pasd_plugin/geometry.py.

Preserves the full source; YOLO guides bounded translation, never a crop.
Only Pillow/NumPy are required for layout. YOLO is loaded lazily on CPU.
"""
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

RESAMPLING = getattr(Image, "Resampling", Image)
ALGORITHM = "full_frame_fit_yolo_translation_protected_feather_v1"


def validate_settings(settings):
    if type(settings["enabled"]) is not bool or type(settings["detector"]["enabled"]) is not bool:
        raise ValueError("Geometry/detector enabled must be booleans")
    for key in ("target_width", "target_height", "cpu_threads"):
        if type(settings[key]) is not int or settings[key] < 1:
            raise ValueError("Expected positive integer: " + key)
    for key in ("background_blur_radius", "foreground_feather_radius", "person_margin"):
        value = float(settings[key])
        if not math.isfinite(value) or value < 0:
            raise ValueError("Expected nonnegative finite value: " + key)
    detector = settings["detector"]
    for key in ("confidence", "min_area_fraction", "ambiguity_ratio"):
        value = float(detector[key])
        if not math.isfinite(value) or not 0 < value <= 1:
            raise ValueError("Expected 0 < value <= 1: " + key)
    if type(detector["imgsz"]) is not int or detector["imgsz"] < 32 or detector["imgsz"] % 32:
        raise ValueError("Detector imgsz must be a positive multiple of 32")


def select_person(detections, source_size, settings):
    """Select a dominant, central person; ambiguous people leave layout centered."""
    width, height = source_size
    candidates = []
    for detection in detections:
        box = [float(v) for v in detection["bbox_xyxy"]]
        confidence = float(detection["confidence"])
        if len(box) != 4 or not all(math.isfinite(v) for v in box + [confidence]):
            continue
        if confidence < settings["confidence"] or confidence > 1:
            continue
        x1, y1, x2, y2 = box
        x1, x2 = max(0., min(width, x1)), max(0., min(width, x2))
        y1, y2 = max(0., min(height, y1)), max(0., min(height, y2))
        area = max(0., x2-x1) * max(0., y2-y1) / (width*height)
        if area < settings["min_area_fraction"]:
            continue
        distance = math.hypot(((x1+x2)/2-width/2)/(width/2),
                              ((y1+y2)/2-height/2)/(height/2)) / math.sqrt(2)
        score = confidence * math.sqrt(area) * max(0.1, 1-distance)
        candidates.append({"bbox_xyxy": [x1, y1, x2, y2], "confidence": confidence,
                           "area_fraction": area, "selection_score": score})
    candidates.sort(key=lambda d: (-d["selection_score"], d["bbox_xyxy"]))
    if not candidates:
        return None, candidates, "no_reliable_person"
    if len(candidates) > 1 and candidates[1]["selection_score"] >= candidates[0]["selection_score"] * settings["ambiguity_ratio"]:
        return None, candidates, "ambiguous_people"
    return candidates[0], candidates, "selected_person"


def _expanded(box, width, height, margin):
    x1, y1, x2, y2 = box
    dx, dy = (x2-x1)*margin, (y2-y1)*margin
    return [max(0., x1-dx), max(0., y1-dy), min(float(width), x2+dx), min(float(height), y2+dy)]


def fit_with_background(source, settings, detections=(), detector_status="disabled"):
    """Return a fixed-size RGB canvas and an auditable source-to-canvas map."""
    source = source.convert("RGB")
    width, height = source.size
    tw, th = settings["target_width"], settings["target_height"]
    scale = min(tw/width, th/height)
    fw, fh = min(tw, max(1, round(width*scale))), min(th, max(1, round(height*scale)))
    slack_x, slack_y = tw-fw, th-fh
    centered = [slack_x//2, slack_y//2]
    selected, candidates, reason = select_person(detections, source.size, settings["detector"])
    left, top = centered
    sx, sy = fw/width, fh/height  # Rounded raster sizes; discrepancy <= 1 pixel.
    if selected is not None:
        x1, y1, x2, y2 = selected["bbox_xyxy"]
        left = min(slack_x, max(0, round(tw/2-(x1+x2)/2*sx)))
        top = min(slack_y, max(0, round(th/2-(y1+y2)/2*sy)))
    right, bottom = slack_x-left, slack_y-top

    cover_scale = max(tw/width, th/height)
    cw, ch = max(tw, round(width*cover_scale)), max(th, round(height*cover_scale))
    cl, ct = (cw-tw)//2, (ch-th)//2
    background = source.resize((cw, ch), RESAMPLING.LANCZOS).crop((cl, ct, cl+tw, ct+th))
    background = background.filter(ImageFilter.GaussianBlur(settings["background_blur_radius"]))
    foreground = source.resize((fw, fh), RESAMPLING.LANCZOS)
    layer = background.copy()
    layer.paste(foreground, (left, top))
    mask = Image.new("L", (tw, th), 0)
    mask.paste(255, (left, top, left+fw, top+fh))
    if settings["foreground_feather_radius"] > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(settings["foreground_feather_radius"]))
    protected = []
    # All reliable person boxes are protected, even if primary selection is ambiguous.
    for candidate in candidates:
        box = _expanded(candidate["bbox_xyxy"], width, height, settings["person_margin"])
        mapped = [max(left, math.floor(left+box[0]*sx)), max(top, math.floor(top+box[1]*sy)),
                  min(left+fw, math.ceil(left+box[2]*sx)), min(top+fh, math.ceil(top+box[3]*sy))]
        mask.paste(255, tuple(mapped))
        protected.append(mapped)
    output = Image.composite(layer, background, mask)
    metadata = {
        "algorithm": ALGORITHM, "source_size": [width, height], "target_size": [tw, th],
        "scale": scale, "resized_size": [fw, fh], "crop_box": None,
        "padding": [left, top, right, bottom], "foreground_box": [left, top, left+fw, top+fh],
        "transform": {"scale_x": sx, "scale_y": sy, "offset_x": left, "offset_y": top},
        "centered_offset": centered, "person_guided_offset": [left, top],
        "offset_changed_by_yolo": [left, top] != centered,
        "placement_reason": reason if detector_status == "ok" else detector_status,
        "detector_status": detector_status, "selected_person": selected, "candidates": candidates,
        "protected_person_boxes": protected, "person_margin": settings["person_margin"],
        "background_cover_scale": cover_scale, "background_resized_size": [cw, ch],
        "background_crop_box": [cl, ct, cl+tw, ct+th],
        "background_blur_radius": settings["background_blur_radius"],
        "foreground_feather_radius": settings["foreground_feather_radius"],
    }
    return output, metadata


class CpuPersonDetector:
    def __init__(self, settings, cpu_threads=4):
        self.settings, self.model = settings, None
        if not settings["enabled"]:
            return
        path = Path(settings["model_path"])
        if not path.is_file():
            raise FileNotFoundError("Local YOLO weight is missing: " + str(path))
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["YOLO_CONFIG_DIR"] = str(Path(__file__).resolve().parent/".cache/ultralytics")
        os.environ["YOLO_AUTOINSTALL"] = "False"
        from ultralytics import YOLO
        import torch
        torch.set_num_threads(cpu_threads)
        self.model = YOLO(str(path))

    def detect(self, image):
        if self.model is None:
            return [], "disabled"
        # System/dependency failures are visible; only actual no-detection falls back.
        results = self.model.predict(image, imgsz=self.settings["imgsz"], device="cpu",
                                     classes=[0], conf=self.settings["confidence"],
                                     verbose=False, save=False, stream=False)
        candidates = []
        for result in results[:1]:
            for box, confidence in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist()):
                candidates.append({"bbox_xyxy": box, "confidence": confidence})
        return candidates, "ok"


def ensure_ir_channels(image, modality):
    """Geometry is channel symmetric; enforce the existing mean/replicate contract."""
    if modality == "ir":
        pixels = np.asarray(image, dtype=np.uint8)
        gray = np.rint(pixels.astype(np.float32).mean(axis=-1)).astype(np.uint8)
        return Image.fromarray(np.repeat(gray[..., None], 3, axis=-1))
    return image
