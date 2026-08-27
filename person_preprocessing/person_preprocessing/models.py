"""Lazy, single-checkpoint YOLO pose inference for prepared person assets."""

from pathlib import Path

import numpy as np


class PoseEstimator:
    """Load one COCO-17 pose checkpoint and select the central person."""

    def __init__(self, weight, device="0", confidence=0.25, imgsz=640, rect=None):
        self.weight = Path(weight).expanduser().resolve()
        if not self.weight.is_file():
            raise FileNotFoundError(self.weight)
        self.device = device
        self.confidence = float(confidence)
        self.imgsz = int(imgsz)
        self.rect = rect
        self.model = None

    def predict(self, image):
        from ultralytics import YOLO

        if self.model is None:
            self.model = YOLO(str(self.weight))
            if self.model.task != "pose":
                raise ValueError("checkpoint is not a pose model")
        options = {}
        if self.rect is not None:
            options["rect"] = bool(self.rect)
        result = self.model.predict(
            image,
            device=self.device,
            imgsz=self.imgsz,
            conf=self.confidence,
            classes=[0],
            verbose=False,
            save=False,
            half=False,
            **options,
        )[0]
        if not len(result.boxes):
            return None
        boxes = result.boxes.xyxy.cpu().numpy()
        confidence = result.boxes.conf.cpu().numpy()
        width, height = image.size
        centers = (boxes[:, :2] + boxes[:, 2:]) / 2
        distance = np.linalg.norm(
            (centers - [width / 2, height / 2]) / [width, height], axis=1
        )
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        index = int(np.argmax(confidence * np.sqrt(area) / (1 + distance)))
        points = result.keypoints.data[index].cpu().numpy().astype(np.float32)
        if points.shape != (17, 3):
            raise ValueError("person assets require COCO-17 keypoints with confidences")
        if not np.isfinite(points).all():
            raise ValueError("pose checkpoint produced non-finite keypoints")
        return {
            "bbox": boxes[index].astype(np.float32),
            "confidence": float(confidence[index]),
            "keypoints": points,
            "model_path": str(self.weight),
            "person_count": int(len(boxes)),
        }
