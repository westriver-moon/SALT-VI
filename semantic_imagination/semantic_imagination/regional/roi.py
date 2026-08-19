from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image

from .schema import Region


COCO_KEYPOINTS = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}

LIP_PARTS = {
    "hat": 1,
    "hair": 2,
    "sunglasses": 4,
    "upper_clothes": 5,
    "dress": 6,
    "coat": 7,
    "socks": 8,
    "pants": 9,
    "jumpsuit": 10,
    "scarf": 11,
    "skirt": 12,
    "face": 13,
    "left_arm": 14,
    "right_arm": 15,
    "left_leg": 16,
    "right_leg": 17,
    "left_shoe": 18,
    "right_shoe": 19,
}


class PoseBackend(Protocol):
    def infer(self, image: Image.Image) -> dict[str, object]: ...


class ParsingBackend(Protocol):
    def infer(self, image: Image.Image) -> np.ndarray: ...


class SamBackend(Protocol):
    def refine(
        self, image: Image.Image, bbox_xyxy: tuple[int, int, int, int], seed_mask: np.ndarray
    ) -> np.ndarray: ...


def _clip_bbox(
    bbox: tuple[float, float, float, float], width: int, height: int
) -> tuple[int, int, int, int]:
    left, top, right, bottom = bbox
    left = max(0, min(width - 1, int(round(left))))
    top = max(0, min(height - 1, int(round(top))))
    right = max(left + 1, min(width, int(round(right))))
    bottom = max(top + 1, min(height, int(round(bottom))))
    return left, top, right, bottom


def _bbox_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    left, top, right, bottom = bbox
    mask[top:bottom, left:right] = True
    return mask


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if not len(xs):
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _point_box(
    point: tuple[float, float] | None,
    size: tuple[float, float],
    image_size: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    if point is None:
        return None
    width, height = size
    x, y = point
    return _clip_bbox(
        (x - width / 2, y - height / 2, x + width / 2, y + height / 2),
        *image_size,
    )


def _keypoint(pose: dict[str, object], name: str) -> tuple[float, float] | None:
    point = dict(pose.get("keypoints", {})).get(name)
    if not point or float(point[2]) < 0.15:
        return None
    return float(point[0]), float(point[1])


def _parsing_mask(parsing: np.ndarray, names: tuple[str, ...]) -> np.ndarray:
    ids = [LIP_PARTS[name] for name in names]
    return np.isin(parsing, ids)


@dataclass
class HumanROIGenerator:
    pose: PoseBackend
    parsing: ParsingBackend
    sam: SamBackend
    strict: bool = True

    def _region(
        self,
        image: Image.Image,
        parsing: np.ndarray,
        region_id: str,
        category: str,
        bbox: tuple[int, int, int, int] | None,
        part_names: tuple[str, ...] = (),
        side: str | None = None,
    ) -> Region | None:
        if bbox is None and part_names:
            bbox = _mask_bbox(_parsing_mask(parsing, part_names))
        if bbox is None:
            return None
        base = _bbox_mask((image.height, image.width), bbox)
        if part_names:
            part = _parsing_mask(parsing, part_names)
            if part.any():
                base &= part
        if not base.any():
            base = _bbox_mask((image.height, image.width), bbox)
        refined = np.asarray(self.sam.refine(image, bbox, base), dtype=bool)
        if refined.shape != base.shape:
            raise ValueError(f"SAM mask shape {refined.shape} does not match image {base.shape}")
        refined &= _bbox_mask(base.shape, bbox)
        if not refined.any():
            refined = base
        return Region(region_id, category, bbox, refined, side=side)

    def regions(self, image: Image.Image, modality: str) -> list[Region]:
        image = image.convert("RGB")
        pose = self.pose.infer(image)
        parsing = np.asarray(self.parsing.infer(image))
        if parsing.shape != (image.height, image.width):
            parsing = np.asarray(
                Image.fromarray(parsing.astype(np.uint8)).resize(
                    image.size, getattr(Image, "Resampling", Image).NEAREST
                )
            )
        person_bbox = pose.get("bbox_xyxy")
        if person_bbox is None:
            if self.strict:
                raise ValueError("pose backend did not detect a person")
            person_bbox = (0, 0, image.width, image.height)
        person_bbox = _clip_bbox(tuple(person_bbox), image.width, image.height)
        left, top, right, bottom = person_bbox
        person_w, person_h = right - left, bottom - top

        eyes = [_keypoint(pose, "left_eye"), _keypoint(pose, "right_eye")]
        visible_eyes = [point for point in eyes if point is not None]
        eye_center = (
            (sum(point[0] for point in visible_eyes) / len(visible_eyes),
             sum(point[1] for point in visible_eyes) / len(visible_eyes))
            if visible_eyes else (left + person_w * 0.5, top + person_h * 0.09)
        )
        head_bbox = _clip_bbox(
            (left + person_w * 0.25, top, right - person_w * 0.25, top + person_h * 0.22),
            image.width,
            image.height,
        )
        eye_bbox = _point_box(
            eye_center, (max(12, person_w * 0.55), max(10, person_h * 0.10)), image.size
        )
        wrist_size = (max(12, person_w * 0.24), max(12, person_h * 0.10))
        shoe_size = (max(16, person_w * 0.36), max(14, person_h * 0.12))
        pocket_size = (max(18, person_w * 0.35), max(20, person_h * 0.18))

        specs = [
            ("eyes", "eyewear", eye_bbox, ("sunglasses", "face"), None),
            ("head", "headwear", head_bbox, ("hat", "hair"), None),
            ("left_wrist", "wrist_accessory", _point_box(_keypoint(pose, "left_wrist"), wrist_size, image.size), (), "left"),
            ("right_wrist", "wrist_accessory", _point_box(_keypoint(pose, "right_wrist"), wrist_size, image.size), (), "right"),
            ("left_arm", "body_marking", None, ("left_arm",), "left"),
            ("right_arm", "body_marking", None, ("right_arm",), "right"),
            ("upper_torso", "clothing_detail", None, ("upper_clothes", "dress", "coat", "jumpsuit"), None),
            ("left_pocket", "pocket_item", _point_box(_keypoint(pose, "left_hip"), pocket_size, image.size), (), "left"),
            ("right_pocket", "pocket_item", _point_box(_keypoint(pose, "right_hip"), pocket_size, image.size), (), "right"),
            ("left_carried", "carried_object", _clip_bbox((left - person_w * .35, top + person_h * .18, left + person_w * .30, top + person_h * .78), image.width, image.height), (), "left"),
            ("right_carried", "carried_object", _clip_bbox((right - person_w * .30, top + person_h * .18, right + person_w * .35, top + person_h * .78), image.width, image.height), (), "right"),
            ("left_foot", "footwear_detail", _point_box(_keypoint(pose, "left_ankle"), shoe_size, image.size), ("left_shoe", "socks"), "left"),
            ("right_foot", "footwear_detail", _point_box(_keypoint(pose, "right_ankle"), shoe_size, image.size), ("right_shoe", "socks"), "right"),
        ]
        regions = []
        for spec in specs:
            region = self._region(image, parsing, *spec)
            if region is not None and int(np.asarray(region.mask).sum()) >= 9:
                regions.append(region)
        if not regions:
            raise ValueError(f"ROI stack produced no usable {modality} regions")
        return regions


class UltralyticsPoseBackend:
    def __init__(self, weights: str | Path, device: str = "cuda:0", confidence: float = 0.25):
        import os

        weights = Path(weights).expanduser().resolve()
        settings_root = weights.parent.parent / "runtime" / "ultralytics"
        settings_root.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("YOLO_CONFIG_DIR", str(settings_root))
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.device = device
        self.confidence = float(confidence)

    def infer(self, image: Image.Image) -> dict[str, object]:
        results = self.model.predict(
            source=np.asarray(image.convert("RGB")),
            device=self.device,
            conf=self.confidence,
            verbose=False,
        )
        result = results[0]
        if result.boxes is None or len(result.boxes) == 0 or result.keypoints is None:
            return {}
        boxes = result.boxes.xyxy.detach().cpu().numpy()
        best = int(np.argmax((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])))
        points = result.keypoints.data[best].detach().cpu().numpy()
        return {
            "bbox_xyxy": tuple(float(value) for value in boxes[best]),
            "keypoints": {
                name: tuple(float(value) for value in points[index])
                for name, index in COCO_KEYPOINTS.items()
            },
        }


class SCHPLIPTorchScriptBackend:
    """Adapter for an externally exported SCHP-LIP TorchScript segmenter."""

    def __init__(self, weights: str | Path, device: str = "cuda:0"):
        import torch

        self.torch = torch
        self.device = torch.device(device)
        self.model = torch.jit.load(str(weights), map_location=self.device).eval()

    def infer(self, image: Image.Image) -> np.ndarray:
        torch = self.torch
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(self.device)
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        with torch.no_grad():
            output = self.model((tensor - mean) / std)
        if isinstance(output, (list, tuple)):
            output = output[-1]
        if isinstance(output, dict):
            output = output.get("out", next(iter(output.values())))
        labels = output.argmax(1)[0].detach().cpu().numpy().astype(np.uint8)
        return labels


class SCHPLIPBackend:
    """Direct adapter for the official Self-Correction-Human-Parsing repository."""

    def __init__(
        self,
        repository: str | Path,
        weights: str | Path,
        device: str = "cuda:0",
    ):
        import importlib
        import sys
        import types

        import torch

        self.torch = torch
        self.device = torch.device(device)
        self.repository = Path(repository).expanduser().resolve()
        if not (self.repository / "networks" / "__init__.py").is_file():
            raise FileNotFoundError(f"invalid SCHP repository: {self.repository}")
        # The official SCHP checkout bundles a 2019 InPlaceABNSync CUDA extension.
        # Its serialized parameters are ordinary BatchNorm parameters, so inference
        # can use this API-compatible implementation without compiling an obsolete
        # extension into the active SALT/PASD environment.
        class InPlaceABNSync(torch.nn.BatchNorm2d):
            def __init__(
                self,
                num_features: int,
                activation: str = "leaky_relu",
                activation_param: float = 0.01,
                **kwargs: object,
            ):
                kwargs.pop("devices", None)
                super().__init__(num_features, **kwargs)
                self.activation = activation
                self.activation_param = float(activation_param)

            def forward(self, value):
                value = super().forward(value)
                if self.activation == "none":
                    return value
                if self.activation == "relu":
                    return torch.nn.functional.relu(value, inplace=False)
                if self.activation == "elu":
                    return torch.nn.functional.elu(
                        value, alpha=self.activation_param, inplace=False
                    )
                return torch.nn.functional.leaky_relu(
                    value, negative_slope=self.activation_param, inplace=False
                )

        compatibility_module = types.ModuleType("modules")
        compatibility_module.InPlaceABN = InPlaceABNSync
        compatibility_module.InPlaceABNSync = InPlaceABNSync
        previous_modules = sys.modules.get("modules")
        sys.path.insert(0, str(self.repository))
        try:
            sys.modules["modules"] = compatibility_module
            networks = importlib.import_module("networks")
            transforms = importlib.import_module("utils.transforms")
        finally:
            if previous_modules is None:
                sys.modules.pop("modules", None)
            else:
                sys.modules["modules"] = previous_modules
        for module in (networks, transforms):
            Path(module.__file__).resolve().relative_to(self.repository)
        self.get_affine_transform = transforms.get_affine_transform
        self.transform_logits = transforms.transform_logits
        self.input_size = np.asarray([473, 473])
        self.model = networks.init_model("resnet101", num_classes=20, pretrained=None)
        payload = torch.load(str(weights), map_location="cpu")
        state = payload["state_dict"]
        state = {
            (name[7:] if name.startswith("module.") else name): value
            for name, value in state.items()
        }
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        invalid_missing = [
            name for name in missing if not name.endswith(".num_batches_tracked")
        ]
        if invalid_missing or unexpected:
            raise ValueError(
                "SCHP checkpoint/model mismatch: "
                f"missing={invalid_missing[:8]}, unexpected={unexpected[:8]}"
            )
        self.model.to(self.device).eval()
        # Do not leak SCHP's generic top-level `networks` / `utils` packages into
        # PASD or SALT imports in the same process.
        for name, module in list(sys.modules.items()):
            module_path = getattr(module, "__file__", None)
            if not module_path or not (
                name == "networks"
                or name.startswith("networks.")
                or name == "utils"
                or name.startswith("utils.")
            ):
                continue
            try:
                Path(module_path).resolve().relative_to(self.repository)
            except ValueError:
                continue
            sys.modules.pop(name, None)
        try:
            sys.path.remove(str(self.repository))
        except ValueError:
            pass

    def infer(self, image: Image.Image) -> np.ndarray:
        import cv2

        torch = self.torch
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        bgr = np.ascontiguousarray(rgb[..., ::-1])
        height, width = bgr.shape[:2]
        center = np.asarray([(width - 1) * 0.5, (height - 1) * 0.5], dtype=np.float32)
        side = float(max(width - 1, height - 1))
        scale = np.asarray([side, side], dtype=np.float32)
        affine = self.get_affine_transform(center, scale, 0, self.input_size)
        warped = cv2.warpAffine(
            bgr,
            affine,
            (int(self.input_size[1]), int(self.input_size[0])),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        tensor = torch.from_numpy(warped.astype(np.float32) / 255.0).permute(2, 0, 1)
        mean = torch.tensor([0.406, 0.456, 0.485]).view(3, 1, 1)
        std = torch.tensor([0.225, 0.224, 0.229]).view(3, 1, 1)
        tensor = ((tensor - mean) / std).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(tensor)
            logits = torch.nn.functional.interpolate(
                output[0][-1][0].unsqueeze(0),
                size=tuple(int(value) for value in self.input_size),
                mode="bilinear",
                align_corners=True,
            )[0].permute(1, 2, 0)
        restored = self.transform_logits(
            logits.detach().cpu().numpy(),
            center,
            scale,
            width,
            height,
            input_size=self.input_size,
        )
        return np.argmax(restored, axis=2).astype(np.uint8)


class SegmentAnythingBackend:
    def __init__(
        self,
        weights: str | Path,
        device: str = "cuda:0",
        repository: str | Path | None = None,
    ):
        import importlib
        import sys

        if repository is not None:
            root = Path(repository).expanduser().resolve()
            sys.path.insert(0, str(root))
            module = importlib.import_module("segment_anything")
            Path(module.__file__).resolve().relative_to(root)
        from segment_anything import SamPredictor, sam_model_registry

        model = sam_model_registry["vit_b"](checkpoint=str(weights)).to(device)
        self.predictor = SamPredictor(model)

    def refine(
        self, image: Image.Image, bbox_xyxy: tuple[int, int, int, int], seed_mask: np.ndarray
    ) -> np.ndarray:
        self.predictor.set_image(np.asarray(image.convert("RGB")))
        masks, scores, _ = self.predictor.predict(
            box=np.asarray(bbox_xyxy, dtype=np.float32), multimask_output=True
        )
        candidates = [np.asarray(mask, dtype=bool) for mask in masks]
        if seed_mask.any():
            overlaps = [
                float((mask & seed_mask).sum())
                / max(1.0, float((mask | seed_mask).sum()))
                for mask in candidates
            ]
            best = int(
                np.argmax(
                    np.asarray(overlaps, dtype=np.float32)
                    + 0.05 * np.asarray(scores, dtype=np.float32)
                )
            )
            if overlaps[best] <= 0:
                return np.asarray(seed_mask, dtype=bool)
        else:
            best = int(np.argmax(scores))
        selected = candidates[best]
        selected &= _bbox_mask(selected.shape, bbox_xyxy)
        return selected
