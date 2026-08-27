"""Recover every remaining v2 fallback with an auditable pose cascade."""

import argparse
import collections
import json
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import yaml

from .geometry import render
from .models import PoseEstimator
from .pipeline import bind_contract, file_identity, pose_weight, write_json, write_jsonl
from .store import PersonAssetStore


def _expand_interval(low, high, limit, minimum_fraction):
    span = max(float(high) - float(low), float(limit) * float(minimum_fraction))
    center = (float(low) + float(high)) / 2
    low, high = center - span / 2, center + span / 2
    if low < 0:
        high -= low
        low = 0.0
    if high > limit:
        low -= high - limit
        high = float(limit)
    return max(0.0, low), min(float(limit), high)


def stabilize_bbox(bbox, image_size, stage):
    """Expand weak boxes toward a central safety envelope before rendering."""
    width, height = map(float, image_size)
    x1, y1, x2, y2 = map(float, bbox)
    anchor_width = width * float(stage["anchor_width_fraction"])
    anchor_height = height * float(stage["anchor_height_fraction"])
    x1 = min(x1, (width - anchor_width) / 2)
    x2 = max(x2, (width + anchor_width) / 2)
    y1 = min(y1, (height - anchor_height) / 2)
    y2 = max(y2, (height + anchor_height) / 2)
    x1, x2 = _expand_interval(
        x1, x2, width, stage["min_render_width_fraction"]
    )
    y1, y2 = _expand_interval(
        y1, y2, height, stage["min_render_height_fraction"]
    )
    return np.asarray([x1, y1, x2, y2], dtype=np.float32)


def candidate_quality(prediction, image_size, keypoint_threshold):
    width, height = map(float, image_size)
    x1, y1, x2, y2 = map(float, prediction["bbox"])
    box_width, box_height = x2 - x1, y2 - y1
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    confidence = np.asarray(prediction["keypoints"], dtype=np.float32)[:, 2]
    threshold = float(keypoint_threshold)
    return {
        "box_confidence": float(prediction["confidence"]),
        "box_area_fraction": float(box_width * box_height / (width * height)),
        "box_height_fraction": float(box_height / height),
        "box_aspect_ratio_hw": float(box_height / max(box_width, 1e-12)),
        "box_center_distance": float(
            math.hypot(
                (center_x - width / 2) / width,
                (center_y - height / 2) / height,
            )
        ),
        "keypoint_confidence_threshold": threshold,
        "keypoint_count": int(np.count_nonzero(confidence >= threshold)),
        "torso_keypoint_count": int(
            np.count_nonzero(confidence[[5, 6, 11, 12]] >= threshold)
        ),
    }


def trusted_crop_decision(prediction, image_size, config):
    metrics = candidate_quality(
        prediction, image_size, config["keypoint_quality_threshold"]
    )
    width, height = map(float, image_size)
    checks = {
        "detector_confidence_too_low": metrics["box_confidence"]
        < float(config["trusted_min_detection_confidence"]),
        "source_too_wide": height / width
        < float(config["trusted_min_source_aspect_ratio_hw"]),
        "box_area_too_small": metrics["box_area_fraction"]
        < float(config["trusted_min_box_area_fraction"]),
        "box_area_too_large": metrics["box_area_fraction"]
        > float(config["trusted_max_box_area_fraction"]),
        "box_too_short": metrics["box_height_fraction"]
        < float(config["trusted_min_box_height_fraction"]),
        "box_too_wide": metrics["box_aspect_ratio_hw"]
        < float(config["trusted_min_box_aspect_ratio_hw"]),
        "box_too_off_center": metrics["box_center_distance"]
        > float(config["trusted_max_box_center_distance"]),
        "too_few_keypoints": metrics["keypoint_count"]
        < int(config["trusted_min_keypoints"]),
        "no_torso_keypoint": metrics["torso_keypoint_count"]
        < int(config["trusted_min_torso_keypoints"]),
    }
    reasons = [name for name, failed in checks.items() if failed]
    return not reasons, metrics, reasons


def _box_iou(box, boxes):
    box = np.asarray(box, dtype=np.float32)
    boxes = np.asarray(boxes, dtype=np.float32)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return intersection / np.maximum(box_area + areas - intersection, 1e-12)


class PersonDetector:
    """Independent detection-head validation for a pose candidate."""

    def __init__(self, config, device):
        self.weight = Path(config["validation_detector_weight"]).resolve()
        if not self.weight.is_file():
            raise FileNotFoundError(self.weight)
        self.device = device
        self.confidence = float(config["validation_detector_inference_confidence"])
        self.imgsz = int(config["validation_detector_imgsz"])
        self.model = None

    def validate(self, image, pose_bbox):
        from ultralytics import YOLO

        if self.model is None:
            self.model = YOLO(str(self.weight))
            if self.model.task != "detect":
                raise ValueError("validation checkpoint is not a detection model")
        result = self.model.predict(
            image,
            device=self.device,
            imgsz=self.imgsz,
            conf=self.confidence,
            classes=[0],
            verbose=False,
            save=False,
            half=False,
            rect=False,
        )[0]
        if not len(result.boxes):
            return {
                "detected": False,
                "confidence": 0.0,
                "pose_iou": 0.0,
                "model_path": str(self.weight),
            }
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        ious = _box_iou(pose_bbox, boxes)
        index = int(np.argmax(ious))
        return {
            "detected": True,
            "bbox": boxes[index].astype(np.float32).tolist(),
            "confidence": float(confidences[index]),
            "pose_iou": float(ious[index]),
            "person_count": int(len(boxes)),
            "model_path": str(self.weight),
        }


def detector_consensus(validation, config):
    checks = {
        "detector_no_person": not validation["detected"],
        "detector_confidence_too_low": validation["confidence"]
        < float(config["validation_detector_min_confidence"]),
        "pose_detector_disagreement": validation["pose_iou"]
        < float(config["validation_detector_min_pose_iou"]),
    }
    reasons = [name for name, failed in checks.items() if failed]
    return not reasons, reasons


def _transform(image, name):
    if name == "identity":
        return image
    if name == "autocontrast":
        return ImageOps.autocontrast(image)
    raise ValueError("unknown v3 detector transform: " + name)


class PoseCascade:
    def __init__(self, config, device):
        self.stages = []
        default_weight = pose_weight(config)
        for stage in config["cascade"]:
            stage = dict(stage)
            estimator = PoseEstimator(
                stage.get("weight", default_weight),
                device,
                stage["confidence"],
                stage["imgsz"],
                rect=stage.get("rect", False),
            )
            self.stages.append((stage, estimator))

    def predict(self, image):
        attempts = []
        for stage, estimator in self.stages:
            prediction = estimator.predict(_transform(image, stage["transform"]))
            attempts.append(
                {
                    "stage": stage["name"],
                    "confidence_threshold": float(stage["confidence"]),
                    "imgsz": int(stage["imgsz"]),
                    "rect": bool(stage.get("rect", False)),
                    "transform": stage["transform"],
                    "detected": prediction is not None,
                }
            )
            if prediction is not None:
                return prediction, stage, attempts
        return None, None, attempts


def refinement_contract(config, source_store, source_base, dataset):
    return {
        "version": 7,
        "overlay_type": "fallback_refinement_v2",
        "dataset": dataset,
        "size_hw": list(source_store.size_hw),
        "base_asset_root": str(source_base.resolve()),
        "base_contract": source_store.contract,
        "refinement": {
            "algorithm": "progressive_pose_cascade_dual_model_guard_v3",
            "weight": file_identity(pose_weight(config)),
            "keypoint_quality_threshold": float(
                config["keypoint_quality_threshold"]
            ),
            "trusted_crop_gate": {
                key: config[key]
                for key in config
                if key.startswith("trusted_")
            },
            "validation_detector": {
                "weight": file_identity(config["validation_detector_weight"]),
                "inference_confidence": float(
                    config["validation_detector_inference_confidence"]
                ),
                "imgsz": int(config["validation_detector_imgsz"]),
                "min_confidence": float(
                    config["validation_detector_min_confidence"]
                ),
                "min_pose_iou": float(config["validation_detector_min_pose_iou"]),
            },
            "cascade": config["cascade"],
        },
    }


def refine(config, datasets, device):
    summaries = []
    source_base = Path(config["source_asset_root"])
    output_base = Path(config["output_root"])
    cascade = PoseCascade(config, device)
    detector = PersonDetector(config, device)
    for dataset in datasets:
        source_store = PersonAssetStore(source_base, dataset)
        fallbacks = [
            row
            for row in source_store.records
            if row.get("localization", {}).get("status") == "fallback_full_frame"
        ]
        root = output_base / dataset
        contract = refinement_contract(config, source_store, source_base, dataset)
        bind_contract(root / "contract.json", contract)
        for parent in fallbacks:
            relative = Path(parent["image"]).relative_to("images")
            metadata = root / "records" / relative.with_suffix(".json")
            if metadata.exists():
                record = json.loads(metadata.read_text(encoding="utf-8"))
                if record["status"] == "tertiary_accepted":
                    image_path = root / record["image"]
                    if not image_path.is_file():
                        raise FileNotFoundError(image_path)
                continue
            native_path = Path(parent["native_image"])
            with Image.open(native_path) as image:
                image = image.convert("RGB")
            prediction, stage, attempts = cascade.predict(image)
            record = {
                key: parent[key]
                for key in (
                    "dataset",
                    "source_key",
                    "identity",
                    "camera",
                    "modality",
                    "references",
                    "aliases",
                    "native_image",
                )
            }
            record.update(
                accepted=prediction is not None,
                status=(
                    "tertiary_accepted"
                    if prediction is not None
                    else "tertiary_unresolved"
                ),
                parent_image=str(source_store.path(parent["source_key"])),
                attempts=attempts,
            )
            if prediction is not None:
                raw_bbox = np.asarray(prediction["bbox"], dtype=np.float32)
                trusted_pose, quality, guard_reasons = trusted_crop_decision(
                    prediction, image.size, config
                )
                validation = detector.validate(image, raw_bbox)
                consensus, validation_reasons = detector_consensus(validation, config)
                guard_reasons.extend(validation_reasons)
                trusted_crop = trusted_pose and consensus
                render_bbox = (
                    stabilize_bbox(raw_bbox, image.size, stage)
                    if trusted_crop
                    else np.asarray(
                        [0.0, 0.0, float(image.width), float(image.height)],
                        dtype=np.float32,
                    )
                )
                output, geometry = render(
                    image,
                    "person_fit",
                    config["size_hw"],
                    render_bbox,
                    config["person_margin"],
                    config["blur_radius"],
                )
                image_path = root / "images" / relative
                image_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = image_path.with_suffix(".png.tmp")
                output.save(temporary, format="PNG", compress_level=3)
                os.replace(temporary, image_path)
                record.update(
                    image="images/" + relative.as_posix(),
                    geometry=geometry,
                    localization={
                        "status": "tertiary_detected_person",
                        "stage": stage["name"],
                        "quality_tier": (
                            stage["quality_tier"]
                            if trusted_crop
                            else "full_frame_quality_guard"
                        ),
                        "render_policy": (
                            "trusted_detection_crop"
                            if trusted_crop
                            else "full_frame_quality_guard"
                        ),
                        "guard_reasons": guard_reasons,
                        "raw_bbox": raw_bbox.tolist(),
                        "render_bbox": render_bbox.tolist(),
                        "confidence": float(prediction["confidence"]),
                        "person_count": int(prediction["person_count"]),
                        "model_path": prediction["model_path"],
                        "candidate_quality": quality,
                        "validation_detector": validation,
                    },
                )
            write_json(metadata, record)
        records = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted((root / "records").rglob("*.json"))
        ]
        write_jsonl(root / "refinements.jsonl", records)
        statuses = collections.Counter(row["status"] for row in records)
        stages = collections.Counter(
            row.get("localization", {}).get("stage", "unresolved") for row in records
        )
        modality_stages = collections.Counter(
            (
                row.get("modality", "unknown"),
                row.get("localization", {}).get("stage", "unresolved"),
            )
            for row in records
        )
        render_policies = collections.Counter(
            row.get("localization", {}).get("render_policy", "unresolved")
            for row in records
        )
        summary = {
            "dataset": dataset,
            "base_inventory_total": len(source_store.records),
            "fallback_total": len(fallbacks),
            "attempt_records": len(records),
            "counts": dict(statuses),
            "stage_counts": dict(stages),
            "render_policy_counts": dict(render_policies),
            "modality_stage_counts": {
                modality + ":" + stage: count
                for (modality, stage), count in sorted(modality_stages.items())
            },
            "complete_fallback_inventory": len(records) == len(fallbacks),
            "complete_detection_coverage": (
                statuses.get("tertiary_accepted", 0) == len(fallbacks)
            ),
        }
        write_json(root / "refinements.summary.json", summary)
        summaries.append(summary)
    return summaries


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("sysu", "regdb", "llcm"),
        default=["sysu", "regdb", "llcm"],
    )
    parser.add_argument("--device", default="0")
    parser.add_argument("--output-root")
    args = parser.parse_args(argv)
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.output_root:
        config["output_root"] = args.output_root
    result = refine(config, args.datasets, args.device)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return int(
        not all(
            row["complete_fallback_inventory"]
            and row["complete_detection_coverage"]
            for row in result
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
