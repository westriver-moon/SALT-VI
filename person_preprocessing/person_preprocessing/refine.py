"""Create a sparse v2 overlay by refining only v1 full-frame fallbacks."""

import argparse
import collections
import json
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image
import yaml

from .geometry import render
from .models import PoseEstimator
from .pipeline import bind_contract, file_identity, pose_weight, write_json, write_jsonl
from .store import PersonAssetStore


def quality_gate(prediction, image_size, config):
    """Return an auditable decision for one low-confidence pose candidate."""
    if prediction is None:
        return False, {"reasons": ["no_candidate"]}
    width, height = map(float, image_size)
    x1, y1, x2, y2 = map(float, prediction["bbox"])
    box_width, box_height = x2 - x1, y2 - y1
    area_fraction = box_width * box_height / (width * height)
    height_fraction = box_height / height
    aspect_ratio = box_height / max(box_width, 1e-12)
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    center_distance = math.hypot(
        (center_x - width / 2) / width,
        (center_y - height / 2) / height,
    )
    keypoint_confidence = np.asarray(prediction["keypoints"], dtype=np.float32)[:, 2]
    threshold = float(config["secondary_keypoint_confidence"])
    keypoint_count = int(np.count_nonzero(keypoint_confidence >= threshold))
    torso_count = int(
        np.count_nonzero(keypoint_confidence[[5, 6, 11, 12]] >= threshold)
    )
    metrics = {
        "box_confidence": float(prediction["confidence"]),
        "box_area_fraction": float(area_fraction),
        "box_height_fraction": float(height_fraction),
        "box_aspect_ratio_hw": float(aspect_ratio),
        "box_center_distance": float(center_distance),
        "keypoint_confidence_threshold": threshold,
        "keypoint_count": keypoint_count,
        "torso_keypoint_count": torso_count,
    }
    checks = {
        "box_area_too_small": area_fraction
        < float(config["secondary_min_box_area_fraction"]),
        "box_area_too_large": area_fraction
        > float(config["secondary_max_box_area_fraction"]),
        "box_too_short": height_fraction
        < float(config["secondary_min_box_height_fraction"]),
        "box_too_wide": aspect_ratio
        < float(config["secondary_min_box_aspect_ratio_hw"]),
        "box_too_off_center": center_distance
        > float(config["secondary_max_box_center_distance"]),
        "too_few_keypoints": keypoint_count
        < int(config["secondary_min_keypoints"]),
        "no_torso_keypoint": torso_count
        < int(config["secondary_min_torso_keypoints"]),
    }
    reasons = [name for name, failed in checks.items() if failed]
    metrics["reasons"] = reasons
    return not reasons, metrics


def refinement_contract(config, source_store, source_base, dataset):
    weight = pose_weight(config)
    return {
        "version": 4,
        "overlay_type": "fallback_refinement_v1",
        "dataset": dataset,
        "size_hw": list(source_store.size_hw),
        "base_asset_root": str(source_base.resolve()),
        "base_contract": source_store.contract,
        "refinement": {
            "algorithm": "square_pose_second_pass_quality_gate_v1",
            "weight": file_identity(weight, include_sha256=True),
            "imgsz": int(config["secondary_pose_imgsz"]),
            "confidence": float(config["secondary_detection_confidence"]),
            "rect": False,
            "min_box_area_fraction": float(
                config["secondary_min_box_area_fraction"]
            ),
            "max_box_area_fraction": float(
                config["secondary_max_box_area_fraction"]
            ),
            "min_box_height_fraction": float(
                config["secondary_min_box_height_fraction"]
            ),
            "min_box_aspect_ratio_hw": float(
                config["secondary_min_box_aspect_ratio_hw"]
            ),
            "max_box_center_distance": float(
                config["secondary_max_box_center_distance"]
            ),
            "keypoint_confidence": float(
                config["secondary_keypoint_confidence"]
            ),
            "min_keypoints": int(config["secondary_min_keypoints"]),
            "min_torso_keypoints": int(
                config["secondary_min_torso_keypoints"]
            ),
        },
    }


def refine(config, datasets, device):
    summaries = []
    source_base = Path(config["source_asset_root"])
    output_base = Path(config["output_root"])
    estimator = PoseEstimator(
        pose_weight(config),
        device,
        config["secondary_detection_confidence"],
        config["secondary_pose_imgsz"],
        rect=False,
    )
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
                if record["status"] == "secondary_accepted":
                    image_path = root / record["image"]
                    if not image_path.is_file():
                        raise FileNotFoundError(image_path)
                continue
            native_path = Path(parent["native_image"])
            with Image.open(native_path) as image:
                image = image.convert("RGB")
            prediction = estimator.predict(image)
            accepted, quality = quality_gate(prediction, image.size, config)
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
                status="secondary_accepted" if accepted else "secondary_rejected",
                parent_image=str(source_store.root / parent["image"]),
                attempt={
                    "rect": False,
                    "confidence_threshold": float(
                        config["secondary_detection_confidence"]
                    ),
                    "quality": quality,
                },
            )
            if prediction:
                record["attempt"].update(
                    bbox=prediction["bbox"].tolist(),
                    confidence=float(prediction["confidence"]),
                    person_count=int(prediction["person_count"]),
                )
            if accepted:
                output, geometry = render(
                    image,
                    "person_fit",
                    config["size_hw"],
                    prediction["bbox"],
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
                        "status": "secondary_detected_person",
                        "bbox": prediction["bbox"].tolist(),
                        "confidence": float(prediction["confidence"]),
                        "person_count": int(prediction["person_count"]),
                        "model_path": prediction["model_path"],
                        "quality": quality,
                    },
                )
            write_json(metadata, record)
        records = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted((root / "records").rglob("*.json"))
        ]
        write_jsonl(root / "refinements.jsonl", records)
        statuses = collections.Counter(row["status"] for row in records)
        rejection_reasons = collections.Counter(
            reason
            for row in records
            if row["status"] == "secondary_rejected"
            for reason in row["attempt"]["quality"]["reasons"]
        )
        summary = {
            "dataset": dataset,
            "base_inventory_total": len(source_store.records),
            "fallback_total": len(fallbacks),
            "attempt_records": len(records),
            "counts": dict(statuses),
            "secondary_rejection_reasons": dict(rejection_reasons),
            "complete_fallback_inventory": len(records) == len(fallbacks),
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
    return int(not all(row["complete_fallback_inventory"] for row in result))


if __name__ == "__main__":
    raise SystemExit(main())
