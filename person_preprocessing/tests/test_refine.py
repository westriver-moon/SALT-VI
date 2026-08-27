import json
from pathlib import Path

import numpy as np
from PIL import Image

from person_preprocessing import PersonAssetStore
from person_preprocessing import pipeline
from person_preprocessing import refine


class MissingEstimator:
    def __init__(self, *args, **kwargs):
        pass

    def predict(self, image):
        return None


class SecondaryEstimator:
    def __init__(self, weight, *args, **kwargs):
        self.weight = Path(weight).resolve()

    def predict(self, image):
        width, height = image.size
        points = np.zeros((17, 3), np.float32)
        points[:8, :2] = [width / 2, height / 2]
        points[:8, 2] = 0.8
        return {
            "bbox": np.asarray([2, 2, width - 2, height - 2], np.float32),
            "confidence": 0.12,
            "keypoints": points,
            "model_path": str(self.weight),
            "person_count": 1,
        }


def test_refine_promotes_quality_gated_fallback(tmp_path, monkeypatch):
    native = tmp_path / "native"
    image_path = native / "sysu/cam3/0001/0001.png"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (20, 50), "gray").save(image_path)
    (native / "contract.json").write_text(json.dumps({"scale": 1}))
    manifest = tmp_path / "images.jsonl"
    row = {
        "dataset": "sysu",
        "source": "cam3/0001/0001.jpg",
        "output": "cam3/0001/0001.png",
        "modality": "ir",
        "width": 20,
        "height": 50,
        "references": [],
        "aliases": [],
    }
    manifest.write_text(json.dumps(row) + "\n")
    weight = tmp_path / "yolo11x-pose.pt"
    weight.write_bytes(b"fixture")
    v1_config = {
        "input_root": str(native),
        "input_manifest": str(manifest),
        "input_scale": 1,
        "output_root": str(tmp_path / "v1"),
        "size_hw": [64, 32],
        "person_margin": 0.05,
        "blur_radius": 2,
        "detection_confidence": 0.25,
        "no_person_fallback": "full_frame_fit",
        "pose_imgsz": 640,
        "pose_weight": str(weight),
    }
    monkeypatch.setattr(pipeline, "PoseEstimator", MissingEstimator)
    pipeline.prepare(v1_config, ["sysu"], ["person_fit"], "cpu")
    v2_config = {
        "source_asset_root": str(tmp_path / "v1/person_fit"),
        "output_root": str(tmp_path / "v2"),
        "size_hw": [64, 32],
        "person_margin": 0.05,
        "blur_radius": 2,
        "pose_weight": str(weight),
        "secondary_detection_confidence": 0.10,
        "secondary_pose_imgsz": 640,
        "secondary_min_box_area_fraction": 0.08,
        "secondary_max_box_area_fraction": 0.95,
        "secondary_min_box_height_fraction": 0.35,
        "secondary_min_box_aspect_ratio_hw": 0.75,
        "secondary_max_box_center_distance": 0.50,
        "secondary_keypoint_confidence": 0.15,
        "secondary_min_keypoints": 5,
        "secondary_min_torso_keypoints": 1,
    }
    monkeypatch.setattr(refine, "PoseEstimator", SecondaryEstimator)

    summary = refine.refine(v2_config, ["sysu"], "cpu")[0]

    store = PersonAssetStore(tmp_path / "v2", "sysu")
    assert store.image(row["source"]).size == (32, 64)
    record = store.record(row["source"])
    assert record["localization"]["status"] == "secondary_detected_person"
    assert len(store.refinements) == 1
    assert store.refinements[0]["status"] == "secondary_accepted"
    assert summary["counts"] == {"secondary_accepted": 1}
    assert summary["complete_fallback_inventory"] is True
