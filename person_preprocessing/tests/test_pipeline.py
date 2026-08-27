import json
from pathlib import Path

import numpy as np
from PIL import Image

from person_preprocessing import PersonAssetStore
from person_preprocessing import pipeline


def _joints(width, height):
    xy = [
        (0.50, 0.12), (0.46, 0.10), (0.54, 0.10), (0.42, 0.12),
        (0.58, 0.12), (0.35, 0.28), (0.65, 0.28), (0.28, 0.43),
        (0.72, 0.43), (0.23, 0.57), (0.77, 0.57), (0.40, 0.55),
        (0.60, 0.55), (0.38, 0.73), (0.62, 0.73), (0.36, 0.92),
        (0.64, 0.92),
    ]
    return np.asarray([[x * width, y * height, 0.9] for x, y in xy], np.float32)


class StubEstimator:
    calls = []

    def __init__(self, weight, *args):
        self.weight = Path(weight).resolve()

    def predict(self, image):
        self.calls.append(image.size)
        width, height = image.size
        return {
            "bbox": np.asarray([1, 1, width - 1, height - 1], np.float32),
            "confidence": 0.9,
            "keypoints": _joints(width, height),
            "model_path": str(self.weight),
            "person_count": 1,
        }


def test_pose_is_cached_on_final_vit_image(tmp_path, monkeypatch):
    native = tmp_path / "native"
    source = native / "sysu/cam1/0001/0001.png"
    source.parent.mkdir(parents=True)
    Image.new("RGB", (20, 50), "gray").save(source)
    (native / "contract.json").write_text(json.dumps({"scale": 1}))
    manifest = tmp_path / "images.jsonl"
    row = {
        "dataset": "sysu",
        "source": "cam1/0001/0001.jpg",
        "output": "cam1/0001/0001.png",
        "modality": "rgb",
        "width": 20,
        "height": 50,
        "references": [{"index": "train", "label": 1}],
        "aliases": [],
    }
    manifest.write_text(json.dumps(row) + "\n")
    weight = tmp_path / "yolo11x-pose.pt"
    weight.write_bytes(b"fixture")
    config = {
        "input_root": str(native),
        "input_manifest": str(manifest),
        "input_scale": 1,
        "output_root": str(tmp_path / "assets"),
        "size_hw": [64, 32],
        "person_margin": 0.05,
        "blur_radius": 2,
        "detection_confidence": 0.25,
        "pose_imgsz": 640,
        "pose_weight": str(weight),
    }
    StubEstimator.calls = []
    monkeypatch.setattr(pipeline, "PoseEstimator", StubEstimator)
    pipeline.prepare(config, ["sysu"], ["person_fit"], "cpu")
    pipeline.pose(config, ["sysu"], ["person_fit"], "cpu")

    assert StubEstimator.calls == [(20, 50), (32, 64)]
    store = PersonAssetStore(tmp_path / "assets/person_fit", "sysu")
    assert store.image(row["source"]).size == (32, 64)
    pose = store.pose(row["source"])
    assert tuple(pose["size_hw"]) == (64, 32)
    assert pose["keypoints"].shape == (17, 3)
    assert store.pose_contract()["coordinate_space"] == "final_vit_input_pixels_xy"
