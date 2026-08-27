import json

import numpy as np
from PIL import Image

from person_preprocessing import PersonAssetStore
from person_preprocessing import refine_v3


class StubCascade:
    def __init__(self, *args, **kwargs):
        pass

    def predict(self, image):
        width, height = image.size
        points = np.zeros((17, 3), np.float32)
        points[:3, 2] = 0.1
        prediction = {
            "bbox": np.asarray(
                [width * 0.45, height * 0.4, width * 0.55, height * 0.6]
            ),
            "confidence": 0.002,
            "keypoints": points,
            "model_path": "stub.pt",
            "person_count": 1,
        }
        stage = {
            "name": "autocontrast_1280",
            "quality_tier": "contrast_recovered",
            "min_render_width_fraction": 0.75,
            "min_render_height_fraction": 1.0,
            "anchor_width_fraction": 0.75,
            "anchor_height_fraction": 1.0,
        }
        return prediction, stage, [{"stage": stage["name"], "detected": True}]


class StubDetector:
    def __init__(self, *args, **kwargs):
        pass

    def validate(self, image, pose_bbox):
        return {
            "detected": False,
            "confidence": 0.0,
            "pose_iou": 0.0,
            "model_path": "detector.pt",
        }


def test_stabilize_bbox_enforces_safe_envelope():
    stage = {
        "min_render_width_fraction": 0.75,
        "min_render_height_fraction": 1.0,
        "anchor_width_fraction": 0.75,
        "anchor_height_fraction": 1.0,
    }
    box = refine_v3.stabilize_bbox([95, 90, 100, 100], (100, 100), stage)
    assert np.allclose(box[[1, 3]], [0, 100])
    assert box[2] - box[0] >= 75
    assert box[0] <= 12.5 and box[2] >= 87.5


def test_detector_consensus_requires_confidence_and_overlap():
    config = {
        "validation_detector_min_confidence": 0.1,
        "validation_detector_min_pose_iou": 0.5,
    }
    accepted, reasons = refine_v3.detector_consensus(
        {"detected": True, "confidence": 0.2, "pose_iou": 0.7}, config
    )
    assert accepted is True and reasons == []
    accepted, reasons = refine_v3.detector_consensus(
        {"detected": True, "confidence": 0.05, "pose_iou": 0.2}, config
    )
    assert accepted is False
    assert reasons == [
        "detector_confidence_too_low",
        "pose_detector_disagreement",
    ]


def test_v3_overlay_promotes_remaining_fallback(tmp_path, monkeypatch):
    native = tmp_path / "native.png"
    Image.new("RGB", (40, 80), "gray").save(native)
    base = tmp_path / "base" / "sysu"
    (base / "images").mkdir(parents=True)
    Image.new("RGB", (32, 64), "gray").save(base / "images/sample.png")
    (base / "contract.json").write_text(
        json.dumps({"version": 2, "size_hw": [64, 32]}), encoding="utf-8"
    )
    row = {
        "dataset": "sysu",
        "source_key": "sample.png",
        "identity": "1",
        "camera": "cam3",
        "modality": "ir",
        "references": [],
        "aliases": [],
        "native_image": str(native),
        "image": "images/sample.png",
        "status": "complete",
        "localization": {"status": "fallback_full_frame"},
    }
    (base / "images.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    weight = tmp_path / "weight.pt"
    weight.write_bytes(b"stub")
    config = {
        "source_asset_root": str(tmp_path / "base"),
        "output_root": str(tmp_path / "v3"),
        "size_hw": [64, 32],
        "person_margin": 0.05,
        "blur_radius": 2.0,
        "pose_weight": str(weight),
        "keypoint_quality_threshold": 0.05,
        "trusted_min_detection_confidence": 0.01,
        "trusted_min_source_aspect_ratio_hw": 0.75,
        "trusted_min_box_area_fraction": 0.08,
        "trusted_max_box_area_fraction": 0.95,
        "trusted_min_box_height_fraction": 0.35,
        "trusted_min_box_aspect_ratio_hw": 0.75,
        "trusted_max_box_center_distance": 0.5,
        "trusted_min_keypoints": 5,
        "trusted_min_torso_keypoints": 1,
        "validation_detector_weight": str(weight),
        "validation_detector_inference_confidence": 0.05,
        "validation_detector_imgsz": 960,
        "validation_detector_min_confidence": 0.1,
        "validation_detector_min_pose_iou": 0.5,
        "cascade": [],
    }
    monkeypatch.setattr(refine_v3, "PoseCascade", StubCascade)
    monkeypatch.setattr(refine_v3, "PersonDetector", StubDetector)

    summary = refine_v3.refine(config, ["sysu"], "cpu")[0]
    store = PersonAssetStore(tmp_path / "v3", "sysu")
    record = store.record("sample.png")

    assert summary["complete_detection_coverage"] is True
    assert summary["counts"] == {"tertiary_accepted": 1}
    assert record["overlay_status"] == "tertiary_accepted"
    assert record["localization"]["stage"] == "autocontrast_1280"
    assert record["localization"]["render_policy"] == "full_frame_quality_guard"
    assert record["localization"]["render_bbox"] == [0.0, 0.0, 40.0, 80.0]
    assert store.image("sample.png").size == (32, 64)
