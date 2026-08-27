import json

import numpy as np
from PIL import Image

from pact.pipeline import build, verify
from pact.store import PACTArtifactStore


def _person_store(tmp_path):
    root = tmp_path / "person/person_fit/sysu"
    image = root / "images/cam1/0001/0001.png"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (32, 64), "gray").save(image)
    contract = {"version": 2, "size_hw": [64, 32], "mode": "person_fit"}
    (root / "contract.json").write_text(json.dumps(contract))
    row = {
        "source_key": "cam1/0001/0001.jpg",
        "status": "complete",
        "image": "images/cam1/0001/0001.png",
        "geometry": {"foreground_box": [0, 0, 32, 64]},
        "modality": "rgb",
        "aliases": [],
    }
    (root / "images.jsonl").write_text(json.dumps(row) + "\n")
    pose_contract = {"version": 2, "images": contract, "format": "coco17_raw_pose_v1"}
    (root / "pose.contract.json").write_text(json.dumps(pose_contract))
    pose = root / "pose/cam1/0001/0001.npz"
    pose.parent.mkdir(parents=True)
    points = np.zeros((17, 3), np.float32)
    points[:, 0] = np.linspace(10, 22, 17)
    points[:, 1] = np.linspace(6, 58, 17)
    points[:, 2] = 0.9
    np.savez_compressed(
        pose,
        keypoints=points,
        bbox=np.asarray([1, 1, 31, 63], np.float32),
        confidence=np.asarray(0.9, np.float32),
        person_count=np.asarray(1, np.int32),
        size_hw=np.asarray([64, 32], np.int32),
    )
    metadata = {
        "source_key": row["source_key"],
        "status": "complete",
        "pose": "pose/cam1/0001/0001.npz",
    }
    pose.with_suffix(".json").write_text(json.dumps(metadata))
    return tmp_path / "person", row["source_key"]


def test_pact_derives_only_plugin_specific_anatomy(tmp_path):
    person_root, source_key = _person_store(tmp_path)
    config = {
        "person_assets_root": str(person_root),
        "output_root": str(tmp_path / "pact"),
        "keypoint_confidence": 0.3,
        "patch_hw": [16, 16],
        "stride_hw": [12, 12],
    }
    build(config, ["sysu"], ["person_fit"])
    assert verify(config, ["sysu"], ["person_fit"])[0]["checked"] == 1
    store = PACTArtifactStore(person_root, tmp_path / "pact", "person_fit", "sysu")
    arrays = store.anatomy(source_key)
    assert arrays["pixel_masks"].shape == (4, 64, 32)
    assert arrays["token_masks"].shape == (4, 5, 2)
