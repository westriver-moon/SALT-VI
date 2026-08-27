import json

import numpy as np
from PIL import Image

from person_preprocessing import PersonAssetStore
from qwen_imagination.text_annotation.prepared import CachedPoseROIGenerator


def _store(tmp_path):
    root = tmp_path / "person/sysu"
    image = root / "images/cam1/0001/0001.png"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (32, 64), "gray").save(image)
    contract = {"version": 2, "size_hw": [64, 32]}
    (root / "contract.json").write_text(json.dumps(contract))
    row = {
        "source_key": "cam1/0001/0001.jpg",
        "status": "complete",
        "image": "images/cam1/0001/0001.png",
        "aliases": [],
    }
    (root / "images.jsonl").write_text(json.dumps(row) + "\n")
    (root / "pose.contract.json").write_text(
        json.dumps({"version": 2, "images": contract})
    )
    pose = root / "pose/cam1/0001/0001.npz"
    pose.parent.mkdir(parents=True)
    points = np.zeros((17, 3), np.float32)
    points[:, 0] = 16
    points[:, 1] = 32
    points[:, 2] = 0.9
    np.savez_compressed(
        pose,
        keypoints=points,
        bbox=np.asarray([1, 2, 31, 63], np.float32),
        confidence=np.asarray(0.9, np.float32),
        person_count=np.asarray(1, np.int32),
        size_hw=np.asarray([64, 32], np.int32),
    )
    pose.with_suffix(".json").write_text(
        json.dumps({"source_key": row["source_key"], "status": "complete"})
    )
    return PersonAssetStore(tmp_path / "person", "sysu"), row["source_key"]


class Probe:
    def regions(self, image, modality, *, pose_result=None, source_key=None):
        self.pose_result = pose_result
        self.source_key = source_key
        return []


def test_qwen_reads_shared_pose_without_a_pose_model(tmp_path):
    store, source_key = _store(tmp_path)
    probe = Probe()
    adapter = CachedPoseROIGenerator(probe, store)
    adapter.regions(store.image(source_key), "rgb", source_key=source_key)
    assert probe.pose_result["bbox_xyxy"] == (1.0, 2.0, 31.0, 63.0)
    assert np.allclose(
        probe.pose_result["keypoints"]["nose"], (16.0, 32.0, 0.9)
    )
    assert probe.source_key is None
