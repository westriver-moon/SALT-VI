"""Read immutable final images and shared model-agnostic person assets."""

import json
from pathlib import Path

import numpy as np
from PIL import Image


class PersonAssetStore:
    """Dataset view over one deterministic image geometry and its pose cache."""

    def __init__(self, root, dataset):
        self.root = Path(root) / dataset
        self.contract = json.loads((self.root / "contract.json").read_text())
        self.size_hw = tuple(self.contract["size_hw"])
        self.refinements = []
        self._image_paths = {}
        if self.contract.get("overlay_type") == "fallback_refinement_v1":
            base = PersonAssetStore(self.contract["base_asset_root"], dataset)
            refinement_manifest = self.root / "refinements.jsonl"
            self.refinements = [
                json.loads(line)
                for line in refinement_manifest.read_text().splitlines()
                if line
            ]
            accepted = {}
            for row in self.refinements:
                if row["status"] == "secondary_accepted":
                    merged = dict(row)
                    merged["overlay_status"] = merged["status"]
                    merged["status"] = "complete"
                    accepted[row["source_key"]] = merged
            self.records = [accepted.get(row["source_key"], row) for row in base.records]
            for row in self.records:
                if row["source_key"] in accepted:
                    path = self.root / row["image"]
                else:
                    path = base.path(row["source_key"])
                for key in [row["source_key"]] + row.get("aliases", []):
                    self._image_paths[key] = path
        else:
            manifest = self.root / "images.jsonl"
            self.records = [
                json.loads(line) for line in manifest.read_text().splitlines() if line
            ]
            for row in self.records:
                path = self.root / row["image"]
                for key in [row["source_key"]] + row.get("aliases", []):
                    self._image_paths[key] = path
        self.by_key = {}
        for row in self.records:
            for key in [row["source_key"]] + row.get("aliases", []):
                if key in self.by_key:
                    raise ValueError("duplicate prepared source key: " + key)
                self.by_key[key] = row

    def record(self, source_key):
        row = self.by_key[str(source_key)]
        if row["status"] != "complete":
            raise ValueError(
                "unavailable prepared image: "
                + str(source_key)
                + " ("
                + row["status"]
                + ")"
            )
        return row

    def path(self, source_key):
        self.record(source_key)
        return self._image_paths[str(source_key)]

    def image(self, source_key):
        with Image.open(self.path(source_key)) as image:
            image = image.convert("RGB")
        if (image.height, image.width) != self.size_hw:
            raise ValueError("prepared image has the wrong dimensions")
        return image

    def pose_contract(self):
        return json.loads((self.root / "pose.contract.json").read_text())

    def _pose_paths(self, source_key):
        row = self.record(source_key)
        relative = Path(row["image"]).relative_to("images")
        return (
            self.root / "pose" / relative.with_suffix(".npz"),
            self.root / "pose" / relative.with_suffix(".json"),
        )

    def pose_record(self, source_key, require_person=True):
        _arrays, metadata = self._pose_paths(source_key)
        row = json.loads(metadata.read_text())
        if row["source_key"] != str(source_key):
            raise ValueError("pose metadata source key mismatch")
        if require_person and row["status"] != "complete":
            raise ValueError(
                "no usable cached pose: " + str(source_key) + " (" + row["status"] + ")"
            )
        return row

    def pose(self, source_key, require_person=True):
        arrays, _metadata = self._pose_paths(source_key)
        self.pose_record(source_key, require_person=require_person)
        with np.load(arrays, allow_pickle=False) as payload:
            result = {key: payload[key] for key in payload.files}
        required = {"keypoints", "bbox", "confidence", "person_count", "size_hw"}
        missing = required - set(result)
        if missing:
            raise ValueError("pose cache is missing: " + ", ".join(sorted(missing)))
        return result


class PersonVisualSource:
    def __init__(self, store, source_keys):
        self.store, self.source_keys = store, list(source_keys)
        for key in self.source_keys:
            store.record(key)

    def __len__(self):
        return len(self.source_keys)

    def __getitem__(self, index):
        return np.array(self.store.image(self.source_keys[int(index)]))

    def sample(self, index):
        return self[int(index)], None


# Transitional names for loaders migrated from the earlier in-core prototype.
PreparedImageStore = PersonAssetStore
PreparedVisualSource = PersonVisualSource
