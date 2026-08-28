"""Read the single canonical person-fit asset release."""

import json
from pathlib import Path

import numpy as np
from PIL import Image


class PersonAssetStore:
    """Dataset view over one immutable person-fit image geometry."""

    def __init__(self, root, dataset):
        self.root = Path(root) / dataset
        self.contract = json.loads((self.root / "contract.json").read_text())
        self.size_hw = tuple(self.contract["size_hw"])
        manifest = self.root / "images.jsonl"
        self.records = [
            json.loads(line) for line in manifest.read_text().splitlines() if line
        ]
        self.by_key = {}
        self._image_paths = {}
        for row in self.records:
            path = self.root / row["image"]
            for key in [row["source_key"]] + row.get("aliases", []):
                if key in self.by_key:
                    raise ValueError("duplicate prepared source key: " + key)
                self.by_key[key] = row
                self._image_paths[key] = path

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


PreparedImageStore = PersonAssetStore
PreparedVisualSource = PersonVisualSource
