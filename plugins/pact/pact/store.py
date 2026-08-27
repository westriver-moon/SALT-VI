"""Read PACT-derived anatomy without owning the shared image or pose cache."""

import json
from pathlib import Path

import numpy as np

from person_preprocessing import PersonAssetStore


class PACTArtifactStore:
    def __init__(self, person_assets_root, pact_root, mode, dataset):
        self.person = PersonAssetStore(Path(person_assets_root) / mode, dataset)
        self.root = Path(pact_root) / mode / dataset
        self.contract = json.loads((self.root / "contract.json").read_text())
        if self.contract["person_pose_contract"] != self.person.pose_contract():
            raise ValueError("PACT artifacts are not bound to the current person pose cache")

    def path(self, source_key):
        row = self.person.record(source_key)
        relative = Path(row["image"]).relative_to("images").with_suffix(".npz")
        return self.root / "anatomy" / relative

    def anatomy(self, source_key):
        path = self.path(source_key)
        with np.load(path, allow_pickle=False) as arrays:
            return {key: arrays[key] for key in arrays.files}
