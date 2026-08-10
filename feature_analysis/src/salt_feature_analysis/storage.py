from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np


@dataclass
class FeatureArtifact:
    features: np.ndarray
    labels: np.ndarray
    cameras: np.ndarray
    sample_ids: np.ndarray
    metadata: Dict[str, Any]


def sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def save_feature_artifact(path: Path, artifact: FeatureArtifact, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite feature artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                features=np.asarray(artifact.features, dtype=np.float32),
                labels=np.asarray(artifact.labels, dtype=np.int64),
                cameras=np.asarray(artifact.cameras, dtype=np.int64),
                sample_ids=np.asarray(artifact.sample_ids, dtype=np.str_),
            )
        os.replace(str(temporary), str(path))
        write_json(path.with_suffix(".meta.json"), artifact.metadata)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_feature_artifact(path: Path) -> FeatureArtifact:
    with np.load(str(path), allow_pickle=False) as payload:
        artifact = FeatureArtifact(
            features=np.asarray(payload["features"], dtype=np.float32),
            labels=np.asarray(payload["labels"], dtype=np.int64),
            cameras=np.asarray(payload["cameras"], dtype=np.int64),
            sample_ids=np.asarray(payload["sample_ids"], dtype=np.str_),
            metadata={},
        )
    metadata_path = path.with_suffix(".meta.json")
    if metadata_path.is_file():
        with metadata_path.open("r", encoding="utf-8") as handle:
            artifact.metadata = json.load(handle)
    _validate_artifact(artifact, path)
    return artifact


def _validate_artifact(artifact: FeatureArtifact, path: Path) -> None:
    if artifact.features.ndim != 2:
        raise ValueError(f"Feature matrix must be 2D in {path}")
    count = artifact.features.shape[0]
    for name, values in (
        ("labels", artifact.labels),
        ("cameras", artifact.cameras),
        ("sample_ids", artifact.sample_ids),
    ):
        if values.ndim != 1 or len(values) != count:
            raise ValueError(f"{name} length mismatch in {path}")
    if len(set(artifact.sample_ids.tolist())) != count:
        raise ValueError(f"sample_ids must be unique inside one artifact: {path}")


def artifact_key(model_id: str, split_tag: str, representation: str) -> str:
    return f"{model_id}::{split_tag}::{representation}"


class ArtifactLayout:
    def __init__(self, output_root: str, run_id: str):
        root = Path(output_root)
        self.feature_root = root / "features" / run_id
        self.table_root = root / "tables" / run_id
        self.figure_root = root / "figures" / run_id
        self.report_root = root / "reports" / run_id
        self.manifest_root = root / "manifests" / run_id

    def create(self) -> None:
        for path in (
            self.feature_root,
            self.table_root,
            self.figure_root,
            self.report_root,
            self.manifest_root,
        ):
            path.mkdir(parents=True, exist_ok=True)

