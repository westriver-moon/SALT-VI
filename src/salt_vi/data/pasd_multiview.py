"""Manifest-backed SYSU PASD views without materializing large NPY arrays."""

from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from .sysu_sources import load_train_source_records

SUPPORTED_BACKENDS = ("array", "pasd_multiview")
SUPPORTED_SAMPLING = ("independent", "paired")


def normalize_backend(value):
    backend = str(value or "array").lower()
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(f"Unsupported SYSU SR backend {backend!r}; expected {SUPPORTED_BACKENDS}")
    return backend


def normalize_sampling(value):
    sampling = str(value or "independent").lower()
    if sampling not in SUPPORTED_SAMPLING:
        raise ValueError(
            f"Unsupported SYSU multiview sampling {sampling!r}; expected {SUPPORTED_SAMPLING}"
        )
    return sampling


def collect_train_sources(data_root: str | Path, modality: str) -> list[str]:
    return [record.source_key for record in load_train_source_records(data_root, modality)]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@lru_cache(maxsize=None)
def _validated_view(path: str, expected_sha256: str) -> Path:
    output = Path(path)
    if not output.is_file():
        raise FileNotFoundError(f"missing PASD view: {output}")
    if _sha256(output) != expected_sha256:
        raise ValueError(f"PASD view checksum mismatch: {output}")
    return output


class PASDMultiviewIndex:
    def __init__(self, manifest_path: str | Path, data_root: str | Path, output_root: str | Path, views: int):
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.data_root = Path(data_root).expanduser().resolve()
        self.output_root = Path(output_root).expanduser().resolve()
        self.views = int(views)
        if self.views not in (0, 1, 5):
            raise ValueError(
                f"SYSU PASD contract requires dynamic, one, or five views, got {self.views}"
            )
        summary = json.loads(
            self.manifest_path.with_suffix(".json").read_text(encoding="utf-8")
        )
        if not summary.get("complete"):
            raise ValueError("PASD dataset manifest is not complete")
        if _sha256(self.manifest_path) != summary["manifest_jsonl_sha256"]:
            raise ValueError("PASD dataset manifest checksum mismatch")
        if int(summary["views_per_source"]) != self.views:
            raise ValueError("PASD manifest view count does not match the training config")

        records = {}
        with self.manifest_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                view = json.loads(line)
                key = str(view["source_key"]).replace("\\", "/").lstrip("/")
                records.setdefault(key, {"views": []})["views"].append(view)
        for key, record in records.items():
            record["views"].sort(key=lambda view: int(view["view_index"]))
            indices = [int(view["view_index"]) for view in record["views"]]
            if self.views and len(indices) != self.views:
                raise ValueError(f"PASD source {key} does not have {self.views} views")
            if not indices or indices != list(range(len(indices))):
                raise ValueError(f"PASD source {key} has invalid view indices {indices}")
            weights = [
                float(view.get("hypothesis_weight", 1.0 / len(indices)))
                for view in record["views"]
            ]
            if any(not np.isfinite(weight) or weight <= 0 for weight in weights):
                raise ValueError(f"PASD source {key} has invalid hypothesis weights")
            if not np.isclose(sum(weights), 1.0, rtol=0.0, atol=1e-6):
                raise ValueError(f"PASD source {key} hypothesis weights do not sum to one")
            for view, weight in zip(record["views"], weights):
                view["hypothesis_weight"] = weight
        if len(records) != int(summary["source_count"]):
            raise ValueError("PASD final manifest source count mismatch")
        if sum(len(record["views"]) for record in records.values()) != int(
            summary["view_count"]
        ):
            raise ValueError("PASD final manifest view count mismatch")
        self.records = records

    def key_for_path(self, image_path: str | Path) -> str:
        image_path = Path(image_path).expanduser().resolve()
        try:
            return str(image_path.relative_to(self.data_root)).replace(os.sep, "/")
        except ValueError as error:
            raise ValueError(f"SYSU image is outside data root: {image_path}") from error

    def record(self, source_key: str) -> dict:
        key = source_key.replace("\\", "/").lstrip("/")
        try:
            return self.records[key]
        except KeyError as error:
            raise KeyError(f"PASD manifest has no source {key}") from error

    def image_path(self, source_key: str, view_index: int) -> Path:
        record = self.record(source_key)
        view = record["views"][int(view_index)]
        path = (self.output_root / view["output"]).resolve()
        path.relative_to(self.output_root)
        return _validated_view(str(path), view["output_sha256"])

    def caption(self, source_key: str, view_index: int) -> str:
        return str(self.record(source_key)["views"][int(view_index)]["caption"])

    def weights(self, source_key: str) -> list[float]:
        return [
            float(view["hypothesis_weight"])
            for view in self.record(source_key)["views"]
        ]


@lru_cache(maxsize=8)
def load_index(manifest_path: str, data_root: str, output_root: str, views: int):
    return PASDMultiviewIndex(manifest_path, data_root, output_root, views)


class PASDTrainViewStore:
    def __init__(
        self,
        data_root: str | Path,
        output_root: str | Path,
        manifest_path: str | Path,
        modality: str,
        labels: np.ndarray,
        views: int = 5,
    ):
        self.modality = str(modality).lower()
        if self.modality not in ("rgb", "ir"):
            raise ValueError(f"Unsupported SYSU modality: {modality}")
        self.index = load_index(
            str(Path(manifest_path).expanduser().resolve()),
            str(Path(data_root).expanduser().resolve()),
            str(Path(output_root).expanduser().resolve()),
            int(views),
        )
        source_records = load_train_source_records(data_root, self.modality)
        self.sources = [record.source_key for record in source_records]
        if len(source_records) != len(labels):
            raise ValueError(
                f"SYSU {self.modality} source count {len(source_records)} "
                f"does not match label count {len(labels)}"
            )
        mismatches = [
            (record.index, record.source_key, record.label, int(labels[record.index]))
            for record in source_records
            if record.label != int(labels[record.index])
        ]
        if mismatches:
            raise ValueError(
                f"SYSU {self.modality} source/label order mismatch: {mismatches[:3]}"
            )
        missing = [source for source in self.sources if source not in self.index.records]
        if missing:
            raise KeyError(f"PASD manifest misses {len(missing)} {self.modality} train sources: {missing[:3]}")

    def __len__(self):
        return len(self.sources)

    def image(self, index: int, view_index: int) -> np.ndarray:
        path = self.index.image_path(self.sources[int(index)], int(view_index))
        with Image.open(path) as image:
            if image.format != "PNG" or image.mode != "RGB" or image.size != (256, 512):
                raise ValueError(f"PASD view contract mismatch: {path}")
            return np.asarray(image).copy()

    def caption(self, index: int, view_index: int) -> str:
        return self.index.caption(self.sources[int(index)], int(view_index))

    def weights(self, index: int) -> list[float]:
        return self.index.weights(self.sources[int(index)])


def eval_view_path(
    image_path: str | Path,
    data_root: str | Path,
    output_root: str | Path,
    manifest_path: str | Path,
    view_index: int,
    views: int = 5,
) -> str:
    index = load_index(
        str(Path(manifest_path).expanduser().resolve()),
        str(Path(data_root).expanduser().resolve()),
        str(Path(output_root).expanduser().resolve()),
        int(views),
    )
    return str(index.image_path(index.key_for_path(image_path), int(view_index)))
