"""Manifest-backed SYSU PASD views without materializing large NPY arrays."""

from __future__ import annotations

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


class PASDMultiviewIndex:
    def __init__(self, manifest_path: str | Path, data_root: str | Path, output_root: str | Path, views: int):
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.data_root = Path(data_root).expanduser().resolve()
        self.output_root = Path(output_root).expanduser().resolve()
        self.views = int(views)
        if self.views != 5:
            raise ValueError(f"SYSU PASD contract requires five views, got {self.views}")
        records = {}
        with self.manifest_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                record = json.loads(line)
                key = str(record["source_key"]).replace("\\", "/").lstrip("/")
                if key in records:
                    raise ValueError(f"duplicate PASD source key: {key}")
                if len(record.get("views", [])) != self.views:
                    raise ValueError(f"PASD source {key} does not have {self.views} views")
                indices = [int(view["view_index"]) for view in record["views"]]
                if indices != list(range(self.views)):
                    raise ValueError(f"PASD source {key} has invalid view indices {indices}")
                records[key] = record
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
        path = self.output_root / view["output"]
        if not path.is_file():
            raise FileNotFoundError(f"missing PASD view: {path}")
        return path

    def caption(self, source_key: str, view_index: int) -> str:
        return str(self.record(source_key)["views"][int(view_index)]["caption"])


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
            image = image.convert("RGB")
            if image.size != (256, 512):
                raise ValueError(f"PASD view has size {image.size}, expected (256, 512): {path}")
            return np.asarray(image).copy()

    def caption(self, index: int, view_index: int) -> str:
        return self.index.caption(self.sources[int(index)], int(view_index))


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
