"""Dataset adapters for the unified SYSU-MM01, RegDB, and LLCM PASD protocol."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from .config import PluginConfig
from .contracts import SourceRecord, stable_seed, write_jsonl


SYSU_RGB_CAMERAS = {1, 2, 4, 5}
SYSU_IR_CAMERAS = {3, 6}


def _canonical_key(value: str, dataset: str) -> str:
    normalized = value.replace("\\", "/").lstrip("./")
    marker = f"datasets/{dataset}/"
    if marker in normalized:
        normalized = normalized.split(marker, 1)[1]
    return normalized.lstrip("/")


def _caption_map(path: Path, dataset: str) -> dict[str, str]:
    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError(f"caption file is not a JSON object: {path}")
    mapped: dict[str, str] = {}
    for raw_key, metadata in values.items():
        if not isinstance(metadata, dict) or not str(metadata.get("description", "")).strip():
            raise ValueError(f"caption has no description: {raw_key}")
        key = _canonical_key(str(raw_key), dataset)
        if key in mapped:
            raise ValueError(f"duplicate caption key after normalization: {key}")
        mapped[key] = str(metadata["description"]).strip()
    return mapped


def _read_id_list(path: Path) -> set[str]:
    return {
        token.strip().zfill(4)
        for token in path.read_text(encoding="utf-8").replace("\n", ",").split(",")
        if token.strip()
    }


def _sysu_sources(root: Path) -> Iterable[dict[str, Any]]:
    identities: dict[str, str] = {}
    for split in ("train", "val", "test"):
        for identity in _read_id_list(root / "exp" / f"{split}_id.txt"):
            if identity in identities:
                raise ValueError(f"SYSU identity is in multiple protocol splits: {identity}")
            identities[identity] = split
    for identity, split in sorted(identities.items()):
        for camera in sorted(SYSU_RGB_CAMERAS | SYSU_IR_CAMERAS):
            directory = root / f"cam{camera}" / identity
            if not directory.is_dir():
                continue
            modality = "rgb" if camera in SYSU_RGB_CAMERAS else "ir"
            for image in sorted(candidate for candidate in directory.iterdir() if candidate.is_file()):
                key = image.relative_to(root).as_posix()
                yield {
                    "source_key": key,
                    "image": image,
                    "modality": modality,
                    "identity": identity,
                    "camera": camera,
                    "protocol": {"sysu": {"split": split}},
                    "source_label": None,
                }


def _regdb_index(path: Path, split: str, modality: str) -> Iterable[tuple[str, str, str]]:
    expected_directory = "Visible" if modality == "rgb" else "Thermal"
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = line.split()[0].replace("\\", "/").lstrip("./")
        if value.split("/", 1)[0].lower() != expected_directory.lower():
            raise ValueError(
                f"RegDB {modality} index must reference {expected_directory}/ paths: {value}"
            )
        yield value, split, modality


def _regdb_sources(root: Path) -> Iterable[dict[str, Any]]:
    sources: dict[str, dict[str, Any]] = {}
    for trial in range(1, 11):
        for split in ("train", "test"):
            for kind, modality in (("visible", "rgb"), ("thermal", "ir")):
                index = root / "idx" / f"{split}_{kind}_{trial}.txt"
                if not index.is_file():
                    raise FileNotFoundError(index)
                for raw_key, detected_split, detected_modality in _regdb_index(index, split, modality):
                    image = (root / raw_key).resolve()
                    key = image.relative_to(root.resolve()).as_posix()
                    if not image.is_file():
                        raise FileNotFoundError(image)
                    current = sources.get(key)
                    if current is None:
                        current = {
                            "source_key": key,
                            "image": image,
                            "modality": detected_modality,
                            "identity": Path(key).parent.name.zfill(4),
                            "camera": None,
                            "protocol": {"regdb": {"trials": {}}},
                            "source_label": None,
                        }
                        sources[key] = current
                    if current["modality"] != detected_modality:
                        raise ValueError(f"RegDB source has incompatible modalities: {key}")
                    trials = current["protocol"]["regdb"]["trials"]
                    previous = trials.get(str(trial))
                    if previous is not None and previous != detected_split:
                        raise ValueError(f"RegDB source has conflicting trial membership: {key} trial={trial}")
                    trials[str(trial)] = detected_split
    yield from (sources[key] for key in sorted(sources))


def _llcm_index(path: Path, split: str, modality: str) -> Iterable[tuple[str, int, str, str]]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 2:
            raise ValueError(f"invalid LLCM index line in {path}: {line!r}")
        yield fields[0].replace("\\", "/").lstrip("./"), int(fields[1]), split, modality


def _llcm_sources(root: Path) -> Iterable[dict[str, Any]]:
    specs = (("train", "vis", "rgb"), ("train", "nir", "ir"), ("test", "vis", "rgb"), ("test", "nir", "ir"))
    sources: dict[str, dict[str, Any]] = {}
    for split, kind, modality in specs:
        index = root / "idx" / f"{split}_{kind}.txt"
        if not index.is_file():
            raise FileNotFoundError(index)
        for raw_key, label, detected_split, detected_modality in _llcm_index(index, split, modality):
            image = (root / raw_key).resolve()
            key = image.relative_to(root.resolve()).as_posix()
            if not image.is_file():
                raise FileNotFoundError(image)
            match = re.search(r"_c(\d+)_", Path(key).name)
            current = sources.get(key)
            if current is not None:
                if current["modality"] != detected_modality or current["source_label"] != label:
                    raise ValueError(f"LLCM source has conflicting index metadata: {key}")
                if current["protocol"]["llcm"]["split"] != detected_split:
                    raise ValueError(f"LLCM source is in both train and test: {key}")
                continue
            sources[key] = {
                "source_key": key,
                "image": image,
                "modality": detected_modality,
                "identity": Path(key).parent.name.zfill(4),
                "camera": int(match.group(1)) if match else None,
                "protocol": {"llcm": {"split": detected_split, "label": label}},
                "source_label": label,
            }
    yield from (sources[key] for key in sorted(sources))


def _sources_for(config: PluginConfig) -> Iterable[dict[str, Any]]:
    if config.dataset == "sysu":
        return _sysu_sources(config.dataset_root)
    if config.dataset == "regdb":
        return _regdb_sources(config.dataset_root)
    if config.dataset == "llcm":
        return _llcm_sources(config.dataset_root)
    raise AssertionError(config.dataset)


def build_records(config: PluginConfig, output_path: str | Path | None = None) -> list[SourceRecord]:
    captions = {modality: _caption_map(path, config.dataset) for modality, path in config.captions.items()}
    sources = list(_sources_for(config))
    if not sources:
        raise ValueError(f"no official sources found for {config.dataset}")
    used = {"rgb": set(), "ir": set()}
    records: list[SourceRecord] = []
    for source in sources:
        key = source["source_key"]
        modality = source["modality"]
        caption = captions[modality].get(key)
        if caption is None:
            raise KeyError(f"missing {modality} caption for {config.dataset} source: {key}")
        used[modality].add(key)
        records.append(
            SourceRecord(
                source_key=key,
                image=str(source["image"]),
                modality=modality,
                identity=str(source["identity"]),
                camera=source["camera"],
                caption=caption,
                seed=stable_seed(config.seed, key),
                output=(Path("images") / Path(key)).with_suffix(".png").as_posix(),
                protocol=source["protocol"],
                source_label=source["source_label"],
            )
        )
    for modality in ("rgb", "ir"):
        extras = sorted(set(captions[modality]).difference(used[modality]))
        if extras:
            raise ValueError(f"unused {modality} captions for {config.dataset}: {extras[:3]} (total={len(extras)})")
    destination = Path(output_path).expanduser().resolve() if output_path else config.records_path
    return write_jsonl(destination, records)
