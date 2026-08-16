from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping

from .datasets import (
    RecordSource,
    atomic_jsonl,
    build_records,
    load_caption_entries,
    records_summary,
    relative_source_key,
)


RGB_CAMERAS = {1, 2, 4, 5}
IR_CAMERAS = {3, 6}
OFFICIAL_COUNTS = {"rgb": 29_033, "ir": 15_712, "total": 44_745}


def _read_ids(path: Path) -> set[str]:
    return {
        token.strip().zfill(4)
        for token in path.read_text(encoding="utf-8").replace("\n", ",").split(",")
        if token.strip()
    }


def read_protocol_splits(dataset_root: str | Path) -> dict[str, set[str]]:
    root = Path(dataset_root).expanduser().resolve()
    splits = {
        "train": _read_ids(root / "exp" / "train_id.txt"),
        "val": _read_ids(root / "exp" / "val_id.txt"),
        "test": _read_ids(root / "exp" / "test_id.txt"),
    }
    if len(set().union(*splits.values())) != sum(len(values) for values in splits.values()):
        raise ValueError("SYSU train/val/test identity lists overlap")
    return splits


def _relative_source_key(source_key: str) -> str:
    return relative_source_key(source_key, "sysu")


def build_sysu_records(
    dataset_root: str | Path,
    caption_candidates: Mapping[str, str | Path],
    output_path: str | Path,
    seed: int = 20_260_808,
    views_per_source: int = 5,
    enforce_official_counts: bool = True,
    include_all: bool = False,
) -> list[dict]:
    """Build the canonical one-row-per-source PASD generation contract."""

    dataset_root = Path(dataset_root).expanduser().resolve()
    splits = read_protocol_splits(dataset_root)
    identity_to_split = {
        identity: split for split, identities in splits.items() for identity in identities
    }
    caption_entries, source_modalities = load_caption_entries(caption_candidates)
    sources: list[RecordSource] = []
    for source_key, metadata in sorted(caption_entries.items()):
        identity = str(metadata["id"]).zfill(4)
        split = identity_to_split.get(identity)
        if split is None and not include_all:
            continue
        if split is None:
            split = "all"
        relative = _relative_source_key(source_key)
        camera = int(metadata["cam"])
        modality = "ir" if camera in IR_CAMERAS else "rgb"
        if camera not in RGB_CAMERAS | IR_CAMERAS:
            raise ValueError(f"unsupported SYSU camera {camera} for {source_key}")
        if modality != source_modalities[source_key]:
            raise ValueError(
                f"caption modality mismatch for {source_key}: "
                f"declared={source_modalities[source_key]} camera={camera}"
            )
        sources.append(
            RecordSource(
                relative_key=relative,
                source_key=relative,
                metadata=metadata,
                modality=modality,
                identity=identity,
                split=split,
                camera=camera,
            )
        )

    records = build_records(
        dataset_root,
        sources,
        output_path,
        seed=seed,
        views_per_source=views_per_source,
    )
    modality_counts = {
        modality: sum(record["modality"] == modality for record in records)
        for modality in ("rgb", "ir")
    }
    expected_total = sum(OFFICIAL_COUNTS[modality] for modality in caption_candidates)
    if enforce_official_counts and (
        len(records) != expected_total
        or any(
            modality_counts[modality] != OFFICIAL_COUNTS[modality]
            for modality in caption_candidates
        )
        or any(
            modality_counts[modality]
            for modality in {"rgb", "ir"}.difference(caption_candidates)
        )
    ):
        raise ValueError(
            "SYSU protocol coverage mismatch: "
            f"total={len(records)} rgb={modality_counts['rgb']} ir={modality_counts['ir']} "
            f"enabled={sorted(caption_candidates)} expected={OFFICIAL_COUNTS}"
        )
    records_summary(
        dataset_root,
        caption_candidates,
        records,
        output_path,
        views_per_source=views_per_source,
        seed=seed,
        split_keys=("train", "val", "test", "all"),
        extra={"include_all": bool(include_all)},
    )
    return records


def select_pilot_records(
    records: list[dict], output_path: str | Path, count: int = 100, seed: int = 20_260_808
) -> list[dict]:
    """Deterministically cover modality, split, scale, and aspect-ratio strata."""

    if count < 1 or count > len(records):
        raise ValueError("pilot count must be between 1 and the number of records")
    groups: dict[tuple[str, str, str, str], list[dict]] = {}
    for record in records:
        width, height = record["source_size"]
        scale = min(256 / width, 512 / height)
        scale_bucket = "downsample" if scale < 1 else "x1_1p5" if scale < 1.5 else "x1p5_2p5" if scale < 2.5 else "x2p5_plus"
        aspect = width / height
        aspect_bucket = "narrow" if aspect < 0.45 else "target" if aspect <= 0.55 else "wide"
        key = (record["modality"], record["split"], scale_bucket, aspect_bucket)
        groups.setdefault(key, []).append(record)
    for key, values in groups.items():
        values.sort(
            key=lambda record: hashlib.sha256(
                f"{seed}\0{key}\0{record['source_key']}".encode("utf-8")
            ).digest()
        )
    selected: list[dict] = []
    keys = sorted(groups)
    while len(selected) < count:
        progressed = False
        for key in keys:
            if groups[key]:
                selected.append(groups[key].pop(0))
                progressed = True
                if len(selected) == count:
                    break
        if not progressed:
            break
    atomic_jsonl(Path(output_path).expanduser().resolve(), selected)
    return selected
