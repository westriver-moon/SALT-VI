from __future__ import annotations

from pathlib import Path
from typing import Mapping

from .datasets import (
    RecordSource,
    build_records,
    load_caption_entries,
    records_summary,
    relative_source_key,
)


OFFICIAL_COUNTS = {"rgb": 4_120, "ir": 4_120, "total": 8_240}


def _read_idx(path: Path) -> dict[str, str]:
    """Read an RegDB trial index into ``{dataset-relative path: split}``."""

    entries: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if not parts:
            continue
        key = parts[0].strip().replace("\\", "/")
        key = key.lstrip("./")
        if key in entries:
            raise ValueError(f"duplicate RegDB index path: {key}")
        entries[key] = path.name.split("_", 1)[0]
    return entries


def read_trial_splits(dataset_root: str | Path, trial: int = 1) -> dict[str, str]:
    root = Path(dataset_root).expanduser().resolve()
    entries: dict[str, str] = {}
    for split, prefix in (("train", "train"), ("test", "test")):
        for modality in ("visible", "thermal"):
            path = root / "idx" / f"{prefix}_{modality}_{trial}.txt"
            for key, detected_split in _read_idx(path).items():
                previous = entries.get(key)
                if previous is not None and previous != detected_split:
                    raise ValueError(f"RegDB index overlaps splits: {key}")
                entries[key] = detected_split
    return entries


def build_regdb_records(
    dataset_root: str | Path,
    caption_candidates: Mapping[str, str | Path],
    output_path: str | Path,
    seed: int = 20_260_808,
    views_per_source: int = 1,
    trial: int = 1,
    enforce_official_counts: bool = True,
    include_all: bool = True,
) -> list[dict]:
    """Build canonical PASD records for the full RegDB RGB and IR sets."""

    if views_per_source != 1:
        raise ValueError("RegDB records currently require one view per source")
    dataset_root = Path(dataset_root).expanduser().resolve()
    caption_entries, source_modalities = load_caption_entries(caption_candidates)
    splits = read_trial_splits(dataset_root, trial=trial)
    sources: list[RecordSource] = []
    for source_key, metadata in sorted(caption_entries.items()):
        relative = relative_source_key(source_key, "regdb")
        modality = source_modalities[source_key]
        parts = Path(relative).parts
        if len(parts) < 2:
            raise ValueError(f"invalid RegDB source path: {source_key}")
        identity = str(parts[-2]).zfill(4)
        split = splits.get(relative)
        if split is None and not include_all:
            continue
        sources.append(
            RecordSource(
                relative_key=relative,
                source_key=relative,
                metadata=metadata,
                modality=modality,
                identity=identity,
                split=split or "all",
                camera=0,
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
            "RegDB coverage mismatch: "
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
        split_keys=("train", "test", "all"),
        extra={"trial": int(trial), "include_all": bool(include_all)},
    )
    return records
