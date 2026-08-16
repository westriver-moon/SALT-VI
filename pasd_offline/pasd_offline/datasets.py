from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from PIL import Image


@dataclass(frozen=True)
class RecordSource:
    """One canonical PASD source resolved from a caption dictionary."""

    relative_key: str
    source_key: str
    metadata: Mapping
    modality: str
    identity: str
    split: str
    camera: int


def atomic_jsonl(path: Path, rows: Iterable[Mapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False)
    temporary = Path(handle.name)
    try:
        with handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def stable_seed(base_seed: int, source_key: str, view_index: int) -> int:
    payload = f"{base_seed}\0{source_key}\0{view_index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31 - 1)


def relative_source_key(source_key: str, dataset_prefix: str) -> str:
    """Strip a ``datasets/<name>/`` caption prefix while keeping paths relative."""

    normalized = source_key.replace("\\", "/")
    marker = f"datasets/{dataset_prefix}/"
    if marker in normalized:
        normalized = normalized.split(marker, 1)[1]
    return normalized.lstrip("/")


def caption_variants(metadata: Mapping, views_per_source: int) -> list[str]:
    description = str(metadata.get("description", "")).strip()
    if not description:
        raise ValueError("each caption record must have a non-empty description")
    if views_per_source == 1:
        return [description]
    paraphrases = metadata.get("paraphrases")
    if not isinstance(paraphrases, list) or len(paraphrases) != 4:
        raise ValueError("each multiview caption record must have one description and four paraphrases")
    captions = [description, *(str(value).strip() for value in paraphrases)]
    if any(not value for value in captions):
        raise ValueError("the five caption variants must be non-empty")
    if len({value.casefold() for value in captions[1:]}) != 4:
        raise ValueError("the four paraphrases must be unique")
    return captions


def record_output_paths(relative: str, views_per_source: int) -> list[str]:
    relative_path = Path(relative)
    if views_per_source == 1:
        return [str((Path("images") / relative_path).with_suffix(".png"))]
    directory = Path("images") / relative_path.parent / relative_path.stem
    return [str(directory / f"view_{index:02d}.png") for index in range(5)]


def load_caption_entries(
    caption_candidates: Mapping[str, str | Path],
) -> tuple[dict[str, Mapping], dict[str, str]]:
    """Load one or more caption dictionaries and validate modality overlap."""

    candidates = {str(key).lower(): value for key, value in caption_candidates.items()}
    if not candidates or not set(candidates).issubset({"rgb", "ir"}):
        raise ValueError("caption_candidates must contain rgb and/or ir entries")
    caption_entries: dict[str, Mapping] = {}
    source_modalities: dict[str, str] = {}
    for requested_modality, path in candidates.items():
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(values, dict):
            raise ValueError(f"caption candidate file is not a JSON object: {path}")
        overlap = set(caption_entries).intersection(values)
        if overlap:
            raise ValueError(f"duplicate caption keys across modalities: {sorted(overlap)[:3]}")
        caption_entries.update(values)
        source_modalities.update({key: requested_modality for key in values})
    return caption_entries, source_modalities


def build_records(
    dataset_root: str | Path,
    sources: list[RecordSource],
    output_path: str | Path,
    seed: int,
    views_per_source: int,
) -> list[dict]:
    """Build canonical one-row-per-source records for any caption-keyed dataset."""

    if views_per_source not in (1, 5):
        raise ValueError("views_per_source must be 1 or 5")
    dataset_root = Path(dataset_root).expanduser().resolve()
    seen: set[str] = set()
    records: list[dict] = []
    for source in sources:
        if source.relative_key in seen:
            raise ValueError(f"duplicate source key: {source.relative_key}")
        seen.add(source.relative_key)
        source_path = (dataset_root / source.relative_key).resolve()
        source_path.relative_to(dataset_root)
        if not source_path.is_file():
            raise FileNotFoundError(f"caption source image is missing: {source_path}")
        captions = caption_variants(source.metadata, views_per_source)
        with Image.open(source_path) as image:
            source_width, source_height = image.size
        outputs = record_output_paths(source.relative_key, views_per_source)
        views = [
            {
                "view_index": index,
                "caption": caption,
                "seed": stable_seed(seed, source.source_key, index),
                "output": outputs[index],
            }
            for index, caption in enumerate(captions)
        ]
        records.append(
            {
                "schema_version": 4,
                "source_key": source.source_key,
                "image": str(source_path),
                "identity": source.identity,
                "camera": source.camera,
                "modality": source.modality,
                "split": source.split,
                "source_size": [source_width, source_height],
                "captions": captions,
                "views": views,
            }
        )
    output_path = Path(output_path).expanduser().resolve()
    atomic_jsonl(output_path, records)
    return records


def write_records_summary(
    output_path: str | Path,
    summary: Mapping,
) -> None:
    output_path = Path(output_path).expanduser().resolve()
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def records_summary(
    dataset_root: str | Path,
    caption_candidates: Mapping[str, str | Path],
    records: list[dict],
    output_path: str | Path,
    *,
    views_per_source: int,
    seed: int,
    split_keys: tuple[str, ...],
    extra: Mapping | None = None,
) -> dict:
    modalities = ("rgb", "ir")
    modality_counts = {
        modality: sum(record["modality"] == modality for record in records)
        for modality in modalities
    }
    payload: dict = {
        "schema_version": 4,
        "dataset_root": str(Path(dataset_root).expanduser().resolve()),
        "caption_candidates": {
            modality: {
                "path": str(Path(path).expanduser().resolve()),
                "sha256": hashlib.sha256(
                    Path(path).expanduser().resolve().read_bytes()
                ).hexdigest(),
            }
            for modality, path in caption_candidates.items()
        },
        "records_sha256": hashlib.sha256(
            Path(output_path).expanduser().resolve().read_bytes()
        ).hexdigest(),
        "records": len(records),
        "views": len(records) * views_per_source,
        "views_per_source": views_per_source,
        "modalities": modality_counts,
        "splits": {
            split: sum(record["split"] == split for record in records)
            for split in split_keys
        },
        "seed": int(seed),
    }
    if extra:
        payload.update(extra)
    write_records_summary(output_path, payload)
    return payload
