from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Iterable, Mapping

from PIL import Image


RGB_CAMERAS = {1, 2, 4, 5}
IR_CAMERAS = {3, 6}
OFFICIAL_COUNTS = {"rgb": 29_033, "ir": 15_712, "total": 44_745}


def _atomic_jsonl(path: Path, rows: Iterable[Mapping]) -> None:
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
    normalized = source_key.replace("\\", "/")
    if "datasets/sysu/" in normalized:
        normalized = normalized.split("datasets/sysu/", 1)[1]
    return normalized.lstrip("/")


def _stable_seed(base_seed: int, source_key: str, view_index: int) -> int:
    payload = f"{base_seed}\0{source_key}\0{view_index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31 - 1)


def _caption_variants(metadata: Mapping, views_per_source: int) -> list[str]:
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


def _record_output_paths(relative: str, views_per_source: int) -> list[str]:
    relative_path = Path(relative)
    if views_per_source == 1:
        return [str((Path("images") / relative_path).with_suffix(".png"))]
    directory = Path("images") / relative_path.parent / relative_path.stem
    return [str(directory / f"view_{index:02d}.png") for index in range(5)]


def build_sysu_records(
    dataset_root: str | Path,
    caption_candidates: Mapping[str, str | Path],
    output_path: str | Path,
    seed: int = 20_260_808,
    views_per_source: int = 5,
    enforce_official_counts: bool = True,
) -> list[dict]:
    """Build the canonical one-row-per-source PASD generation contract."""

    dataset_root = Path(dataset_root).expanduser().resolve()
    if views_per_source not in (1, 5):
        raise ValueError("views_per_source must be 1 or 5")
    splits = read_protocol_splits(dataset_root)
    identity_to_split = {
        identity: split for split, identities in splits.items() for identity in identities
    }
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

    records: list[dict] = []
    for source_key, metadata in sorted(caption_entries.items()):
        identity = str(metadata["id"]).zfill(4)
        split = identity_to_split.get(identity)
        if split is None:
            continue
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
        source_path = (dataset_root / relative).resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"caption source image is missing: {source_path}")
        captions = _caption_variants(metadata, views_per_source)
        with Image.open(source_path) as image:
            source_width, source_height = image.size
        outputs = _record_output_paths(relative, views_per_source)
        views = [
            {
                "view_index": index,
                "caption": caption,
                "seed": _stable_seed(seed, relative, index),
                "output": outputs[index],
            }
            for index, caption in enumerate(captions)
        ]
        records.append(
            {
                "schema_version": 4,
                "source_key": relative,
                "image": str(source_path),
                "identity": identity,
                "camera": camera,
                "modality": modality,
                "split": split,
                "source_size": [source_width, source_height],
                "captions": captions,
                "views": views,
            }
        )

    modality_counts = {
        modality: sum(record["modality"] == modality for record in records)
        for modality in ("rgb", "ir")
    }
    expected_total = sum(OFFICIAL_COUNTS[modality] for modality in candidates)
    if enforce_official_counts and (
        len(records) != expected_total
        or any(modality_counts[modality] != OFFICIAL_COUNTS[modality] for modality in candidates)
        or any(modality_counts[modality] for modality in {"rgb", "ir"}.difference(candidates))
    ):
        raise ValueError(
            "SYSU protocol coverage mismatch: "
            f"total={len(records)} rgb={modality_counts['rgb']} ir={modality_counts['ir']} "
            f"enabled={sorted(candidates)} expected={OFFICIAL_COUNTS}"
        )
    output_path = Path(output_path).expanduser().resolve()
    _atomic_jsonl(output_path, records)
    summary = {
        "schema_version": 4,
        "dataset_root": str(dataset_root),
        "caption_candidates": {
            modality: {
                "path": str(Path(path).expanduser().resolve()),
                "sha256": hashlib.sha256(Path(path).expanduser().resolve().read_bytes()).hexdigest(),
            }
            for modality, path in candidates.items()
        },
        "records_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "records": len(records),
        "views": len(records) * views_per_source,
        "views_per_source": views_per_source,
        "modalities": modality_counts,
        "splits": {
            split: sum(record["split"] == split for record in records)
            for split in ("train", "val", "test")
        },
        "seed": int(seed),
    }
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
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
    _atomic_jsonl(Path(output_path).expanduser().resolve(), selected)
    return selected
