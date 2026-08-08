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


def _caption_variants(metadata: Mapping) -> list[str]:
    description = str(metadata.get("description", "")).strip()
    paraphrases = metadata.get("paraphrases")
    if not description or not isinstance(paraphrases, list) or len(paraphrases) != 4:
        raise ValueError("each multiview caption record must have one description and four paraphrases")
    captions = [description, *(str(value).strip() for value in paraphrases)]
    if any(not value for value in captions):
        raise ValueError("the five caption variants must be non-empty")
    if len({value.casefold() for value in captions[1:]}) != 4:
        raise ValueError("the four paraphrases must be unique")
    return captions


def _record_output_paths(relative: str) -> list[str]:
    relative_path = Path(relative)
    directory = Path("images") / relative_path.parent / relative_path.stem
    return [str(directory / f"view_{index:02d}.png") for index in range(5)]


def build_sysu_multiview_records(
    dataset_root: str | Path,
    rgb_caption_candidates: str | Path,
    ir_caption_candidates: str | Path,
    output_path: str | Path,
    seed: int = 20_260_808,
    enforce_official_counts: bool = True,
) -> list[dict]:
    """Build the canonical one-row-per-source five-view generation contract."""

    dataset_root = Path(dataset_root).expanduser().resolve()
    splits = read_protocol_splits(dataset_root)
    identity_to_split = {
        identity: split for split, identities in splits.items() for identity in identities
    }
    caption_entries: dict[str, Mapping] = {}
    for path in (rgb_caption_candidates, ir_caption_candidates):
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(values, dict):
            raise ValueError(f"caption candidate file is not a JSON object: {path}")
        overlap = set(caption_entries).intersection(values)
        if overlap:
            raise ValueError(f"duplicate caption keys across modalities: {sorted(overlap)[:3]}")
        caption_entries.update(values)

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
        source_path = (dataset_root / relative).resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"caption source image is missing: {source_path}")
        captions = _caption_variants(metadata)
        with Image.open(source_path) as image:
            source_width, source_height = image.size
        outputs = _record_output_paths(relative)
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
                "schema_version": 2,
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
    if enforce_official_counts and (
        len(records) != OFFICIAL_COUNTS["total"]
        or modality_counts["rgb"] != OFFICIAL_COUNTS["rgb"]
        or modality_counts["ir"] != OFFICIAL_COUNTS["ir"]
    ):
        raise ValueError(
            "SYSU protocol coverage mismatch: "
            f"total={len(records)} rgb={modality_counts['rgb']} ir={modality_counts['ir']} "
            f"expected={OFFICIAL_COUNTS}"
        )
    output_path = Path(output_path).expanduser().resolve()
    _atomic_jsonl(output_path, records)
    summary = {
        "schema_version": 2,
        "dataset_root": str(dataset_root),
        "records": len(records),
        "views": len(records) * 5,
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


def build_sysu_records(
    dataset_root: str | Path,
    caption_dicts: list[str | Path],
    output_path: str | Path,
    identity_caption_maps: list[str | Path] | None = None,
    caption_scope: str = "image",
) -> int:
    """Legacy one-caption record builder retained for compatibility."""

    dataset_root = Path(dataset_root).resolve()
    image_entries: dict[str, dict] = {}
    for path in caption_dicts:
        image_entries.update(json.loads(Path(path).read_text(encoding="utf-8")))

    identity_captions: dict[str, list[str]] = {}
    for path in identity_caption_maps or []:
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        for identity, captions in values.items():
            identity_captions.setdefault(str(int(identity)), []).extend(captions)

    if caption_scope == "identity":
        for metadata in image_entries.values():
            identity = str(int(metadata["id"]))
            if identity not in identity_captions:
                identity_captions.setdefault(identity, [])
                description = metadata["description"]
                if description not in identity_captions[identity]:
                    identity_captions[identity].append(description)

    rows = []
    for source_key, metadata in sorted(image_entries.items()):
        relative = _relative_source_key(source_key)
        camera = int(metadata["cam"])
        modality = "ir" if camera in IR_CAMERAS else "rgb"
        identity = str(int(metadata["id"]))
        record = {
            "image": str(dataset_root / relative),
            "output": str(Path("images") / modality / Path(relative).with_suffix(".png")),
            "modality": modality,
            "identity": identity,
        }
        if caption_scope == "identity":
            record["caption_pool_key"] = identity
        else:
            record["caption"] = metadata["description"]
        rows.append(record)
    output_path = Path(output_path)
    _atomic_jsonl(output_path, rows)
    if caption_scope == "identity":
        output_path.with_suffix(".caption-pool.json").write_text(
            json.dumps(identity_captions, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return len(rows)
