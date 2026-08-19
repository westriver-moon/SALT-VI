from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Iterable


WEIGHT_FIELDS = {
    "uniform": "uniform_weight",
    "proposal": "proposal_weight",
    "posterior": "posterior_weight",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: str | Path, payload: object) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def source_metadata_path(root: str | Path, source_key: str) -> Path:
    source = Path(source_key)
    return Path(root) / "metadata" / source.parent / f"{source.stem}.json"


def save_source_record(root: str | Path, record: dict) -> Path:
    return atomic_json(source_metadata_path(root, record["source_key"]), record)


def load_source_record(root: str | Path, source_key: str) -> dict | None:
    path = source_metadata_path(root, source_key)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _flat_rows(records: Iterable[dict], weight_field: str) -> list[dict]:
    rows = []
    for record in records:
        worlds = list(record["worlds"])
        weights = [float(world[weight_field]) for world in worlds]
        total = sum(weights)
        if total <= 0:
            raise ValueError(f"source {record['source_key']} has no {weight_field} mass")
        for index, (world, weight) in enumerate(zip(worlds, weights)):
            row = {
                key: record[key]
                for key in ("source_key", "identity", "camera", "modality", "split")
            }
            row.update(
                {
                    "view_index": index,
                    "hypothesis_weight": float(weight / total),
                    "caption": world["caption"],
                    "seed": int(world["seed"]),
                    "output": world["output"],
                    "output_sha256": world["output_sha256"],
                    "output_bytes": int(world["output_bytes"]),
                    "output_size": [256, 512],
                    "world_id": world["world_id"],
                    "assignments": world["assignments"],
                    "proposal_mass": float(world["proposal_mass"]),
                    "uniform_weight": float(world["uniform_weight"]),
                    "proposal_weight": float(world["proposal_weight"]),
                    "posterior_weight": float(world["posterior_weight"]),
                    "E_LR": float(world["e_lr"]),
                    "E_ID": float(world["e_id"]),
                    "E_edit": float(world["e_edit"]),
                    "mask_path": world.get("mask_path"),
                    "mask_sha256": world.get("mask_sha256"),
                    "fallback": bool(record.get("fallback", False)),
                }
            )
            rows.append(row)
    return rows


def _atomic_jsonl(path: Path, rows: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        digest = hashlib.sha256()
        for row in rows:
            encoded = (
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            ).encode("utf-8")
            stream.write(encoded)
            digest.update(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return digest.hexdigest()


def consolidate_manifests(
    output_root: str | Path,
    records: list[dict],
    *,
    expected_source_count: int,
    build_sha256: str,
) -> dict[str, dict]:
    output_root = Path(output_root).resolve()
    if len(records) != expected_source_count:
        raise ValueError(
            f"QRI source record count {len(records)} != expected {expected_source_count}"
        )
    records = sorted(records, key=lambda record: record["source_key"])
    raw_path = output_root / "manifests" / "regional.jsonl"
    raw_sha = _atomic_jsonl(raw_path, records)
    summaries = {}
    for variant, weight_field in WEIGHT_FIELDS.items():
        rows = _flat_rows(records, weight_field)
        path = output_root / "manifests" / f"manifest.{variant}.jsonl"
        digest = _atomic_jsonl(path, rows)
        summary = {
            "schema_version": 2,
            "plugin": "qwen-regional-imagination-v1",
            "weight_variant": variant,
            "views_per_source": 0,
            "source_count": len(records),
            "view_count": len(rows),
            "complete": True,
            "build_sha256": build_sha256,
            "manifest_jsonl": path.name,
            "manifest_jsonl_sha256": digest,
            "regional_manifest": str(raw_path.relative_to(output_root)),
            "regional_manifest_sha256": raw_sha,
            "fallback_source_count": sum(bool(record.get("fallback")) for record in records),
        }
        atomic_json(path.with_suffix(".json"), summary)
        summaries[variant] = summary
    return summaries
