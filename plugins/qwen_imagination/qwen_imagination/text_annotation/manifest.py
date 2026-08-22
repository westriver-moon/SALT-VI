from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


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


def source_record_path(root: str | Path, source_key: str) -> Path:
    source = Path(source_key)
    return Path(root) / "metadata" / source.parent / f"{source.stem}.json"


def save_source_record(root: str | Path, record: dict) -> Path:
    return atomic_json(source_record_path(root, record["source_key"]), record)


def load_source_record(root: str | Path, source_key: str) -> dict | None:
    path = source_record_path(root, source_key)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def valid_cached_record(record: dict | None, run_signature: dict) -> bool:
    return bool(
        record
        and record.get("status") == "complete"
        and record.get("run_signature") == run_signature
    )


def _atomic_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def consolidate_shard(
    root: str | Path,
    records: list[dict],
    *,
    split: str,
    shard_index: int,
    num_shards: int,
    expected_source_count: int,
) -> dict:
    if len(records) != int(expected_source_count):
        raise ValueError(
            f"record count {len(records)} != expected {expected_source_count}"
        )
    root = Path(root)
    stem = f"{split}.shard-{shard_index:05d}-of-{num_shards:05d}"
    rows = []
    for record in records:
        row = {
            key: record.get(key)
            for key in (
                "schema_version",
                "annotation_version",
                "source_key",
                "identity",
                "camera",
                "modality",
                "split",
                "status",
            )
        }
        if record.get("status") == "complete":
            row.update(
                {
                    "global": record["annotation"]["global"],
                    "regions": record["annotation"]["regions"],
                    "sampled_text_worlds": record["sampled_text_worlds"],
                    "selected_region_ids": record["selected_region_ids"],
                    "annotation_provenance": record.get("annotation_provenance"),
                }
            )
        else:
            row["failure"] = record.get("failure")
        rows.append(row)
    manifest = _atomic_jsonl(root / "manifests" / f"{stem}.jsonl", rows)
    completed = sum(record.get("status") == "complete" for record in records)
    failed = len(records) - completed
    summary = {
        "schema_version": 1,
        "split": split,
        "shard_index": int(shard_index),
        "num_shards": int(num_shards),
        "source_count": len(records),
        "completed_source_count": completed,
        "failed_source_count": failed,
        "complete": failed == 0,
        "manifest": str(manifest),
    }
    atomic_json(root / "manifests" / f"{stem}.summary.json", summary)
    return summary
