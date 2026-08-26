"""Canonical records and artifact helpers for every supported dataset."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SourceRecord:
    source_key: str
    image: str
    modality: str
    identity: str
    camera: int | None
    caption: str
    seed: int
    output: str
    protocol: dict[str, Any]
    source_label: int | None = None

    def payload(self) -> dict[str, Any]:
        return {"schema_version": SCHEMA_VERSION, **asdict(self)}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(base_seed: int, source_key: str) -> int:
    payload = f"{base_seed}\0{source_key}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31 - 1)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_jsonl(path: str | Path, records: Iterable[SourceRecord]) -> list[SourceRecord]:
    path = Path(path).expanduser().resolve()
    materialized = sorted(records, key=lambda record: record.source_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        for record in materialized:
            handle.write(json.dumps(record.payload(), ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return materialized


def load_records(path: str | Path) -> list[SourceRecord]:
    path = Path(path).expanduser().resolve()
    records: list[SourceRecord] = []
    seen: set[str] = set()
    fields = SourceRecord.__dataclass_fields__
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if int(value.get("schema_version", -1)) != SCHEMA_VERSION:
            raise ValueError(f"unsupported records schema in {path}")
        record = SourceRecord(**{key: value.get(key) for key in fields})
        if record.source_key in seen:
            raise ValueError(f"duplicate source key: {record.source_key}")
        if record.modality not in ("rgb", "ir"):
            raise ValueError(f"unsupported modality for {record.source_key}: {record.modality}")
        if not record.caption.strip():
            raise ValueError(f"empty caption for {record.source_key}")
        seen.add(record.source_key)
        records.append(record)
    if not records:
        raise ValueError(f"records are empty: {path}")
    return sorted(records, key=lambda record: record.source_key)
