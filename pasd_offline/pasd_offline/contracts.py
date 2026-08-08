from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Iterable

from .config import GenerationConfig
from .tasks import GenerationTask


BUILD_CONTRACT_NAME = "build-contract.json"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def content_identity(path: Path) -> dict:
    path = path.expanduser().resolve()
    if path.is_file():
        return {"path": str(path), "file_count": 1, "sha256": file_sha256(path)}
    files = sorted(value for value in path.rglob("*") if value.is_file())
    digest = hashlib.sha256()
    for value in files:
        relative = value.relative_to(path).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(bytes.fromhex(file_sha256(value)))
    return {"path": str(path), "file_count": len(files), "sha256": digest.hexdigest()}


def implementation_identity(root: Path | None = None) -> dict:
    root = root or Path(__file__).resolve().parents[1]
    roots = [root / "pasd_offline", root / "vendor" / "pasd"]
    files = sorted(
        (value for source_root in roots for value in source_root.rglob("*.py")),
        key=lambda value: value.relative_to(root).as_posix(),
    )
    digest = hashlib.sha256()
    for value in files:
        digest.update(value.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(file_sha256(value)))
    return {"file_count": len(files), "sha256": digest.hexdigest()}


def input_identity(tasks: Iterable[GenerationTask]) -> dict:
    sources = {
        task.source_key or str(task.image): task.image.expanduser().resolve() for task in tasks
    }
    digest = hashlib.sha256()
    for source_key, path in sorted(sources.items()):
        digest.update(source_key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(file_sha256(path)))
    return {"source_count": len(sources), "sha256": digest.hexdigest()}


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False)
    temporary = Path(handle.name)
    try:
        with handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_contract_digest(payload: dict) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def prepare_build_contract(
    config: GenerationConfig,
    records_path: str | Path,
    tasks: list[GenerationTask],
    implementation_root: Path | None = None,
) -> dict:
    records_path = Path(records_path).expanduser().resolve()
    models = {
        "stable_diffusion": content_identity(config.pretrained_model_path),
        "pasd": content_identity(config.pasd_model_path),
        "person_detector": (
            content_identity(config.person_detector_model)
            if config.person_detector_model is not None
            else None
        ),
    }
    payload = {
        "schema_version": 1,
        "records": content_identity(records_path),
        "inputs": input_identity(tasks),
        "models": models,
        "implementation": implementation_identity(implementation_root),
    }
    digest = build_contract_digest(payload)
    document = {**payload, "build_contract_sha256": digest}
    _atomic_json(config.output_root / BUILD_CONTRACT_NAME, document)
    config.build_contract_sha256 = digest
    return document

def load_build_contract(config: GenerationConfig) -> dict:
    path = config.output_root / BUILD_CONTRACT_NAME
    document = json.loads(path.read_text(encoding="utf-8"))
    digest = document.pop("build_contract_sha256")
    if build_contract_digest(document) != digest:
        raise ValueError(f"build contract digest mismatch: {path}")
    document["build_contract_sha256"] = digest
    config.build_contract_sha256 = digest
    return document
