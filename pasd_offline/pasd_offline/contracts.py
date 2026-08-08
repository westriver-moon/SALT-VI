from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import tempfile
from importlib.metadata import version
from pathlib import Path
from typing import Iterable

from .config import GenerationConfig
from .tasks import GenerationTask


GENERATION_IDENTITY_NAME = "generation-identity.json"
DATASET_SCOPE_NAME = "dataset-scope.json"
RUNTIME_PACKAGES = ("torch", "diffusers", "transformers", "xformers")


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


def runtime_environment_identity(config: GenerationConfig, root: Path | None = None) -> dict:
    import torch

    root = root or Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,compute_cap,driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    allowed = set(config.gpu_allowlist)
    gpus = []
    for line in result.stdout.splitlines():
        index, name, compute_capability, driver = (
            value.strip() for value in line.split(",", 3)
        )
        if int(index) in allowed:
            gpus.append(
                {
                    "index": int(index),
                    "name": name,
                    "compute_capability": compute_capability,
                    "driver_version": driver,
                }
            )
    return {
        "requirements_lock": content_identity(root / "requirements-lock.txt"),
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "packages": {name: version(name) for name in RUNTIME_PACKAGES},
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpus": gpus,
    }


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


def identity_digest(payload: dict) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def generation_identity_payload(
    config: GenerationConfig,
    implementation_root: Path | None = None,
    environment: dict | None = None,
) -> dict:
    return {
        "schema_version": 2,
        "generation": config.output_contract(),
        "models": {
            "stable_diffusion": content_identity(config.pretrained_model_path),
            "pasd": content_identity(config.pasd_model_path),
            "person_detector": (
                content_identity(config.person_detector_model)
                if config.person_detector_model is not None
                else None
            ),
        },
        "implementation": implementation_identity(implementation_root),
        "environment": environment or runtime_environment_identity(config, implementation_root),
    }


def dataset_scope_payload(records_path: str | Path, tasks: list[GenerationTask]) -> dict:
    return {
        "schema_version": 1,
        "records": content_identity(Path(records_path)),
        "inputs": input_identity(tasks),
    }


def _write_identity(path: Path, payload: dict, digest_field: str) -> dict:
    digest = identity_digest(payload)
    document = {**payload, digest_field: digest}
    _atomic_json(path, document)
    return document


def _load_identity(path: Path, digest_field: str) -> dict:
    document = json.loads(path.read_text(encoding="utf-8"))
    digest = document.pop(digest_field)
    if identity_digest(document) != digest:
        raise ValueError(f"identity digest mismatch: {path}")
    document[digest_field] = digest
    return document


def prepare_generation_identity(
    config: GenerationConfig,
    implementation_root: Path | None = None,
    environment: dict | None = None,
) -> dict:
    document = _write_identity(
        config.output_root / GENERATION_IDENTITY_NAME,
        generation_identity_payload(config, implementation_root, environment),
        "generation_identity_sha256",
    )
    config.generation_identity_sha256 = document["generation_identity_sha256"]
    return document


def prepare_dataset_scope(
    config: GenerationConfig, records_path: str | Path, tasks: list[GenerationTask]
) -> dict:
    document = _write_identity(
        config.output_root / DATASET_SCOPE_NAME,
        dataset_scope_payload(records_path, tasks),
        "dataset_scope_sha256",
    )
    config.dataset_scope_sha256 = document["dataset_scope_sha256"]
    return document


def prepare_contracts(
    config: GenerationConfig,
    records_path: str | Path,
    tasks: list[GenerationTask],
    implementation_root: Path | None = None,
    environment: dict | None = None,
) -> tuple[dict, dict]:
    generation = prepare_generation_identity(config, implementation_root, environment)
    scope = prepare_dataset_scope(config, records_path, tasks)
    return generation, scope


def load_generation_identity(config: GenerationConfig) -> dict:
    document = _load_identity(
        config.output_root / GENERATION_IDENTITY_NAME, "generation_identity_sha256"
    )
    config.generation_identity_sha256 = document["generation_identity_sha256"]
    return document


def load_dataset_scope(config: GenerationConfig) -> dict:
    document = _load_identity(
        config.output_root / DATASET_SCOPE_NAME, "dataset_scope_sha256"
    )
    config.dataset_scope_sha256 = document["dataset_scope_sha256"]
    return document


def verify_generation_identity(
    config: GenerationConfig,
    implementation_root: Path | None = None,
    environment: dict | None = None,
) -> None:
    expected = load_generation_identity(config)["generation_identity_sha256"]
    actual = identity_digest(
        generation_identity_payload(config, implementation_root, environment)
    )
    if actual != expected:
        raise ValueError("generation identity changed after dataset build started")


def verify_dataset_scope(
    config: GenerationConfig, records_path: str | Path, tasks: list[GenerationTask]
) -> None:
    expected = load_dataset_scope(config)["dataset_scope_sha256"]
    actual = identity_digest(dataset_scope_payload(records_path, tasks))
    if actual != expected:
        raise ValueError("dataset scope changed after dataset build started")
