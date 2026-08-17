"""Resumable generation and artifact consolidation for unified PASD records."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Callable

from PIL import Image, ImageStat

from .config import PluginConfig
from .contracts import SourceRecord, atomic_json, load_records, sha256_file


def _document_digest(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _build_payload(config: PluginConfig, records_path: Path) -> dict:
    return {
        "schema_version": 1,
        "config": config.output_contract(),
        "records_sha256": sha256_file(records_path),
    }


def prepare_build(config: PluginConfig, records_path: str | Path) -> dict:
    records_path = Path(records_path).expanduser().resolve()
    payload = _build_payload(config, records_path)
    document = {**payload, "build_sha256": _document_digest(payload)}
    destination = config.output_root / "build.json"
    if destination.is_file() and json.loads(destination.read_text(encoding="utf-8")) != document:
        raise ValueError(f"output root belongs to a different build: {destination}")
    atomic_json(destination, document)
    config.build_sha256 = document["build_sha256"]
    return document


def load_build(config: PluginConfig) -> dict:
    path = config.output_root / "build.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    digest = document.pop("build_sha256", "")
    if not digest or _document_digest(document) != digest:
        raise ValueError(f"invalid build fingerprint: {path}")
    document["build_sha256"] = digest
    config.build_sha256 = digest
    return document


def _source_contract(record: SourceRecord, config: PluginConfig, input_sha256: str) -> str:
    if not config.build_sha256:
        raise ValueError("source contract requires a loaded build")
    payload = {
        "build_sha256": config.build_sha256,
        "input_sha256": input_sha256,
        "record": record.payload(),
    }
    return _document_digest(payload)


def _metadata_path(root: Path, source_key: str) -> Path:
    source = Path(source_key)
    return root / "metadata" / source.parent / f"{source.stem}.json"


def _output_path(root: Path, record: SourceRecord) -> Path:
    path = (root / record.output).resolve()
    path.relative_to(root.resolve())
    return path


def _atomic_png(path: Path, image: Image.Image, compress_level: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.png")
    try:
        image.save(temporary, format="PNG", compress_level=compress_level)
        with Image.open(temporary) as check:
            check.verify()
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _matching_marker(config: PluginConfig, record: SourceRecord) -> dict | None:
    marker_path = _metadata_path(config.output_root, record.source_key)
    if not marker_path.is_file():
        return None
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        digest = sha256_file(record.image)
        if marker.get("source_contract_sha256") != _source_contract(record, config, digest):
            return None
        output = _output_path(config.output_root, record)
        artifact = marker["artifact"]
        if not output.is_file() or output.stat().st_size != int(artifact["output_bytes"]):
            return None
        return marker
    except (KeyError, OSError, ValueError, json.JSONDecodeError):
        return None


def is_generated(config: PluginConfig, record: SourceRecord) -> bool:
    return _matching_marker(config, record) is not None


def validate_record(config: PluginConfig, record: SourceRecord) -> dict:
    marker = _matching_marker(config, record)
    if marker is None:
        raise ValueError("source is not generated for the current build")
    input_sha256 = sha256_file(record.image)
    if marker.get("input_sha256") != input_sha256:
        raise ValueError("input checksum mismatch")
    if marker.get("build_sha256") != config.build_sha256:
        raise ValueError("build fingerprint mismatch")
    output = _output_path(config.output_root, record)
    artifact = marker["artifact"]
    if sha256_file(output) != artifact["output_sha256"]:
        raise ValueError("output checksum mismatch")
    with Image.open(output) as image:
        image.load()
        if image.format != "PNG" or image.mode != "RGB" or image.size != (256, 512):
            raise ValueError(f"invalid output contract: {image.format}/{image.mode}/{image.size}")
        if max(ImageStat.Stat(image).var) <= 0:
            raise ValueError("output is constant")
    return marker


def generate_record(generator, config: PluginConfig, record: SourceRecord, physical_gpu: int | None) -> dict:
    image = Path(record.image)
    if not image.is_file():
        raise FileNotFoundError(image)
    outputs, geometry = generator.generate_views(
        image, [record.caption], [record.seed], modality=record.modality, batch_size=1
    )
    if len(outputs) != 1:
        raise RuntimeError(f"PASD returned {len(outputs)} images for {record.source_key}")
    output = outputs[0].convert("RGB")
    if output.size != (config.target_width, config.target_height):
        raise ValueError(f"PASD output has wrong size for {record.source_key}: {output.size}")
    if max(ImageStat.Stat(output).var) <= 0:
        raise ValueError(f"PASD output is constant for {record.source_key}")
    output_path = _output_path(config.output_root, record)
    _atomic_png(output_path, output, config.png_compress_level)
    input_sha256 = sha256_file(image)
    marker = {
        "schema_version": 1,
        "record": record.payload(),
        "input_sha256": input_sha256,
        "physical_gpu": physical_gpu,
        "num_inference_steps": config.num_inference_steps,
        "geometry": geometry,
        "build_sha256": config.build_sha256,
        "source_contract_sha256": _source_contract(record, config, input_sha256),
        "artifact": {
            "output": record.output,
            "output_sha256": sha256_file(output_path),
            "output_bytes": output_path.stat().st_size,
            "output_size": list(output.size),
        },
        "completed_at_unix": time.time(),
    }
    atomic_json(_metadata_path(config.output_root, record.source_key), marker)
    return marker


def generate_records(
    config: PluginConfig,
    records: list[SourceRecord],
    *,
    physical_gpu: int | None = None,
    max_sources: int | None = None,
    generator_factory: Callable[[PluginConfig], object] | None = None,
) -> dict:
    if physical_gpu is not None and physical_gpu not in config.gpu_allowlist:
        raise ValueError(f"physical GPU {physical_gpu} is outside the configured allowlist")
    config.output_root.mkdir(parents=True, exist_ok=True)
    generator = None
    completed = skipped = 0
    for record in records:
        if max_sources is not None and completed >= max_sources:
            break
        if is_generated(config, record):
            skipped += 1
            continue
        if generator is None:
            if generator_factory is None:
                from .runtime import PASDGenerator

                generator_factory = PASDGenerator
            generator = generator_factory(config)
        generate_record(generator, config, record, physical_gpu)
        completed += 1
    return {"completed": completed, "skipped": skipped, "requested": len(records)}


def consolidate_manifest(config: PluginConfig, records: list[SourceRecord]) -> dict:
    manifest = config.output_root / "manifest.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=manifest.parent, delete=False) as handle:
        temporary = Path(handle.name)
        digest = hashlib.sha256()
        validated = 0
        errors = []
        for record in records:
            try:
                marker = validate_record(config, record)
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
                errors.append({"source_key": record.source_key, "error": str(error)})
                continue
            row = {
                **record.payload(),
                "input_sha256": marker["input_sha256"],
                "build_sha256": marker["build_sha256"],
                "source_contract_sha256": marker["source_contract_sha256"],
                "artifact": marker["artifact"],
                "geometry": marker["geometry"],
            }
            encoded = (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
            handle.write(encoded)
            digest.update(encoded)
            validated += 1
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, manifest)
    finally:
        temporary.unlink(missing_ok=True)
    summary = {
        "schema_version": 1,
        "dataset": config.dataset,
        "views_per_source": 1,
        "source_count": validated,
        "expected_source_count": len(records),
        "complete": not errors and validated == len(records),
        "manifest_jsonl": "manifest.jsonl",
        "manifest_jsonl_sha256": digest.hexdigest(),
        "errors": errors[:1000],
        "error_count": len(errors),
        "build_sha256": config.build_sha256,
    }
    atomic_json(config.output_root / "manifest.json", summary)
    return summary


def load_protocol_records(path: str | Path) -> list[SourceRecord]:
    return load_records(path)
