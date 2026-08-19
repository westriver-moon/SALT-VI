"""Protocol and geometry validation for generated PASD artifacts."""

from __future__ import annotations

import json

from .config import PluginConfig
from .contracts import SourceRecord, atomic_json, sha256_file
from .generation import validate_record


def validate_geometry(geometry: dict, config: PluginConfig) -> None:
    if not isinstance(geometry, dict):
        raise ValueError("missing geometry audit")
    mode = geometry.get("mode")
    if mode == "direct_rewrite":
        expected = [config.target_width, config.target_height]
        if (
            geometry.get("source_size") != expected
            or geometry.get("target_size") != expected
        ):
            raise ValueError(
                "direct rewrite must use the canonical source and target canvas"
            )
        if geometry.get("resized_size") != expected:
            raise ValueError("direct rewrite cannot resize the canonical canvas")
        if geometry.get("padding") != [0, 0, 0, 0]:
            raise ValueError("direct rewrite cannot add padding")
        transform = geometry.get("transform", {})
        if transform != {
            "scale_x": 1.0,
            "scale_y": 1.0,
            "offset_x": 0.0,
            "offset_y": 0.0,
        }:
            raise ValueError("direct rewrite must preserve identity coordinates")
        if geometry.get("background_restoration") is not False:
            raise ValueError("direct rewrite cannot restore a blurred background")
        return
    if mode != "person_fit_blurred_background":
        raise ValueError(f"unknown geometry mode: {mode}")
    source_width, source_height = geometry["source_size"]
    resized_width, resized_height = geometry["resized_size"]
    target_width, target_height = geometry["target_size"]
    scale = float(geometry["scale"])
    if [target_width, target_height] != [config.target_width, config.target_height]:
        raise ValueError("geometry target mismatch")
    if min(source_width, source_height, resized_width, resized_height, scale) <= 0:
        raise ValueError("non-positive geometry")
    if (
        abs(resized_width - source_width * scale) > 1.01
        or abs(resized_height - source_height * scale) > 1.01
    ):
        raise ValueError("aspect ratio was stretched")
    left, top, right, bottom = geometry["padding"]
    if min(left, top, right, bottom) < 0:
        raise ValueError("negative padding")
    if (
        left + resized_width + right != target_width
        or top + resized_height + bottom != target_height
    ):
        raise ValueError("padding size mismatch")
    if geometry.get("foreground_box") != [
        left,
        top,
        left + resized_width,
        top + resized_height,
    ]:
        raise ValueError("foreground box mismatch")
    if float(geometry.get("background_blur_radius", 0)) <= 0:
        raise ValueError("invalid blurred background audit")


def validate_protocol(record: SourceRecord, dataset: str) -> None:
    if dataset == "sysu":
        if set(record.protocol) != {"sysu"} or record.protocol["sysu"].get(
            "split"
        ) not in {"train", "val", "test"}:
            raise ValueError("invalid SYSU protocol membership")
    elif dataset == "regdb":
        trials = record.protocol.get("regdb", {}).get("trials", {})
        if set(trials) != {str(index) for index in range(1, 11)}:
            raise ValueError("RegDB record does not contain all ten trial memberships")
        if any(value not in {"train", "test"} for value in trials.values()):
            raise ValueError("invalid RegDB trial membership")
    elif dataset == "llcm":
        membership = record.protocol.get("llcm", {})
        if (
            membership.get("split") not in {"train", "test"}
            or membership.get("label") != record.source_label
        ):
            raise ValueError("invalid LLCM protocol membership")
    else:
        raise ValueError(f"unknown dataset: {dataset}")


def validate_manifest(config: PluginConfig, records: list[SourceRecord]) -> list[dict]:
    """Verify that the manifest exactly represents the validated source records."""

    errors: list[dict] = []
    manifest_path = config.output_root / "manifest.jsonl"
    summary_path = config.output_root / "manifest.json"
    if not manifest_path.is_file() or not summary_path.is_file():
        return [
            {"source_key": None, "error": "missing manifest.jsonl or manifest.json"}
        ]
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        rows = [
            json.loads(line)
            for line in manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return [{"source_key": None, "error": f"invalid manifest: {error}"}]

    expected = {record.source_key: record for record in records}
    seen: set[str] = set()
    for row in rows:
        key = row.get("source_key")
        record = expected.get(key)
        if key in seen:
            errors.append({"source_key": key, "error": "duplicate manifest source"})
            continue
        seen.add(key)
        if record is None:
            errors.append({"source_key": key, "error": "unknown manifest source"})
            continue
        try:
            expected_payload = record.payload()
            if any(
                row.get(field) != value for field, value in expected_payload.items()
            ):
                raise ValueError("manifest record differs from records contract")
            marker = validate_record(config, record)
            for field in (
                "input_sha256",
                "build_sha256",
                "source_contract_sha256",
                "artifact",
                "geometry",
            ):
                if row.get(field) != marker.get(field):
                    raise ValueError(f"manifest {field} differs from source metadata")
            validate_geometry(row["geometry"], config)
        except (KeyError, OSError, ValueError, json.JSONDecodeError) as error:
            errors.append({"source_key": key, "error": str(error)})
    missing = sorted(set(expected).difference(seen))
    errors.extend(
        {"source_key": key, "error": "source absent from manifest"} for key in missing
    )

    digest = sha256_file(manifest_path)
    if summary.get("dataset") != config.dataset:
        errors.append({"source_key": None, "error": "manifest dataset mismatch"})
    if summary.get("expected_source_count") != len(records) or summary.get(
        "source_count"
    ) != len(rows):
        errors.append({"source_key": None, "error": "manifest source counts mismatch"})
    if summary.get("complete") is not True or summary.get("error_count") != 0:
        errors.append({"source_key": None, "error": "manifest is incomplete"})
    if summary.get("build_sha256") != config.build_sha256:
        errors.append(
            {"source_key": None, "error": "manifest build fingerprint mismatch"}
        )
    if summary.get("manifest_jsonl_sha256") != digest:
        errors.append({"source_key": None, "error": "manifest checksum mismatch"})
    return errors


def validate_dataset(config: PluginConfig, records: list[SourceRecord]) -> dict:
    errors = []
    for record in records:
        try:
            validate_protocol(record, config.dataset)
            marker = validate_record(config, record)
            validate_geometry(marker["geometry"], config)
        except (KeyError, OSError, ValueError, json.JSONDecodeError) as error:
            errors.append({"source_key": record.source_key, "error": str(error)})
    errors.extend(validate_manifest(config, records))
    report = {
        "dataset": config.dataset,
        "expected_sources": len(records),
        "validated_sources": len(records) - len(errors),
        "error_count": len(errors),
        "errors": errors[:1000],
        "complete": not errors,
    }
    atomic_json(config.output_root / "validation-report.json", report)
    return report
