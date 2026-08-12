from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

from PIL import Image

from .geometry import PersonDetection, prepare_control_image
from .sysu import IR_CAMERAS, OFFICIAL_COUNTS, read_protocol_splits


TARGET_SIZE = (256, 512)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_manifest(root: Path, rows: list[dict], build: dict) -> dict:
    rows.sort(key=lambda row: (row["source_key"], int(row["view_index"])))
    manifest = root / "manifest.jsonl"
    digest = hashlib.sha256()
    with manifest.open("wb") as stream:
        for row in rows:
            encoded = (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
            stream.write(encoded)
            digest.update(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    summary = {
        "schema_version": 5,
        "views_per_source": 1,
        "source_count": len(rows),
        "view_count": len(rows),
        "expected_source_count": len(rows),
        "complete": True,
        "modalities": {
            modality: sum(row["modality"] == modality for row in rows)
            for modality in ("rgb", "ir")
        },
        "splits": {
            split: sum(row["split"] == split for row in rows)
            for split in ("train", "val", "test")
        },
        "manifest_jsonl": "manifest.jsonl",
        "manifest_jsonl_sha256": digest.hexdigest(),
        "build_sha256": hashlib.sha256(
            json.dumps(build, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }
    _atomic_json(root / "build.json", build)
    _atomic_json(root / "manifest.json", summary)
    return summary


def _load_complete_manifest(root: Path) -> list[dict]:
    manifest = root / "manifest.jsonl"
    summary = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if not summary.get("complete"):
        raise ValueError(f"source dataset is incomplete: {root}")
    if sha256_file(manifest) != summary["manifest_jsonl_sha256"]:
        raise ValueError(f"source manifest checksum mismatch: {manifest}")
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != int(summary["view_count"]):
        raise ValueError(f"source manifest count mismatch: {manifest}")
    return rows


def _collect_ir_sources(dataset_root: Path, enforce_official_counts: bool) -> list[dict]:
    splits = read_protocol_splits(dataset_root)
    identity_to_split = {
        identity: split for split, identities in splits.items() for identity in identities
    }
    rows = []
    for identity in sorted(identity_to_split):
        for camera in sorted(IR_CAMERAS):
            directory = dataset_root / f"cam{camera}" / identity
            if not directory.is_dir():
                continue
            for source in sorted(path for path in directory.iterdir() if path.is_file()):
                rows.append(
                    {
                        "source_key": source.relative_to(dataset_root).as_posix(),
                        "source": source,
                        "identity": identity,
                        "camera": camera,
                        "modality": "ir",
                        "split": identity_to_split[identity],
                    }
                )
    if enforce_official_counts and len(rows) != OFFICIAL_COUNTS["ir"]:
        raise ValueError(f"SYSU IR coverage mismatch: {len(rows)} != {OFFICIAL_COUNTS['ir']}")
    return rows


def _copy_rgb_rows(rgb_root: Path, combined_root: Path, rows: list[dict]) -> list[dict]:
    copied = []
    seen = set()
    for index, source in enumerate(rows, 1):
        if source.get("modality") != "rgb" or int(source.get("view_index", -1)) != 0:
            raise ValueError("RGB source manifest must contain exactly one RGB view per source")
        source_key = str(source["source_key"])
        if source_key in seen:
            raise ValueError(f"duplicate RGB source: {source_key}")
        seen.add(source_key)
        relative = Path(str(source["output"]))
        input_path = (rgb_root / relative).resolve()
        input_path.relative_to(rgb_root)
        output_path = combined_root / relative
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(input_path, output_path)
        row = dict(source)
        row.setdefault("hypothesis_weight", 1.0)
        row["image_backend"] = "pasd"
        copied.append(row)
        if index % 5000 == 0:
            print(f"copied RGB views: {index}/{len(rows)}", flush=True)
    return copied


def _build_ir_rows(dataset_root: Path, output_root: Path, sources: list[dict]) -> list[dict]:
    rows = []
    for index, source in enumerate(sources, 1):
        with Image.open(source["source"]) as image:
            detection = PersonDetection(
                (0.0, 0.0, float(image.width), float(image.height)), 0.0, "full_frame"
            )
            output, geometry = prepare_control_image(image, detection, target_size=TARGET_SIZE)
        relative = Path("images") / Path(source["source_key"]).with_suffix(".png")
        path = output_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        output.save(path, format="PNG", compress_level=4)
        rows.append(
            {
                "source_key": source["source_key"],
                "identity": source["identity"],
                "camera": source["camera"],
                "modality": "ir",
                "split": source["split"],
                "view_index": 0,
                "caption": "",
                "seed": 0,
                "output": relative.as_posix(),
                "output_sha256": sha256_file(path),
                "output_size": [256, 512],
                "source_sha256": sha256_file(source["source"]),
                "hypothesis_weight": 1.0,
                "image_backend": "geometry_only_blurpad",
                "semantic_generation": False,
                "geometry": geometry,
            }
        )
        if index % 1000 == 0:
            print(f"generated IR geometry views: {index}/{len(sources)}", flush=True)
    return rows


def validate_geomatched_dataset(
    root: str | Path,
    expected_modalities: dict[str, int],
    dataset_root: str | Path | None = None,
) -> dict:
    root = Path(root).expanduser().resolve()
    dataset_root = Path(dataset_root).expanduser().resolve() if dataset_root else None
    identity_to_split = None
    if dataset_root:
        splits = read_protocol_splits(dataset_root)
        identity_to_split = {
            identity: split for split, identities in splits.items() for identity in identities
        }
    rows = _load_complete_manifest(root)
    errors = []
    keys = set()
    counts = {"rgb": 0, "ir": 0}
    for index, row in enumerate(rows, 1):
        source_key = str(row.get("source_key", ""))
        key = (source_key, int(row.get("view_index", -1)))
        if key in keys:
            errors.append(f"duplicate:{source_key}")
            continue
        keys.add(key)
        modality = str(row.get("modality", ""))
        if modality not in counts:
            errors.append(f"modality:{source_key}:{modality}")
            continue
        counts[modality] += 1
        try:
            if dataset_root:
                source = (dataset_root / source_key).resolve()
                source.relative_to(dataset_root)
                if not source.is_file():
                    raise ValueError("missing_source")
                parts = Path(source_key).parts
                camera_part = str(parts[0])
                camera = int(camera_part[3:] if camera_part.startswith("cam") else camera_part)
                identity = str(parts[1]).zfill(4)
                expected_modality = "ir" if camera in IR_CAMERAS else "rgb"
                if camera != int(row["camera"]) or identity != str(row["identity"]):
                    raise ValueError("source_identity")
                if modality != expected_modality or row["split"] != identity_to_split[identity]:
                    raise ValueError("source_protocol")
            path = (root / str(row["output"])).resolve()
            path.relative_to(root)
            if sha256_file(path) != row["output_sha256"]:
                raise ValueError("checksum")
            with Image.open(path) as image:
                if image.format != "PNG" or image.mode != "RGB" or image.size != TARGET_SIZE:
                    raise ValueError(f"image_contract:{image.format}:{image.mode}:{image.size}")
            if modality == "ir":
                geometry = row["geometry"]
                source_width, source_height = geometry["source_size"]
                resized_width, resized_height = geometry["resized_size"]
                scale = float(geometry["scale"])
                if abs(resized_width - source_width * scale) > 1.01:
                    raise ValueError("ir_width_stretched")
                if abs(resized_height - source_height * scale) > 1.01:
                    raise ValueError("ir_height_stretched")
                if row.get("semantic_generation") is not False:
                    raise ValueError("ir_semantic_generation")
                if dataset_root and sha256_file(source) != row.get("source_sha256"):
                    raise ValueError("ir_source_checksum")
        except (KeyError, OSError, ValueError) as error:
            errors.append(f"invalid:{source_key}:{error}")
        if index % 5000 == 0:
            print(f"validated views: {index}/{len(rows)}", flush=True)
    if counts != expected_modalities:
        errors.append(f"counts:{counts}!={expected_modalities}")
    report = {
        "root": str(root),
        "source_count": len(rows),
        "modalities": counts,
        "error_count": len(errors),
        "errors": errors[:1000],
        "complete": not errors,
    }
    _atomic_json(root / "validation-report.json", report)
    if errors:
        raise ValueError(json.dumps(report, ensure_ascii=False))
    return report


def build_geomatched_dataset(
    dataset_root: str | Path,
    rgb_root: str | Path,
    ir_root: str | Path,
    combined_root: str | Path,
    *,
    enforce_official_counts: bool = True,
) -> dict:
    dataset_root = Path(dataset_root).expanduser().resolve()
    rgb_root = Path(rgb_root).expanduser().resolve()
    ir_root = Path(ir_root).expanduser().resolve()
    combined_root = Path(combined_root).expanduser().resolve()
    if ir_root.exists() or combined_root.exists():
        raise FileExistsError("refusing to replace an existing derived dataset")
    ir_stage = ir_root.with_name(f".{ir_root.name}.building-{os.getpid()}")
    combined_stage = combined_root.with_name(f".{combined_root.name}.building-{os.getpid()}")
    for stage in (ir_stage, combined_stage):
        if stage.exists():
            shutil.rmtree(stage)
        stage.mkdir(parents=True)
    try:
        rgb_rows = _load_complete_manifest(rgb_root)
        if enforce_official_counts and len(rgb_rows) != OFFICIAL_COUNTS["rgb"]:
            raise ValueError(f"SYSU RGB coverage mismatch: {len(rgb_rows)}")
        ir_sources = _collect_ir_sources(dataset_root, enforce_official_counts)
        ir_rows = _build_ir_rows(dataset_root, ir_stage, ir_sources)
        build = {
            "schema_version": 1,
            "dataset_root": str(dataset_root),
            "target_size": [256, 512],
            "geometry": {
                "mode": "person_fit_blurred_background",
                "background_blur_radius": 24.0,
                "foreground_feather_radius": 2.0,
                "semantic_generation": False,
            },
            "source_rgb_root": str(rgb_root),
            "source_rgb_manifest_sha256": sha256_file(rgb_root / "manifest.jsonl"),
            "protocol_sha256": {
                name: sha256_file(dataset_root / "exp" / name)
                for name in ("train_id.txt", "val_id.txt", "test_id.txt")
            },
            "builder_sha256": sha256_file(__file__),
        }
        _write_manifest(ir_stage, ir_rows, {**build, "dataset_kind": "ir_geometry_only"})
        combined_rows = _copy_rgb_rows(rgb_root, combined_stage, rgb_rows)
        for row in ir_rows:
            source = ir_stage / row["output"]
            target = combined_stage / row["output"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            combined_rows.append(dict(row))
        _write_manifest(
            combined_stage,
            combined_rows,
            {**build, "dataset_kind": "pasd_rgb_plus_geometry_only_ir"},
        )
        ir_expected = {"rgb": 0, "ir": len(ir_rows)}
        combined_expected = {"rgb": len(rgb_rows), "ir": len(ir_rows)}
        ir_report = validate_geomatched_dataset(ir_stage, ir_expected, dataset_root)
        combined_report = validate_geomatched_dataset(combined_stage, combined_expected, dataset_root)
        os.replace(ir_stage, ir_root)
        os.replace(combined_stage, combined_root)
        return {"ir": ir_report, "combined": combined_report}
    except Exception:
        shutil.rmtree(ir_stage, ignore_errors=True)
        shutil.rmtree(combined_stage, ignore_errors=True)
        raise
