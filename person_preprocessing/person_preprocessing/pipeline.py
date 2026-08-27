"""Build fixed ViT images, then cache generic pose on those final images."""

import argparse
import collections
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
from PIL import Image
import yaml

from .geometry import render
from .models import PoseEstimator
from .store import PersonAssetStore


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def read_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line
    ]


def file_identity(path, include_sha256=False):
    path = Path(path).expanduser().resolve()
    stat = path.stat()
    identity = {
        "path": str(path),
        "bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if include_sha256:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        identity["sha256"] = digest.hexdigest()
    return identity


def bind_contract(path, contract):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != contract:
            raise ValueError(
                "output contract changed; choose a new output directory: " + str(path)
            )
    else:
        write_json(path, contract)


def pose_weight(config):
    if config.get("pose_weight"):
        return Path(config["pose_weight"]).expanduser().resolve()
    legacy = {Path(value).expanduser().resolve() for value in config.get("pose_weights", {}).values()}
    if len(legacy) != 1:
        raise ValueError("configure exactly one shared pose_weight")
    return legacy.pop()


def source_metadata(row):
    path = Path(row["source"])
    match = re.search(r"(?:^|/|_)(?:cam|c)(\d+)(?:/|_)", row["source"])
    if row["dataset"] == "regdb":
        camera = row["modality"]
    else:
        camera = "cam" + str(int(match.group(1))) if match else ""
    return {
        "dataset": row["dataset"],
        "source_key": row["source"],
        "identity": path.parent.name,
        "camera": camera,
        "modality": row["modality"],
        "references": row.get("references", []),
        "aliases": row.get("aliases", []),
    }


def selected_rows(config, datasets, limit, source_keys=None):
    if limit < 0:
        raise ValueError("limit-per-modality must be nonnegative")
    rows = [
        row
        for row in read_jsonl(config["input_manifest"])
        if row["dataset"] in datasets
    ]
    if source_keys is not None:
        keys = {(row["dataset"], row["source"]) for row in source_keys}
        available = {(row["dataset"], row["source"]) for row in rows}
        if not keys or keys - available:
            raise ValueError("source-list must contain known keys in selected datasets")
        rows = [row for row in rows if (row["dataset"], row["source"]) in keys]
    if limit:
        groups = collections.defaultdict(list)
        for row in rows:
            groups[(row["dataset"], row["modality"])].append(row)
        rows = [
            group[int(index)]
            for group in groups.values()
            for index in np.linspace(
                0, len(group) - 1, min(limit, len(group)), dtype=int
            )
        ]
    return rows


def prepare(config, datasets, modes, device, limit=0, source_keys=None):
    """Create deterministic final images; person_fit uses pose only for localization."""
    if limit and source_keys is not None:
        raise ValueError("choose a sample count or an explicit source list")
    rows = selected_rows(config, datasets, limit, source_keys)
    native_contract = json.loads(
        (Path(config["input_root"]) / "contract.json").read_text(encoding="utf-8")
    )
    scale = int(native_contract.get("scale", config.get("input_scale", 1)))
    locator = (
        PoseEstimator(
            pose_weight(config),
            device,
            config["detection_confidence"],
            config["pose_imgsz"],
        )
        if "person_fit" in modes
        else None
    )
    summaries = []
    for mode in modes:
        for dataset in datasets:
            selected = [row for row in rows if row["dataset"] == dataset]
            root = Path(config["output_root"]) / mode / dataset
            contract = {
                "version": 2,
                "mode": mode,
                "dataset": dataset,
                "size_hw": [int(value) for value in config["size_hw"]],
                "input_manifest": file_identity(config["input_manifest"]),
                "input_root": str(Path(config["input_root"]).resolve()),
                "native_sr_contract": native_contract,
                "geometry": (
                    "uniform_affine_person_crop_v1"
                    if mode == "person_fit"
                    else "bicubic_resize_v1"
                ),
            }
            if mode == "person_fit":
                contract.update(
                    locator_weight=file_identity(pose_weight(config), include_sha256=True),
                    detection_confidence=float(config["detection_confidence"]),
                    pose_imgsz=int(config["pose_imgsz"]),
                    person_margin=float(config["person_margin"]),
                    blur_radius=float(config["blur_radius"]),
                )
            bind_contract(root / "contract.json", contract)
            for row in selected:
                metadata = root / "records" / Path(row["output"]).with_suffix(".json")
                image_path = root / "images" / row["output"]
                if metadata.exists():
                    record = json.loads(metadata.read_text(encoding="utf-8"))
                    if record["status"] == "complete" and not image_path.is_file():
                        raise FileNotFoundError(image_path)
                    continue
                native_path = Path(config["input_root"]) / dataset / row["output"]
                with Image.open(native_path) as image:
                    image = image.convert("RGB")
                expected = (int(row["width"]) * scale, int(row["height"]) * scale)
                if image.size != expected:
                    raise ValueError(
                        f"input size {image.size} does not match inventory {expected}: {native_path}"
                    )
                person = locator.predict(image) if mode == "person_fit" else None
                record = source_metadata(row)
                record.update(
                    mode=mode,
                    native_image=str(native_path),
                    image="images/" + row["output"],
                )
                if mode == "person_fit" and person is None:
                    record["status"] = "no_person_detected"
                else:
                    output, geometry = render(
                        image,
                        mode,
                        config["size_hw"],
                        person["bbox"] if person else None,
                        config["person_margin"],
                        config["blur_radius"],
                    )
                    image_path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = image_path.with_suffix(".png.tmp")
                    output.save(temporary, format="PNG", compress_level=3)
                    os.replace(temporary, image_path)
                    record.update(status="complete", geometry=geometry)
                    if person:
                        record["localization"] = {
                            key: value.tolist() if isinstance(value, np.ndarray) else value
                            for key, value in person.items()
                            if key != "keypoints"
                        }
                write_json(metadata, record)
            records = [
                json.loads(path.read_text(encoding="utf-8"))
                for path in sorted((root / "records").rglob("*.json"))
            ]
            write_jsonl(root / "images.jsonl", records)
            summary = {
                "dataset": dataset,
                "mode": mode,
                "selected": len(selected),
                "inventory_total": sum(
                    row["dataset"] == dataset
                    for row in read_jsonl(config["input_manifest"])
                ),
                "counts": dict(collections.Counter(row["status"] for row in records)),
                "manifest_records": len(records),
            }
            summary["complete_inventory"] = (
                summary["counts"].get("complete", 0) == summary["inventory_total"]
            )
            write_json(root / "images.summary.json", summary)
            summaries.append(summary)
    return summaries


def pose(config, datasets, modes, device):
    """Run pose once on each final ViT-sized image and cache only generic outputs."""
    weight = pose_weight(config)
    estimator = PoseEstimator(
        weight,
        device,
        config["detection_confidence"],
        config["pose_imgsz"],
    )
    summaries = []
    for mode in modes:
        for dataset in datasets:
            store = PersonAssetStore(Path(config["output_root"]) / mode, dataset)
            if not store.records:
                raise ValueError("cannot estimate pose for an empty image manifest")
            contract = {
                "version": 2,
                "images": store.contract,
                "format": "coco17_raw_pose_v1",
                "coordinate_space": "final_vit_input_pixels_xy",
                "weight": file_identity(weight, include_sha256=True),
                "detection_confidence": float(config["detection_confidence"]),
                "pose_imgsz": int(config["pose_imgsz"]),
            }
            bind_contract(store.root / "pose.contract.json", contract)
            results = []
            for row in store.records:
                if row["status"] != "complete":
                    results.append(
                        {"source_key": row["source_key"], "status": "image_unavailable"}
                    )
                    continue
                relative = Path(row["image"]).relative_to("images").with_suffix(".npz")
                path = store.root / "pose" / relative
                metadata = path.with_suffix(".json")
                if metadata.exists() and path.is_file():
                    results.append(json.loads(metadata.read_text(encoding="utf-8")))
                    continue
                prediction = estimator.predict(store.image(row["source_key"]))
                arrays = {
                    "keypoints": (
                        prediction["keypoints"]
                        if prediction
                        else np.zeros((17, 3), dtype=np.float32)
                    ),
                    "bbox": (
                        prediction["bbox"]
                        if prediction
                        else np.zeros(4, dtype=np.float32)
                    ),
                    "confidence": np.asarray(
                        prediction["confidence"] if prediction else 0.0,
                        dtype=np.float32,
                    ),
                    "person_count": np.asarray(
                        prediction["person_count"] if prediction else 0,
                        dtype=np.int32,
                    ),
                    "size_hw": np.asarray(store.size_hw, dtype=np.int32),
                }
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(".npz.tmp")
                with temporary.open("wb") as stream:
                    np.savez_compressed(stream, **arrays)
                os.replace(temporary, path)
                record = {
                    "source_key": row["source_key"],
                    "dataset": dataset,
                    "mode": mode,
                    "modality": row["modality"],
                    "image": row["image"],
                    "pose": path.relative_to(store.root).as_posix(),
                    "status": "complete" if prediction else "no_person_detected",
                    "model_path": str(weight),
                    "bbox": arrays["bbox"].tolist() if prediction else None,
                    "confidence": float(arrays["confidence"]),
                    "person_count": int(arrays["person_count"]),
                }
                write_json(metadata, record)
                results.append(record)
            write_jsonl(store.root / "pose.jsonl", results)
            summary = {
                "dataset": dataset,
                "mode": mode,
                "images": len(store.records),
                "counts": dict(collections.Counter(row["status"] for row in results)),
            }
            summary["complete_inventory"] = (
                len(results)
                == sum(
                    row["dataset"] == dataset
                    for row in read_jsonl(config["input_manifest"])
                )
                and all(row["status"] != "image_unavailable" for row in results)
            )
            write_json(store.root / "pose.summary.json", summary)
            summaries.append(summary)
    return summaries


def verify(config, datasets, modes, require_pose=False):
    reports = []
    inventory = read_jsonl(config["input_manifest"])
    for mode in modes:
        for dataset in datasets:
            store = PersonAssetStore(Path(config["output_root"]) / mode, dataset)
            checked, rejected, posed = 0, 0, 0
            expected = {
                row["source"]: row for row in inventory if row["dataset"] == dataset
            }
            if require_pose and store.pose_contract().get("images") != store.contract:
                raise ValueError("pose cache is not bound to the current final images")
            for row in store.records:
                original = expected[row["source_key"]]
                if any(
                    row[key] != value
                    for key, value in source_metadata(original).items()
                ):
                    raise ValueError(
                        "source metadata differs from inventory: " + row["source_key"]
                    )
                if row["status"] != "complete":
                    rejected += 1
                    continue
                image = np.asarray(store.image(row["source_key"]))
                if row["modality"] == "ir" and not np.array_equal(
                    image, np.repeat(image[..., :1], 3, axis=2)
                ):
                    raise ValueError("infrared channels no longer agree: " + row["source_key"])
                if require_pose:
                    arrays = store.pose(row["source_key"], require_person=False)
                    if arrays["keypoints"].shape != (17, 3):
                        raise ValueError("incorrect COCO-17 keypoint shape")
                    if arrays["bbox"].shape != (4,):
                        raise ValueError("incorrect pose bbox shape")
                    if tuple(arrays["size_hw"].tolist()) != store.size_hw:
                        raise ValueError("pose and final image dimensions differ")
                    if not all(np.isfinite(value).all() for value in arrays.values()):
                        raise ValueError("non-finite pose cache")
                    posed += 1
                checked += 1
            total = len(expected)
            reports.append(
                {
                    "dataset": dataset,
                    "mode": mode,
                    "valid_images": checked,
                    "pose_artifacts": posed,
                    "rejected_images": rejected,
                    "inventory_total": total,
                    "missing_records": total - len(store.records),
                    "complete_inventory": checked == total,
                }
            )
    return reports


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "pose", "verify"))
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("sysu", "regdb", "llcm"),
        default=["sysu", "regdb", "llcm"],
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("resize", "person_fit"),
        default=["resize", "person_fit"],
    )
    parser.add_argument("--device", default="0")
    parser.add_argument("--output-root")
    parser.add_argument("--limit-per-modality", type=int, default=0)
    parser.add_argument("--source-list")
    parser.add_argument("--require-pose", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args(argv)
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.output_root:
        config["output_root"] = args.output_root
    if args.command == "prepare":
        source_keys = read_jsonl(args.source_list) if args.source_list else None
        result = prepare(
            config,
            args.datasets,
            args.modes,
            args.device,
            args.limit_per_modality,
            source_keys,
        )
    elif args.command == "pose":
        result = pose(config, args.datasets, args.modes, args.device)
    else:
        result = verify(config, args.datasets, args.modes, args.require_pose)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return int(
        args.require_complete
        and not all(row.get("complete_inventory", False) for row in result)
    )
