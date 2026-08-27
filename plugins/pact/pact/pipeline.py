"""Derive PACT-only anatomy from the shared final-image pose cache."""

import argparse
import collections
import json
import os
from pathlib import Path

import numpy as np
import yaml

from person_preprocessing import PersonAssetStore

from .anatomy import anatomical_masks


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _bind_contract(path, contract):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != contract:
            raise ValueError("PACT contract changed; choose a new output directory")
    else:
        _write_json(path, contract)


def build(config, datasets, modes):
    summaries = []
    for mode in modes:
        for dataset in datasets:
            person = PersonAssetStore(Path(config["person_assets_root"]) / mode, dataset)
            root = Path(config["output_root"]) / mode / dataset
            contract = {
                "version": 1,
                "plugin": "pact-v1",
                "person_pose_contract": person.pose_contract(),
                "anatomy": "coco17_four_parts_v1",
                "background_semantics": "padding_only; uncovered foreground is residual",
                "keypoint_confidence": float(config["keypoint_confidence"]),
                "patch_hw": [int(value) for value in config["patch_hw"]],
                "stride_hw": [int(value) for value in config["stride_hw"]],
            }
            _bind_contract(root / "contract.json", contract)
            results = []
            for row in person.records:
                if row["status"] != "complete":
                    results.append(
                        {"source_key": row["source_key"], "status": "image_unavailable"}
                    )
                    continue
                relative = Path(row["image"]).relative_to("images").with_suffix(".npz")
                path = root / "anatomy" / relative
                metadata = path.with_suffix(".json")
                if path.is_file() and metadata.is_file():
                    results.append(json.loads(metadata.read_text(encoding="utf-8")))
                    continue
                pose_record = person.pose_record(row["source_key"], require_person=False)
                pose = person.pose(row["source_key"], require_person=False)
                arrays = anatomical_masks(
                    pose["keypoints"],
                    person.size_hw,
                    row["geometry"]["foreground_box"],
                    config["keypoint_confidence"],
                    config["patch_hw"],
                    config["stride_hw"],
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(".npz.tmp")
                with temporary.open("wb") as stream:
                    np.savez_compressed(stream, **arrays)
                os.replace(temporary, path)
                result = {
                    "source_key": row["source_key"],
                    "dataset": dataset,
                    "mode": mode,
                    "modality": row["modality"],
                    "status": (
                        "complete"
                        if pose_record["status"] == "complete"
                        else "no_person_detected"
                    ),
                    "anatomy": path.relative_to(root).as_posix(),
                    "part_valid": arrays["part_valid"].tolist(),
                    "part_confidence": arrays["part_confidence"].tolist(),
                }
                _write_json(metadata, result)
                results.append(result)
            _write_jsonl(root / "anatomy.jsonl", results)
            summary = {
                "dataset": dataset,
                "mode": mode,
                "images": len(person.records),
                "counts": dict(collections.Counter(row["status"] for row in results)),
                "usable_parts": dict(
                    zip(
                        ("head", "torso", "arms", "legs"),
                        np.sum(
                            [row.get("part_valid", [False] * 4) for row in results],
                            axis=0,
                        )
                        .astype(int)
                        .tolist(),
                    )
                ),
            }
            _write_json(root / "anatomy.summary.json", summary)
            summaries.append(summary)
    return summaries


def verify(config, datasets, modes):
    reports = []
    ph, pw = (int(value) for value in config["patch_hw"])
    sy, sx = (int(value) for value in config["stride_hw"])
    for mode in modes:
        for dataset in datasets:
            from .store import PACTArtifactStore

            store = PACTArtifactStore(
                config["person_assets_root"], config["output_root"], mode, dataset
            )
            checked = 0
            for row in store.person.records:
                if row["status"] != "complete":
                    continue
                arrays = store.anatomy(row["source_key"])
                height, width = store.person.size_hw
                if arrays["pixel_masks"].shape != (4, height, width):
                    raise ValueError("incorrect PACT pixel mask shape")
                if arrays["token_masks"].shape != (
                    4,
                    (height - ph) // sy + 1,
                    (width - pw) // sx + 1,
                ):
                    raise ValueError("incorrect PACT token grid")
                partition = (
                    arrays["pixel_masks"].astype(float).sum(0)
                    + arrays["background_mask"]
                    + arrays["residual_mask"]
                )
                if not np.allclose(partition, 1, atol=0.002):
                    raise ValueError("PACT pixel partition is inconsistent")
                token_partition = (
                    arrays["token_masks"].astype(float).sum(0)
                    + arrays["token_background"]
                    + arrays["token_residual"]
                )
                if not np.allclose(token_partition, 1, atol=0.002):
                    raise ValueError("PACT token partition is inconsistent")
                checked += 1
            reports.append({"dataset": dataset, "mode": mode, "checked": checked})
    return reports


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "verify"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--person-assets-root")
    parser.add_argument("--output-root")
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
    args = parser.parse_args(argv)
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.person_assets_root:
        config["person_assets_root"] = args.person_assets_root
    if args.output_root:
        config["output_root"] = args.output_root
    result = (
        build(config, args.datasets, args.modes)
        if args.command == "build"
        else verify(config, args.datasets, args.modes)
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0
