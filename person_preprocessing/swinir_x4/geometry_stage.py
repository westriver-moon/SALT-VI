#!/usr/bin/env python3
"""CPU-only geometry after native SwinIR x4; never overwrite native SR images."""
import argparse
import collections
import importlib.metadata
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image

import pipeline as p
from geometry import ALGORITHM, CpuPersonDetector, ensure_ir_channels, fit_with_background, validate_settings


def load_settings(path):
    settings = json.loads(Path(path).read_text())
    validate_settings(settings)
    settings["output_dir"] = str((Path(path).resolve().parent/settings["output_dir"]).resolve())
    settings["python"] = os.path.abspath(str(Path(path).resolve().parent/settings["python"]))
    model = settings["detector"]["model_path"]
    settings["detector"]["model_path"] = str((Path(path).resolve().parent/model).resolve())
    return settings


def stat_record(path):
    stat = path.stat()
    return {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def contract_value(settings, native_root, native_config):
    native = json.loads((native_root/"contract.json").read_text())
    if native["config"] != native_config:
        raise ValueError("Native SR config does not match this pipeline's manifest/config")
    semantic = {k: v for k, v in settings.items() if k not in ("enabled", "python", "library_dirs", "output_dir")}
    versions = {name: importlib.metadata.version(name) for name in ("Pillow", "numpy")}
    detector_weight = None
    if settings["detector"]["enabled"]:
        detector_weight = stat_record(Path(settings["detector"]["model_path"]))
        versions.update({name: importlib.metadata.version(name) for name in ("torch", "ultralytics")})
    return {"algorithm": ALGORITHM, "settings": semantic, "versions": versions,
            "native_root": str(native_root), "native_contract": native,
            "detector_weight": detector_weight, "device": "cpu"}


def prepare_contract(root, contract, verify=False):
    path = root/"contract.json"
    if path.exists():
        if json.loads(path.read_text()) != contract:
            raise ValueError("Geometry contract changed; use a new geometry output directory")
    elif verify:
        raise FileNotFoundError("Geometry contract does not exist: " + str(path))
    else:
        if root.exists() and any(root.iterdir()):
            raise ValueError("Refusing a nonempty output directory without a geometry contract")
        p.write_json(path, contract)


def paths(root, row):
    output = root/row["dataset"]/row["output"]
    metadata = root/"metadata"/row["dataset"]/Path(row["output"]).with_suffix(".json")
    return output, metadata


def output_valid(path, row, settings):
    try:
        with Image.open(path) as im:
            im.load()
            if im.format != "PNG" or im.mode != "RGB" or im.size != (settings["target_width"], settings["target_height"]):
                return False
            if row["modality"] == "ir":
                pixels = np.asarray(im)
                return bool(np.array_equal(pixels[..., 0], pixels[..., 1]) and np.array_equal(pixels[..., 0], pixels[..., 2]))
        return True
    except (OSError, ValueError):
        return False


def completed(root, native_path, row, settings):
    output, marker = paths(root, row)
    try:
        metadata = json.loads(marker.read_text())
        return (metadata["native_file"] == stat_record(native_path)
                and metadata["output_file"] == stat_record(output)
                and metadata["geometry"]["algorithm"] == ALGORITHM
                and output_valid(output, row, settings))
    except (OSError, ValueError, KeyError):
        return False


def process_one(row, native_root, root, config, settings, detector):
    native_path = native_root/row["dataset"]/row["output"]
    if not p.output_valid(native_path, row):
        raise ValueError("Native x4 input missing/invalid (run SR first): " + str(native_path))
    if completed(root, native_path, row, settings):
        p.ensure_aliases(root, row)
        return {"status": "skipped"}
    started = time.perf_counter()
    native_stat = stat_record(native_path)
    with Image.open(native_path) as im:
        source = im.convert("RGB")
    detections, detector_status = detector.detect(source)
    result, geometry = fit_with_background(source, settings, detections, detector_status)
    result = ensure_ir_channels(result, row["modality"])
    output, marker = paths(root, row)
    p.save_output(output, np.asarray(result), config)
    if not output_valid(output, row, settings):
        raise RuntimeError("Geometry image validation failed: " + str(output))
    if native_stat != stat_record(native_path):
        raise RuntimeError("Native input changed while processing: " + str(native_path))
    elapsed = time.perf_counter()-started
    record = {"dataset": row["dataset"], "source": row["source"], "modality": row["modality"],
              "native_path": str(native_path), "native_file": native_stat,
              "output": row["output"], "output_file": stat_record(output),
              "seconds": elapsed, "geometry": geometry}
    p.write_json(marker, record)
    p.ensure_aliases(root, row)
    return {"status": "generated", "dataset": row["dataset"], "source": row["source"],
            "modality": row["modality"], "seconds": elapsed,
            "placement_reason": geometry["placement_reason"],
            "offset_changed_by_yolo": geometry["offset_changed_by_yolo"]}


def run_stage(config, settings, rows, native_root, root, available_only=False, limit=0, verify=False):
    native_root, root = Path(native_root).resolve(), Path(root).resolve()
    if root == native_root or root in native_root.parents or native_root in root.parents:
        raise ValueError("Native SR and geometry output directories must be separate, non-nested paths")
    contract = contract_value(settings, native_root, config)
    selected = rows
    unavailable = 0
    if available_only:
        selected = [r for r in rows if (native_root/r["dataset"]/r["output"]).is_file()]
        unavailable = len(rows)-len(selected)
    if limit:
        selected = selected[:limit]
    if not selected:
        raise ValueError("No native SR images selected; nothing generated")
    prepare_contract(root, contract, verify)
    counts = collections.Counter()
    started = time.perf_counter()
    detector = None
    if not verify:
        detector = CpuPersonDetector(settings["detector"], settings["cpu_threads"])
    for index, row in enumerate(selected):
        if verify:
            native_path = native_root/row["dataset"]/row["output"]
            valid = p.output_valid(native_path, row) and completed(root, native_path, row, settings)
            counts["valid" if valid else "invalid"] += 1
            canonical = root/row["dataset"]/row["output"]
            for alias in row["aliases"]:
                path = root/row["dataset"]/Path(alias).with_suffix(".png")
                if not path.is_symlink() or path.resolve() != canonical.resolve() or not canonical.is_file():
                    counts["invalid_alias"] += 1
        else:
            record = process_one(row, native_root, root, config, settings, detector)
            counts[record["status"]] += 1
            if record["status"] == "generated":
                counts[record["placement_reason"]] += 1
                counts["yolo_shifted"] += int(record["offset_changed_by_yolo"])
                with (root/"progress.jsonl").open("a") as stream:
                    stream.write(json.dumps(record)+"\n")
        if (index+1) % 50 == 0:
            print(json.dumps({"processed": index+1, "selected": len(selected), "counts": counts}), flush=True)
    summary = {"selected": len(selected), "manifest_total": len(rows), "counts": dict(counts),
               "unavailable_native_inputs": unavailable, "available_only": available_only,
               "limited": bool(limit), "complete_selected": not (counts["invalid"] or counts["invalid_alias"]),
               "elapsed_seconds": time.perf_counter()-started, "output_root": str(root), "device": "cpu"}
    if not verify:
        p.write_json(root/"last_run.json", summary)
    print(json.dumps(summary), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(p.HERE/"config.json"))
    parser.add_argument("--geometry-config", default=str(p.HERE/"geometry_config.json"))
    parser.add_argument("--geometry", choices=("on", "off"))
    parser.add_argument("--datasets", nargs="+", choices=("sysu", "regdb", "llcm"))
    parser.add_argument("--source-root")
    parser.add_argument("--output-root")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--available-only", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    settings = load_settings(args.geometry_config)
    enabled = settings["enabled"] if args.geometry is None else args.geometry == "on"
    if not enabled and not args.verify:
        print(json.dumps({"geometry": "off", "generated": 0}))
        return
    if args.limit < 0:
        parser.error("limit must be nonnegative")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = str(settings["cpu_threads"])
    config = p.load_config(args.config)
    rows = p.load_rows(config, args.datasets)
    result = run_stage(config, settings, rows, args.source_root or config["output_dir"],
                       args.output_root or settings["output_dir"], args.available_only, args.limit, args.verify)
    if not result["complete_selected"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
