"""Materialize the canonical person-fit assets from the audited YOLO26 gate."""

import argparse
import collections
import concurrent.futures
import json
import os
import re
from pathlib import Path

from PIL import Image
import yaml

from .geometry import render


DATASETS = ("sysu", "regdb", "llcm")


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


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
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def bind_contract(path, contract):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != contract:
            raise ValueError("output contract changed; choose a new output directory: " + str(path))
    else:
        write_json(path, contract)


def _valid_bbox(box, width, height):
    return (
        isinstance(box, list)
        and len(box) == 4
        and 0 <= box[0] < box[2] <= width + 1e-3
        and 0 <= box[1] < box[3] <= height + 1e-3
    )


def unsafe_partial_box(row, policy):
    """Reject the empirically observed upper-body-only failure family."""
    if row.get("final_decision") != "pass":
        return False
    box = row["annotation_bbox"]
    height = float(row["height"])
    top_ratio = box[1] / height
    bottom_ratio = box[3] / height
    evidence = (
        "missing_lower_body_evidence" in row.get("risk_reasons", [])
        or "schp_component_bbox_correction" in row.get("schp_risks", [])
    )
    return (
        evidence
        and top_ratio <= float(policy["partial_box_max_top_ratio"])
        and bottom_ratio < float(policy["partial_box_min_bottom_ratio"])
    )


def validate_gate(inventory, gate_rows, datasets, config):
    inventory_keys = {
        (row["dataset"], row["source"])
        for row in inventory
        if row["dataset"] in datasets
    }
    gate_keys = set()
    counts = collections.defaultdict(collections.Counter)
    policy = config["quality_policy"]
    for row in gate_rows:
        if row["dataset"] not in datasets:
            continue
        key = (row["dataset"], row["source_key"])
        if key in gate_keys:
            raise ValueError("duplicate quality-gate key: " + repr(key))
        gate_keys.add(key)
        decision = row.get("final_decision")
        if decision not in {"pass", "fallback"}:
            raise ValueError("invalid final decision: " + repr(key))
        if decision == "pass":
            if not _valid_bbox(row.get("annotation_bbox"), row["width"], row["height"]):
                raise ValueError("invalid pass bbox: " + repr(key))
            if unsafe_partial_box(row, policy):
                raise ValueError("unsafe partial-person box remains in pass set: " + repr(key))
        counts[row["dataset"]][decision] += 1
        counts[row["dataset"]]["total"] += 1
    if gate_keys != inventory_keys:
        missing = inventory_keys - gate_keys
        extra = gate_keys - inventory_keys
        raise ValueError(f"gate/inventory mismatch: missing={len(missing)} extra={len(extra)}")
    expected = config.get("expected_counts", {})
    for dataset in datasets:
        if dataset in expected and dict(counts[dataset]) != expected[dataset]:
            raise ValueError(
                f"{dataset} count mismatch: {dict(counts[dataset])} != {expected[dataset]}"
            )
    return counts


def _source_metadata(row):
    source_key = row["source"]
    match = re.search(r"(?:^|/|_)(?:cam|c)(\d+)(?:/|_)", source_key)
    camera = row["modality"] if row["dataset"] == "regdb" else (
        "cam" + str(int(match.group(1))) if match else ""
    )
    return {
        "dataset": row["dataset"],
        "source_key": source_key,
        "identity": Path(source_key).parent.name,
        "camera": camera,
        "modality": row["modality"],
        "references": row.get("references", []),
        "aliases": row.get("aliases", []),
    }


def _materialize_one(task):
    inventory_row, gate_row, output_root, size_hw, margin, blur_radius = task
    dataset = inventory_row["dataset"]
    relative = Path(inventory_row["output"])
    image_relative = Path("images") / relative
    image_path = Path(output_root) / dataset / image_relative
    native_path = Path(gate_row["native_image"])
    with Image.open(native_path) as source:
        source = source.convert("RGB")
    if source.size != (int(gate_row["width"]), int(gate_row["height"])):
        raise ValueError(f"native size mismatch: {native_path}: {source.size}")
    if gate_row["final_decision"] == "pass":
        bbox = [float(value) for value in gate_row["annotation_bbox"]]
        localization_status = "detected_person"
    else:
        bbox = [0.0, 0.0, float(source.width), float(source.height)]
        localization_status = "fallback_full_frame"
    if not image_path.is_file():
        output, geometry = render(
            source, "person_fit", size_hw, bbox, margin, blur_radius
        )
        image_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = image_path.with_suffix(image_path.suffix + f".tmp-{os.getpid()}")
        output.save(temporary, format="PNG", compress_level=3)
        os.replace(temporary, image_path)
    else:
        with Image.open(image_path) as prepared:
            if prepared.size != (int(size_hw[1]), int(size_hw[0])):
                raise ValueError("existing prepared image has wrong size: " + str(image_path))
        _output, geometry = render(
            source, "person_fit", size_hw, bbox, margin, blur_radius
        )
    record = _source_metadata(inventory_row)
    record.update(
        mode="person_fit",
        native_image=str(native_path),
        image=image_relative.as_posix(),
        status="complete",
        geometry=geometry,
        localization={
            "status": localization_status,
            "bbox": bbox,
            "final_reason": gate_row.get("final_reason"),
            "annotation_source": gate_row.get("annotation_source"),
            "audit_id": gate_row.get("audit_id"),
        },
        quality_gate={
            "final_decision": gate_row["final_decision"],
            "machine_decision": gate_row.get("machine_decision"),
            "risk_reasons": gate_row.get("risk_reasons", []),
            "schp_risks": gate_row.get("schp_risks", []),
        },
    )
    return record


def materialize(config, datasets=DATASETS, workers=16):
    datasets = tuple(datasets)
    inventory = [row for row in read_jsonl(config["input_manifest"]) if row["dataset"] in datasets]
    gate_rows = [row for row in read_jsonl(config["quality_gate"]) if row["dataset"] in datasets]
    validate_gate(inventory, gate_rows, datasets, config)
    gate_by_key = {(row["dataset"], row["source_key"]): row for row in gate_rows}
    output_root = Path(config["output_root"])
    size_hw = tuple(int(value) for value in config["size_hw"])
    tasks = [
        (
            row,
            gate_by_key[(row["dataset"], row["source"])],
            str(output_root),
            size_hw,
            float(config["person_margin"]),
            float(config["blur_radius"]),
        )
        for row in inventory
    ]
    records = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        for record in executor.map(_materialize_one, tasks, chunksize=32):
            records.append(record)
    summaries = []
    for dataset in datasets:
        selected = sorted(
            (row for row in records if row["dataset"] == dataset),
            key=lambda row: row["source_key"],
        )
        root = output_root / dataset
        contract = {
            "version": 1,
            "release": "v1_yolo26_quality_gate_current",
            "dataset": dataset,
            "size_hw": list(size_hw),
            "input_root": str(Path(config["input_root"]).resolve()),
            "input_manifest": str(Path(config["input_manifest"]).resolve()),
            "quality_gate": str(Path(config["quality_gate"]).resolve()),
            "models": config["models"],
            "quality_policy": config["quality_policy"],
            "geometry": "uniform_affine_person_crop_with_full_frame_fallback",
            "person_margin": float(config["person_margin"]),
            "blur_radius": float(config["blur_radius"]),
        }
        bind_contract(root / "contract.json", contract)
        write_jsonl(root / "images.jsonl", selected)
        decisions = collections.Counter(
            row["quality_gate"]["final_decision"] for row in selected
        )
        summary = {
            "dataset": dataset,
            "total": len(selected),
            "passed": decisions["pass"],
            "fallback": decisions["fallback"],
            "fallback_rate": decisions["fallback"] / len(selected),
            "complete_inventory": len(selected) == config["expected_counts"][dataset]["total"],
        }
        write_json(root / "images.summary.json", summary)
        summaries.append(summary)
    return summaries


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args(argv)
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    summaries = materialize(config, args.datasets, args.workers)
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    return int(not all(row["complete_inventory"] for row in summaries))


if __name__ == "__main__":
    raise SystemExit(main())
