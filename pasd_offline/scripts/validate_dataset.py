from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pasd_offline.contracts import load_dataset_scope, load_generation_identity  # noqa: E402
from pasd_offline.generate import consolidate_manifest, validate_source  # noqa: E402
from pasd_offline.config import GenerationConfig  # noqa: E402
from pasd_offline.tasks import group_tasks_by_source, load_tasks  # noqa: E402


def validate_geometry(metadata: dict, config: GenerationConfig) -> None:
    geometry = metadata.get("geometry")
    if not isinstance(geometry, dict):
        raise ValueError("missing geometry audit")
    source_width, source_height = geometry["source_size"]
    target_width, target_height = geometry["target_size"]
    resized_width, resized_height = geometry["resized_size"]
    scale = float(geometry["scale"])
    if [target_width, target_height] != [config.target_width, config.target_height]:
        raise ValueError("geometry target mismatch")
    if min(source_width, source_height, resized_width, resized_height, scale) <= 0:
        raise ValueError("non-positive geometry")
    if abs(resized_width - source_width * scale) > 1.01 or abs(resized_height - source_height * scale) > 1.01:
        raise ValueError("aspect ratio was stretched")
    detection = geometry.get("person_detection")
    expanded = geometry.get("expanded_person_bbox")
    if not isinstance(detection, dict) or len(detection.get("bbox_xyxy", ())) != 4 or len(expanded or ()) != 4:
        raise ValueError("missing person crop audit")

    if geometry["mode"] == "person_safe_cover_crop":
        left, top, right, bottom = geometry["crop_box"]
        if geometry["padding"] != [0, 0, 0, 0]:
            raise ValueError("crop mode has padding")
        if right - left != target_width or bottom - top != target_height:
            raise ValueError("crop size mismatch")
        if not (0 <= left <= right <= resized_width and 0 <= top <= bottom <= resized_height):
            raise ValueError("crop is outside resized image")
        x1, y1, x2, y2 = (float(value) * scale for value in expanded)
        if x1 < left - 1 or y1 < top - 1 or x2 > right + 1 or y2 > bottom + 1:
            raise ValueError("crop cuts the expanded person box")
    elif geometry["mode"] == "person_fit_edge_pad":
        if geometry["crop_box"] is not None:
            raise ValueError("padding mode has a crop")
        left, top, right, bottom = geometry["padding"]
        if min(left, top, right, bottom) < 0:
            raise ValueError("negative padding")
        if left + resized_width + right != target_width or top + resized_height + bottom != target_height:
            raise ValueError("padding size mismatch")
    else:
        raise ValueError(f"unknown geometry mode: {geometry['mode']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a fixed-size audited SYSU PASD dataset")
    parser.add_argument("--config", required=True)
    parser.add_argument("--records", required=True)
    args = parser.parse_args()
    config = GenerationConfig.from_yaml(args.config)
    load_generation_identity(config)
    load_dataset_scope(config)
    output_root = config.output_root
    tasks = load_tasks(args.records, "all", seed=0)
    groups = group_tasks_by_source(tasks)
    errors = []
    view_count = 0
    for group in groups:
        source_key = group[0].source_key
        try:
            metadata = validate_source(group, output_root, config)
            validate_geometry(metadata, config)
            view_count += len(metadata["views"])
        except (OSError, KeyError, ValueError, json.JSONDecodeError) as error:
            errors.append(f"invalid:{source_key}:{error}")
    expected_sources = len(groups)
    expected_views = expected_sources * config.views_per_source
    summary = {
        "expected_sources": expected_sources,
        "record_sources": len(groups),
        "valid_views": view_count,
        "expected_views": expected_views,
        "errors": errors[:1000],
        "error_count": len(errors),
        "complete": not errors and view_count == expected_views,
    }
    report = output_root / "validation-report.json"
    report.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["complete"]:
        raise SystemExit(1)
    consolidate_manifest(output_root, tasks, config, args.records)


if __name__ == "__main__":
    main()
