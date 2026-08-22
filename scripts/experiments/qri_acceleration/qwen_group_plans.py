#!/usr/bin/env python3
"""Run one short Qwen plan per identity-modality anchor and cache it."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
PLUGIN_ROOT = PROJECT_ROOT / "plugins" / "qwen_imagination"
for path in (PROJECT_ROOT, PROJECT_ROOT / "src", PLUGIN_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fast_components import OneShotQwenPlanner  # noqa: E402
from qwen_imagination.regional.config import load_regional_config  # noqa: E402
from qwen_imagination.regional.schema import Region  # noqa: E402


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def source_lr(config, source_path: Path, modality: str) -> Image.Image:
    with Image.open(source_path) as image:
        lr = image.convert("RGB").resize(
            (config.source_size_hw[1], config.source_size_hw[0]),
            Image.Resampling.BICUBIC,
        )
    return lr.convert("L").convert("RGB") if modality == "ir" else lr


def load_region(output_root: Path, row: dict) -> Region:
    mask = np.asarray(Image.open(output_root / row["mask"]).convert("L")) > 0
    return Region(
        row["region_id"],
        row["category"],
        tuple(int(value) for value in row["bbox_xyxy"]),
        mask,
        side=row.get("side"),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--endpoint", default="")
    parser.add_argument("--model-id", default="")
    parser.add_argument("--limit-groups", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_regional_config(args.config)
    output_root = args.output_root.expanduser().resolve()
    rows = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((output_root / "records").rglob("*.json"))
    ]
    anchors = {}
    for row in rows:
        key = (str(row["identity"]), row["modality"])
        if key not in anchors or (
            float(row["anchor_score"]), row["source_key"]
        ) > (float(anchors[key]["anchor_score"]), anchors[key]["source_key"]):
            anchors[key] = row
    groups = sorted(anchors.items())
    if args.limit_groups:
        groups = groups[: args.limit_groups]

    if not args.dry_run and (not args.endpoint or not args.model_id):
        parser.error("--endpoint and --model-id are required unless --dry-run is set")
    planner = None if args.dry_run else OneShotQwenPlanner(args.endpoint, args.model_id)
    processed = 0
    skipped = 0
    timings = []
    if args.dry_run:
        summary = {
            "schema_version": 1,
            "grouping": "identity_modality",
            "groups_discovered": len(anchors),
            "groups_selected": len(groups),
            "dry_run": True,
            "anchors": [
                {
                    "identity": identity,
                    "modality": modality,
                    "source_key": row["source_key"],
                    "anchor_score": row["anchor_score"],
                }
                for (identity, modality), row in groups
            ],
        }
        atomic_json(output_root / "metrics" / "phase_b_dry_run.json", summary)
        print(json.dumps(summary, separators=(",", ":")))
        return 0
    for (identity, modality), row in groups:
        plan_path = output_root / "group_plans" / modality / f"{identity}.json"
        if not args.overwrite and plan_path.is_file():
            skipped += 1
            continue
        lr = source_lr(config, Path(row["source"]), modality)
        swin = Image.open(output_root / row["swin"]).convert("RGB")
        regions = [load_region(output_root, item) for item in row["regions"]]
        tick = time.perf_counter()
        plan = planner.plan(lr, swin, regions, max_worlds=3, seed=processed)
        seconds = time.perf_counter() - tick
        payload = {
            "schema_version": 1,
            "identity": identity,
            "modality": modality,
            "anchor_source_key": row["source_key"],
            "anchor_score": row["anchor_score"],
            "seconds": seconds,
            "plan": plan,
        }
        atomic_json(plan_path, payload)
        timings.append(seconds)
        processed += 1
        print(
            json.dumps(
                {
                    "processed": processed,
                    "identity": identity,
                    "modality": modality,
                    "seconds": round(seconds, 3),
                    "regions": len(regions),
                    "worlds": len(plan["worlds"]),
                },
                separators=(",", ":"),
            ),
            flush=True,
        )

    summary = {
        "schema_version": 1,
        "grouping": "identity_modality",
        "groups_discovered": len(anchors),
        "groups_selected": len(groups),
        "processed": processed,
        "skipped": skipped,
        "mean_seconds": sum(timings) / max(1, len(timings)),
        "total_seconds": sum(timings),
    }
    atomic_json(output_root / "metrics" / "phase_b_summary.json", summary)
    print(json.dumps(summary, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
