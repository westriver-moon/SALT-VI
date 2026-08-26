#!/usr/bin/env python3
"""Build the scalable QRI base: batched SwinIR plus cached-SAM top ROIs."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
PLUGIN_ROOT = PROJECT_ROOT / "plugins" / "qwen_imagination"
for path in (PROJECT_ROOT, PROJECT_ROOT / "src", PLUGIN_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fast_components import CachedSamBackend  # noqa: E402
from qwen_imagination.regional.composite import atomic_mask, atomic_png  # noqa: E402
from qwen_imagination.regional.config import load_regional_config  # noqa: E402
from qwen_imagination.regional.roi import (  # noqa: E402
    HumanROIGenerator,
    SCHPLIPBackend,
    SegmentAnythingBackend,
    UltralyticsPoseBackend,
)
from qwen_imagination.regional.runtime import OfficialSwinIRBackend  # noqa: E402
from qwen_imagination.regional.sysu_sources import load_train_source_records  # noqa: E402
from qwen_imagination.regional.tta import blur_information  # noqa: E402
from salt_vi.utils.super_resolution.build_sysu_swinir_x2 import infer  # noqa: E402


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
    if modality == "ir":
        lr = lr.convert("L").convert("RGB")
    return lr


def chunks(items, size):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--top-regions", type=int, default=2)
    parser.add_argument("--per-modality-limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1 or args.top_regions < 1:
        raise ValueError("batch-size and top-regions must be positive")

    config = load_regional_config(args.config)
    output_root = args.output_root.expanduser().resolve()
    record_root = output_root / "records"
    swin_root = output_root / "swin"
    mask_root = output_root / "masks"
    output_root.mkdir(parents=True, exist_ok=True)

    load_started = time.perf_counter()
    swin = OfficialSwinIRBackend(
        config.swinir["root"], config.assets["swinir_model"].path, args.device
    )
    pose = UltralyticsPoseBackend(config.assets["yolo_pose"].path, device=args.device)
    parsing = SCHPLIPBackend(
        config.roi["schp_root"], config.assets["schp_lip"].path, device=args.device
    )
    cached_sam = CachedSamBackend(
        SegmentAnythingBackend(
            config.assets["sam_vit_b"].path,
            device=args.device,
            repository=config.roi["sam_root"],
        )
    )
    roi = HumanROIGenerator(pose=pose, parsing=parsing, sam=cached_sam, strict=True)
    load_seconds = time.perf_counter() - load_started

    summaries = []
    skipped = 0
    processed = 0
    started = time.perf_counter()
    for modality in config.modalities:
        records = list(load_train_source_records(config.dataset_root, modality))
        if args.per_modality_limit:
            records = records[: args.per_modality_limit]
        pending = []
        for source in records:
            record_path = record_root / Path(source.source_key).with_suffix(".json")
            if not args.overwrite and record_path.is_file():
                skipped += 1
                continue
            pending.append(source)

        for batch in chunks(pending, args.batch_size):
            lr_images = [
                source_lr(config, config.dataset_root / item.source_key, modality)
                for item in batch
            ]
            arrays = np.stack(
                [np.asarray(image, dtype=np.uint8) for image in lr_images]
            )
            tick = time.perf_counter()
            restored = infer(swin.model, arrays, modality, args.device)
            batch_swin_seconds = time.perf_counter() - tick
            for source, lr, restored_array in zip(batch, lr_images, restored):
                reference = Image.fromarray(restored_array, mode="RGB")
                reference_path = swin_root / Path(source.source_key).with_suffix(".png")
                atomic_png(reference_path, reference, compress_level=2)

                before_set = cached_sam.set_image_calls
                before_predict = cached_sam.predict_calls
                tick = time.perf_counter()
                pose_fallback = False
                try:
                    roi.strict = True
                    regions = roi.regions(reference, modality)
                except ValueError as error:
                    if "pose backend did not detect a person" not in str(error):
                        raise
                    pose_fallback = True
                    roi.strict = False
                    regions = roi.regions(reference, modality)
                finally:
                    roi.strict = True
                roi_seconds = time.perf_counter() - tick

                scored = sorted(
                    (
                        (float(blur_information(reference, region.mask)), region)
                        for region in regions
                    ),
                    key=lambda item: (-item[0], item[1].region_id),
                )[: args.top_regions]
                region_rows = []
                for score, region in scored:
                    mask_path = (
                        mask_root
                        / Path(source.source_key).parent
                        / Path(source.source_key).stem
                        / f"{region.region_id}.png"
                    )
                    atomic_mask(
                        mask_path,
                        Image.fromarray(
                            np.asarray(region.mask, dtype=np.uint8) * 255,
                            mode="L",
                        ),
                        compress_level=2,
                    )
                    region_rows.append(
                        {
                            "region_id": region.region_id,
                            "category": region.category,
                            "bbox_xyxy": list(region.bbox_xyxy),
                            "side": region.side,
                            "u_blur": score,
                            "mask": str(mask_path.relative_to(output_root)),
                        }
                    )

                record_path = record_root / Path(source.source_key).with_suffix(".json")
                row = {
                    "schema_version": 1,
                    "source_key": source.source_key,
                    "source": str(config.dataset_root / source.source_key),
                    "identity": source.identity,
                    "camera": source.camera,
                    "modality": modality,
                    "swin": str(reference_path.relative_to(output_root)),
                    "regions": region_rows,
                    "anchor_score": max((item["u_blur"] for item in region_rows), default=0.0),
                    "pose_fallback": pose_fallback,
                    "timing_seconds": {
                        "swin_batch_share": batch_swin_seconds / len(batch),
                        "roi": roi_seconds,
                    },
                    "sam": {
                        "set_image_calls": cached_sam.set_image_calls - before_set,
                        "predict_calls": cached_sam.predict_calls - before_predict,
                    },
                }
                atomic_json(record_path, row)
                summaries.append(row)
                processed += 1
                print(
                    json.dumps(
                        {
                            "processed": processed,
                            "source": source.source_key,
                            "modality": modality,
                            "regions": len(region_rows),
                            "fallback": pose_fallback,
                            "swin_share": round(batch_swin_seconds / len(batch), 4),
                            "roi": round(roi_seconds, 4),
                        },
                        separators=(",", ":"),
                    ),
                    flush=True,
                )

    elapsed = time.perf_counter() - started
    summary = {
        "schema_version": 1,
        "config": str(args.config),
        "device": args.device,
        "batch_size": args.batch_size,
        "top_regions": args.top_regions,
        "model_load_seconds": load_seconds,
        "processed": processed,
        "skipped": skipped,
        "elapsed_seconds": elapsed,
        "mean_seconds_per_processed": elapsed / max(1, processed),
        "pose_fallbacks": sum(item["pose_fallback"] for item in summaries),
        "sam_set_image_calls_valid": all(
            item["sam"]["set_image_calls"] == 1 for item in summaries
        ),
    }
    atomic_json(output_root / "metrics" / "phase_a_summary.json", summary)
    print(json.dumps(summary, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
