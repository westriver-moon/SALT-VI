#!/usr/bin/env python3
"""Build a deterministic GPU0 pilot set with one SwinIR pass and cached SAM."""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
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

from fast_components import CachedSamBackend  # noqa: E402
from qwen_imagination.regional.config import (  # noqa: E402
    load_regional_config,
)
from qwen_imagination.regional.roi import (  # noqa: E402
    HumanROIGenerator,
    SCHPLIPBackend,
    SegmentAnythingBackend,
    UltralyticsPoseBackend,
)
from qwen_imagination.regional.runtime import (  # noqa: E402
    OfficialSwinIRBackend,
)
from qwen_imagination.regional.sysu_sources import (  # noqa: E402
    load_train_source_records,
)
from qwen_imagination.regional.tta import (  # noqa: E402
    blur_information,
    restore_tta_set,
    swin_instability,
)


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def sample_sources(config, per_camera: int, seed: int):
    rng = random.Random(seed)
    selected = []
    for modality in config.modalities:
        groups = {}
        for record in load_train_source_records(config.dataset_root, modality):
            groups.setdefault(record.camera, []).append(record)
        for camera, records in sorted(groups.items()):
            records = list(records)
            rng.shuffle(records)
            selected.extend((modality, item) for item in records[:per_camera])
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--per-camera", type=int, default=2)
    parser.add_argument("--tta-samples", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--artifact-name", default="pilot_base")
    parser.add_argument("--metrics-name", default="prepare_pilot.json")
    args = parser.parse_args()

    config = load_regional_config(args.config)
    run_root = args.run_root.expanduser().resolve()
    output_root = run_root / "artifacts" / args.artifact_name
    metrics_path = run_root / "metrics" / args.metrics_name
    selected = sample_sources(config, args.per_camera, args.seed)

    started = time.perf_counter()
    swin = OfficialSwinIRBackend(
        config.swinir["root"], config.assets["swinir_model"].path, args.device
    )
    swin_load_seconds = time.perf_counter() - started

    roi_device = args.device
    pose = UltralyticsPoseBackend(config.assets["yolo_pose"].path, device=roi_device)
    parsing = SCHPLIPBackend(
        config.roi["schp_root"], config.assets["schp_lip"].path, device=roi_device
    )
    base_sam = SegmentAnythingBackend(
        config.assets["sam_vit_b"].path,
        device=roi_device,
        repository=config.roi["sam_root"],
    )
    cached_sam = CachedSamBackend(base_sam)
    roi = HumanROIGenerator(pose=pose, parsing=parsing, sam=cached_sam, strict=True)

    records = []
    for index, (modality, source) in enumerate(selected):
        source_path = config.dataset_root / source.source_key
        artifact_dir = output_root / Path(source.source_key).parent / Path(source.source_key).stem
        artifact_dir.mkdir(parents=True, exist_ok=True)
        with Image.open(source_path) as image:
            lr = image.convert("RGB").resize(
                (config.source_size_hw[1], config.source_size_hw[0]),
                Image.Resampling.BICUBIC,
            )
        if modality == "ir":
            lr = lr.convert("L").convert("RGB")
        lr.save(artifact_dir / "lr.png", compress_level=2)

        tick = time.perf_counter()
        reference = swin.restore(lr, modality).convert("RGB")
        swin_seconds = time.perf_counter() - tick
        reference.save(artifact_dir / "swin.png", compress_level=2)

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

        tta_seconds = None
        tta_values = []
        if index < args.tta_samples:
            tick = time.perf_counter()
            _, variants = restore_tta_set(
                swin, lr, modality, reference=reference
            )
            tta_seconds = time.perf_counter() - tick
            tta_values = [
                float(swin_instability(reference, variants, region.mask))
                for region in regions
            ]

        region_rows = []
        for region in regions:
            mask_path = artifact_dir / "regions" / f"{region.region_id}.png"
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(
                np.asarray(region.mask, dtype=np.uint8) * 255, mode="L"
            ).save(mask_path, compress_level=2)
            region_rows.append(
                {
                    "region_id": region.region_id,
                    "category": region.category,
                    "bbox_xyxy": list(region.bbox_xyxy),
                    "side": region.side,
                    "mask": str(mask_path.relative_to(run_root)),
                    "u_blur": float(blur_information(reference, region.mask)),
                }
            )
        record = {
            "index": index,
            "source_key": source.source_key,
            "source": str(source_path),
            "identity": source.identity,
            "camera": source.camera,
            "modality": modality,
            "lr": str((artifact_dir / "lr.png").relative_to(run_root)),
            "swin": str((artifact_dir / "swin.png").relative_to(run_root)),
            "regions": region_rows,
            "timing_seconds": {
                "swin": swin_seconds,
                "roi": roi_seconds,
                "tta12_extra": tta_seconds,
            },
            "sam": {
                "set_image_calls": cached_sam.set_image_calls - before_set,
                "predict_calls": cached_sam.predict_calls - before_predict,
            },
            "pose_fallback": pose_fallback,
            "tta_u_swin": tta_values,
        }
        atomic_json(artifact_dir / "record.json", record)
        records.append(record)
        print(
            json.dumps(
                {
                    "index": index + 1,
                    "total": len(selected),
                    "source": source.source_key,
                    "regions": len(regions),
                    "swin_seconds": round(swin_seconds, 3),
                    "roi_seconds": round(roi_seconds, 3),
                },
                separators=(",", ":"),
            ),
            flush=True,
        )

    total_seconds = time.perf_counter() - started
    payload = {
        "schema_version": 1,
        "experiment": "qri-fast-search-gpu0-20260821",
        "device": args.device,
        "git_revision": subprocess.check_output(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "sample_count": len(records),
        "swin_load_seconds": swin_load_seconds,
        "total_seconds": total_seconds,
        "records": records,
    }
    atomic_json(metrics_path, payload)
    print(json.dumps({"metrics": str(metrics_path), "sample_count": len(records)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
