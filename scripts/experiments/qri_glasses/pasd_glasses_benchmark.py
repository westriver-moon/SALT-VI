#!/usr/bin/env python3
"""Run a bounded multi-seed PASD eyewear recovery benchmark on one SYSU image."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
PLUGIN_ROOT = PROJECT_ROOT / "plugins" / "qwen_imagination"
QRI_ACCELERATION = PROJECT_ROOT / "scripts" / "experiments" / "qri_acceleration"
for path in (PROJECT_ROOT, PROJECT_ROOT / "src", PLUGIN_ROOT, QRI_ACCELERATION):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fast_components import mask_metrics  # noqa: E402
from pasd_plugin.config import PluginConfig  # noqa: E402
from pasd_plugin.runtime import PASDGenerator  # noqa: E402
from qwen_imagination.regional.composite import (  # noqa: E402
    lr_cycle_energy,
    paste_roi_realization,
    roi_control_image,
    roi_crop_box,
    soft_mask,
)


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-config", required=True, type=Path)
    parser.add_argument("--pasd-config", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    spec = yaml.safe_load(args.benchmark_config.read_text(encoding="utf-8"))
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    reference = Image.open(spec["inputs"]["swin_reference"]).convert("RGB")
    lr = Image.open(spec["inputs"]["source"]).convert("RGB").resize(
        (reference.width // 2, reference.height // 2), Image.Resampling.BICUBIC
    )
    archived_mask = np.asarray(
        Image.open(spec["inputs"]["eye_mask"]).convert("L")
    ) > 0
    mask = soft_mask(
        archived_mask,
        dilation_px=int(spec["mask"]["dilation_px"]),
        feather_px=float(spec["mask"]["feather_px"]),
    )
    bbox = tuple(int(value) for value in spec["inputs"]["eye_bbox_xyxy"])

    config = PluginConfig.from_yaml(args.pasd_config)
    config.device = args.device
    config.process_size = int(spec["pasd"]["process_size"])
    config.num_inference_steps = int(spec["pasd"]["steps"])
    config.validate_assets = lambda: None
    load_started = time.perf_counter()
    generator = PASDGenerator(config)
    load_seconds = time.perf_counter() - load_started

    records = []
    for variant in spec["pasd"]["variants"]:
        crop_box = roi_crop_box(
            bbox,
            reference.size,
            context_scale=float(variant["context_scale"]),
            target_size=(256, 512),
        )
        control = roi_control_image(reference, crop_box, (256, 512))
        variant_root = output_root / variant["name"]
        variant_root.mkdir(parents=True, exist_ok=True)
        control_path = variant_root / "control.png"
        control.save(control_path, compress_level=2)
        seeds = [int(value) for value in spec["seeds"]]
        captions = [str(spec["prompt"]["positive"])] * len(seeds)
        negatives = [str(spec["prompt"]["negative"])] * len(seeds)
        torch.cuda.empty_cache()
        tick = time.perf_counter()
        generated, geometry = generator.generate_views(
            control_path,
            captions,
            seeds,
            modality="rgb",
            batch_size=min(len(seeds), int(spec["pasd"]["batch_size"])),
            added_prompt=str(spec["prompt"]["added"]),
            negative_prompts=negatives,
            guidance_scale=float(variant["guidance_scale"]),
            conditioning_scale=float(variant["conditioning_scale"]),
        )
        elapsed = time.perf_counter() - tick
        for seed, generated_crop in zip(seeds, generated):
            composite = paste_roi_realization(
                reference, generated_crop.convert("RGB"), crop_box, mask
            )
            generated_path = variant_root / f"seed_{seed}_pasd.png"
            composite_path = variant_root / f"seed_{seed}_composite.png"
            generated_crop.save(generated_path, compress_level=2)
            composite.save(composite_path, compress_level=2)
            metrics = mask_metrics(reference, composite, mask)
            metrics["lr_cycle_energy"] = float(lr_cycle_energy(composite, lr, "rgb"))
            records.append(
                {
                    "backend": "pasd_sd15",
                    "variant": variant,
                    "seed": seed,
                    "seconds_batch_share": elapsed / len(seeds),
                    "crop_box_xyxy": list(crop_box),
                    "metrics": metrics,
                    "geometry": geometry,
                    "generated": str(generated_path),
                    "composite": str(composite_path),
                }
            )
            print(
                json.dumps(
                    {
                        "variant": variant["name"],
                        "seed": seed,
                        "seconds": round(elapsed / len(seeds), 3),
                        "inside_change": round(metrics["inside_mean_abs_change"], 3),
                    },
                    separators=(",", ":"),
                ),
                flush=True,
            )

    del generator
    gc.collect()
    torch.cuda.empty_cache()
    payload = {
        "schema_version": 1,
        "benchmark": "qri-glasses-gpu0-20260821",
        "backend": "pasd_sd15",
        "model_load_seconds": load_seconds,
        "prompt": spec["prompt"],
        "records": records,
    }
    path = output_root / "metrics.json"
    atomic_json(path, payload)
    print(json.dumps({"metrics": str(path), "record_count": len(records)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
