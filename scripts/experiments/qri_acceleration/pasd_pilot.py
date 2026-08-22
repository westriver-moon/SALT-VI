#!/usr/bin/env python3
"""Run a tightly bounded PASD speed/quality pilot on GPU0."""

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
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
PLUGIN_ROOT = PROJECT_ROOT / "plugins" / "qwen_imagination"
for path in (PROJECT_ROOT, PROJECT_ROOT / "src", PLUGIN_ROOT, SCRIPT_DIR):
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
from qwen_imagination.regional.qwen_v2 import (  # noqa: E402
    V2_NON_EDIT_STATES,
)


VARIANTS = (
    {"name": "p512_s20", "process_size": 512, "steps": 20},
    {"name": "p256_s12", "process_size": 256, "steps": 12},
    {"name": "p256_s8", "process_size": 256, "steps": 8},
)


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def first_positive(plan: dict):
    for world in plan["worlds"]:
        for region_id, candidate in world.items():
            if candidate["state"] not in V2_NON_EDIT_STATES:
                return region_id, candidate
    for region_id, candidates in plan["regions"].items():
        for candidate in candidates:
            if candidate["state"] not in V2_NON_EDIT_STATES:
                return region_id, candidate
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()
    run_root = args.run_root.expanduser().resolve()
    base = json.loads((run_root / "metrics" / "prepare_pilot.json").read_text())
    qwen = json.loads((run_root / "metrics" / "qwen_pilot.json").read_text())
    base_by_key = {item["source_key"]: item for item in base["records"]}

    config = PluginConfig.from_yaml(args.config)
    config.device = args.device
    # The formal preflight owns asset verification. This bounded speed pilot
    # deliberately avoids a second full-file checksum pass.
    config.validate_assets = lambda: None
    config.process_size = 512
    config.num_inference_steps = 20
    load_started = time.perf_counter()
    generator = PASDGenerator(config)
    load_seconds = time.perf_counter() - load_started

    results = []
    for sample_index, qrow in enumerate(qwen["records"][: args.limit]):
        match = first_positive(qrow["one_shot"])
        if match is None:
            continue
        region_id, candidate = match
        base_row = base_by_key[qrow["source_key"]]
        region_row = next(
            item for item in base_row["regions"] if item["region_id"] == region_id
        )
        lr = Image.open(run_root / base_row["lr"]).convert("RGB")
        reference = Image.open(run_root / base_row["swin"]).convert("RGB")
        mask_array = np.asarray(
            Image.open(run_root / region_row["mask"]).convert("L")
        ) > 0
        mask = soft_mask(mask_array, dilation_px=4, feather_px=3.0)
        crop_box = roi_crop_box(
            tuple(region_row["bbox_xyxy"]),
            reference.size,
            context_scale=1.25,
            target_size=(256, 512),
        )
        control = roi_control_image(reference, crop_box, (256, 512))
        sample_root = (
            run_root
            / "artifacts"
            / "pasd_pilot"
            / f"sample_{sample_index:02d}_{region_id}"
        )
        sample_root.mkdir(parents=True, exist_ok=True)
        control_path = sample_root / "control.png"
        control.save(control_path, compress_level=2)
        caption = (
            f"same person, {region_row['category']} at {region_id}: "
            f"{candidate['value']}, crisp visible structure, photorealistic surveillance image"
        )
        for variant in VARIANTS:
            generator.config.process_size = int(variant["process_size"])
            generator.config.num_inference_steps = int(variant["steps"])
            torch.cuda.empty_cache()
            tick = time.perf_counter()
            generated, geometry = generator.generate_views(
                control_path,
                [caption],
                [20260821 + sample_index],
                modality=base_row["modality"],
                batch_size=1,
                added_prompt=(
                    "crisp localized detail, preserve identity and surrounding structure"
                ),
                negative_prompts=[
                    "different person, changed identity, changed pose, changed body proportions, "
                    "changes outside the target region, distorted anatomy, blurry, cartoon, painting"
                ],
                guidance_scale=9.0,
                conditioning_scale=0.5,
            )
            seconds = time.perf_counter() - tick
            generated_crop = generated[0].convert("RGB")
            composite = paste_roi_realization(
                reference, generated_crop, crop_box, mask
            )
            variant_root = sample_root / variant["name"]
            variant_root.mkdir(parents=True, exist_ok=True)
            generated_crop.save(variant_root / "pasd.png", compress_level=2)
            composite.save(variant_root / "composite.png", compress_level=2)
            metrics = mask_metrics(reference, composite, mask)
            metrics["lr_cycle_energy"] = float(
                lr_cycle_energy(composite, lr, base_row["modality"])
            )
            result = {
                "source_key": base_row["source_key"],
                "region_id": region_id,
                "category": region_row["category"],
                "candidate": candidate,
                "caption": caption,
                "variant": variant,
                "seconds": seconds,
                "metrics": metrics,
                "crop_box_xyxy": list(crop_box),
                "geometry": geometry,
                "composite": str((variant_root / "composite.png").relative_to(run_root)),
            }
            results.append(result)
            print(
                json.dumps(
                    {
                        "source": base_row["source_key"],
                        "variant": variant["name"],
                        "seconds": round(seconds, 3),
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
        "device": args.device,
        "model_load_seconds": load_seconds,
        "records": results,
    }
    path = run_root / "metrics" / "pasd_pilot.json"
    atomic_json(path, payload)
    print(json.dumps({"metrics": str(path), "record_count": len(results)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
