#!/usr/bin/env python3
"""Run one real localized QRI-v2 PASD realization for visual validation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", "--plugin-root", dest="plugin_root", type=Path)
    parser.add_argument("--pasd-config", required=True, type=Path)
    parser.add_argument("--swin", required=True, type=Path)
    parser.add_argument("--mask", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument(
        "--value",
        default=(
            "thin dark eyeglass frames around both eyes with a visible bridge "
            "and symmetric temple arms"
        ),
    )
    args = parser.parse_args()
    repo = (args.plugin_root or Path(__file__).resolve().parents[3]).resolve()
    if not (repo / "qwen_imagination").is_dir():
        repo = Path(__file__).resolve().parents[3]
    salt_root = repo.parents[1]
    sys.path[:0] = [str(repo), str(salt_root / "src"), str(salt_root)]

    from qwen_imagination.regional.composite import (
        paste_roi_realization,
        roi_control_image,
        roi_crop_box,
        soft_mask,
    )
    from qwen_imagination.regional.runtime import (
        ExistingPASDBackend,
        PASDGenerationOptions,
        validate_qri_pasd_generation,
    )

    reference = Image.open(args.swin).convert("RGB")
    mask_array = np.asarray(Image.open(args.mask).convert("L")) > 0
    ys, xs = np.where(mask_array)
    bbox = (int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1))
    crop_box = roi_crop_box(
        bbox, reference.size, context_scale=1.75, target_size=(256, 512)
    )
    control = roi_control_image(reference, crop_box, (256, 512))
    args.output_root.mkdir(parents=True, exist_ok=True)
    control_path = args.output_root / "eye_roi_control.png"
    control.save(control_path)

    options = PASDGenerationOptions(
        guidance_scale=7.0,
        conditioning_scale=0.75,
        added_prompt=(
            "high-detail localized semantic realization, make the requested detail crisp "
            "and recognizable at surveillance scale, preserve the same person and "
            "surrounding observed structure"
        ),
        negative_prompt=(
            "different person, changed identity, changed pose, changed body proportions, "
            "different clothing, changes outside the requested region, unrequested "
            "accessories outside the requested region, duplicated object, distorted anatomy, "
            "painting, cartoon, artificial texture, blurry, noise, raster lines, over-smoothed"
        ),
    )
    caption = (
        "same pedestrian and same surveillance frame; edit only the target eyes region "
        "(eyewear); clearly realize exactly this plausible hypothesis: "
        f"{args.value}; make its defining shape and boundaries visually legible; preserve "
        "all surrounding identity, pose, clothing and anatomy"
    )
    backend = ExistingPASDBackend(args.pasd_config, device=args.device)
    generation = backend.generate(
        control_path, [caption], [20260820], "rgb", options=options
    )
    validate_qri_pasd_generation(generation, (256, 512))
    generated = generation.images[0]
    generated_path = args.output_root / "eye_roi_pasd.png"
    generated.save(generated_path)
    feathered = soft_mask(mask_array, dilation_px=4, feather_px=3.0)
    feathered.save(args.output_root / "eye_soft_mask.png")
    composite = paste_roi_realization(reference, generated, crop_box, feathered)
    composite_path = args.output_root / "eye_composite.png"
    composite.save(composite_path)

    baseline = np.asarray(reference, dtype=np.float32)
    realized = np.asarray(composite, dtype=np.float32)
    change = np.abs(realized - baseline)
    payload = {
        "bbox_xyxy": list(bbox),
        "crop_box_xyxy": list(crop_box),
        "value": args.value,
        "caption": caption,
        "sampling": options.manifest(),
        "geometry": generation.geometry,
        "eye_mean_abs_change": float(change[mask_array].mean()),
        "eye_p95_abs_change": float(np.percentile(change[mask_array], 95)),
        "eye_fraction_changed_gt10": float(
            (change[mask_array].mean(axis=1) > 10).mean()
        ),
        "outputs": {
            "control": str(control_path),
            "pasd": str(generated_path),
            "mask": str(args.output_root / "eye_soft_mask.png"),
            "composite": str(composite_path),
        },
    }
    (args.output_root / "validation.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
