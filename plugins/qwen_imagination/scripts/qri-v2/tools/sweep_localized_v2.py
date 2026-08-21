#!/usr/bin/env python3
"""Sweep focused PASD strength on one eye ROI without reloading the model."""

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
    feathered = soft_mask(mask_array, dilation_px=4, feather_px=3.0)
    args.output_root.mkdir(parents=True, exist_ok=True)
    variants = (
        ("focused_c050", 1.25, 9.0, 0.50),
        ("focused_c030", 1.10, 10.0, 0.30),
        ("focused_c010", 1.10, 12.0, 0.10),
    )
    caption = (
        "thin black rectangular eyeglasses, clear frames around both eyes, visible bridge "
        "and temple arms, same face, photorealistic"
    )
    added = "crisp local eyewear detail, surveillance photograph"
    negative = (
        "no glasses, bare eyes, missing eyeglass frames, different person, distorted face, "
        "blur, cartoon, painting"
    )
    backend = ExistingPASDBackend(args.pasd_config, device=args.device)
    baseline = np.asarray(reference, dtype=np.float32)
    results = []
    for index, (name, context, guidance, conditioning) in enumerate(variants):
        crop_box = roi_crop_box(
            bbox, reference.size, context_scale=context, target_size=(256, 512)
        )
        control = roi_control_image(reference, crop_box, (256, 512))
        control_path = args.output_root / f"{name}_control.png"
        control.save(control_path)
        options = PASDGenerationOptions(
            guidance_scale=guidance,
            conditioning_scale=conditioning,
            added_prompt=added,
            negative_prompt=negative,
        )
        generation = backend.generate(
            control_path,
            [caption],
            [20260820 + index * 1009],
            "rgb",
            options=options,
        )
        validate_qri_pasd_generation(generation, (256, 512))
        generated = generation.images[0]
        generated.save(args.output_root / f"{name}_pasd.png")
        composite = paste_roi_realization(reference, generated, crop_box, feathered)
        composite.save(args.output_root / f"{name}_composite.png")
        change = np.abs(np.asarray(composite, dtype=np.float32) - baseline)
        results.append(
            {
                "name": name,
                "context_scale": context,
                "guidance_scale": guidance,
                "conditioning_scale": conditioning,
                "crop_box_xyxy": list(crop_box),
                "eye_mean_abs_change": float(change[mask_array].mean()),
                "eye_p95_abs_change": float(np.percentile(change[mask_array], 95)),
                "eye_fraction_changed_gt10": float(
                    (change[mask_array].mean(axis=1) > 10).mean()
                ),
            }
        )
    payload = {
        "bbox_xyxy": list(bbox),
        "caption": caption,
        "added_prompt": added,
        "negative_prompt": negative,
        "variants": results,
    }
    (args.output_root / "sweep.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
