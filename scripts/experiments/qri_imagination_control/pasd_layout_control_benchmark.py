#!/usr/bin/env python3
"""Test generic EditSpec layout hints through the repository's PASD ControlNet."""

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
from PIL import Image, ImageFilter


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
PLUGIN_ROOT = PROJECT_ROOT / "plugins" / "qwen_imagination"
for path in (PROJECT_ROOT, PROJECT_ROOT / "src", PLUGIN_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from edit_program import (  # noqa: E402
    feathered_region_map,
    harmonize_proposal,
    load_edit_program,
    rasterize_creation_map,
    rasterize_edit_region,
)
from pasd_plugin.config import PluginConfig  # noqa: E402
from pasd_plugin.runtime import PASDGenerator  # noqa: E402
from qwen_imagination.regional.composite import roi_control_image, roi_crop_box  # noqa: E402


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def image_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float32)


def mask_array(mask: Image.Image) -> np.ndarray:
    return np.asarray(mask.convert("L"), dtype=np.float32) / 255.0


def blend(reference: Image.Image, proposal: Image.Image, alpha: Image.Image) -> Image.Image:
    source = image_array(reference)
    target = image_array(proposal)
    weight = mask_array(alpha)[..., None]
    output = np.clip(source * (1.0 - weight) + target * weight, 0.0, 255.0)
    return Image.fromarray(np.round(output).astype(np.uint8), mode="RGB")


def weighted_delta(reference: Image.Image, candidate: Image.Image, mask: Image.Image) -> float:
    delta = np.abs(image_array(candidate) - image_array(reference)).mean(axis=2)
    weight = mask_array(mask)
    return float((delta * weight).sum() / max(float(weight.sum()), 1e-8))


def overlay_layout(
    reference: Image.Image,
    layout: Image.Image,
    color_rgb: list[int],
    opacity: float,
) -> Image.Image:
    source = image_array(reference)
    color = np.asarray(color_rgb, dtype=np.float32)[None, None, :]
    alpha = mask_array(layout)[..., None] * float(opacity)
    output = np.clip(source * (1.0 - alpha) + color * alpha, 0.0, 255.0)
    return Image.fromarray(np.round(output).astype(np.uint8), mode="RGB")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--pasd-config", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    spec = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if int(spec.get("schema_version", 0)) != 1:
        raise ValueError("benchmark config requires schema_version=1")
    reference = Image.open(spec["reference"]).convert("RGB")
    program = load_edit_program(Path(spec["edit_program"]))
    layout = rasterize_creation_map(program, reference.size)
    edit_region = rasterize_edit_region(program, reference.size)
    bbox = tuple(int(value) for value in spec["roi_bbox_xyxy"])

    semantic_control_full = None
    semantic_spec = spec.get("semantic_proposal")
    if semantic_spec:
        candidate = Image.open(semantic_spec["candidate"]).convert("RGB")
        candidate_box = tuple(int(value) for value in semantic_spec["crop_box_xyxy"])
        left, top, right, bottom = candidate_box
        proposal = reference.copy()
        proposal.paste(
            candidate.resize((right - left, bottom - top), Image.Resampling.LANCZOS),
            (left, top),
        )
        proposal_blur = float(semantic_spec.get("proposal_blur_px", 0.0))
        if proposal_blur > 0.0:
            proposal = proposal.filter(ImageFilter.GaussianBlur(proposal_blur))
        proposal = harmonize_proposal(
            reference,
            proposal,
            edit_region,
            strength=float(semantic_spec.get("harmonize_strength", 0.7)),
        )
        proposal_alpha = feathered_region_map(
            edit_region, float(semantic_spec.get("region_feather_px", 3.0))
        )
        proposal_max_alpha = float(semantic_spec.get("region_max_alpha", 1.0))
        proposal_alpha = Image.fromarray(
            np.round(
                np.asarray(proposal_alpha, dtype=np.float32) * proposal_max_alpha
            ).astype(np.uint8),
            mode="L",
        )
        semantic_control_full = blend(reference, proposal, proposal_alpha)

    pasd_config = PluginConfig.from_yaml(args.pasd_config)
    pasd_config.device = args.device
    pasd_config.geometry_mode = "direct_rewrite"
    pasd_config.process_size = int(spec["pasd"]["process_size"])
    pasd_config.num_inference_steps = int(spec["pasd"]["steps"])
    # Assets were already verified when the offline PASD package was staged.
    pasd_config.validate_assets = lambda: None
    load_started = time.perf_counter()
    generator = PASDGenerator(pasd_config)
    load_seconds = time.perf_counter() - load_started

    prompt = str(spec["prompt"]["positive"])
    added_prompt = str(spec["prompt"]["added"])
    negative_prompt = str(spec["prompt"]["negative"])
    positive_tokens = len(
        generator.pipeline.tokenizer(
            ", ".join((prompt, added_prompt)), add_special_tokens=True
        ).input_ids
    )
    if positive_tokens > int(generator.pipeline.tokenizer.model_max_length):
        raise ValueError(
            f"positive prompt has {positive_tokens} tokens; maximum is "
            f"{generator.pipeline.tokenizer.model_max_length}"
        )

    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    layout.save(output_root / "layout_anchor.png", compress_level=2)
    edit_region.save(output_root / "edit_region.png", compress_level=2)
    if semantic_control_full is not None:
        semantic_control_full.save(
            output_root / "semantic_proposal_control.png", compress_level=2
        )
    records = []
    for variant in spec["pasd"]["variants"]:
        name = str(variant["name"])
        seed = int(variant.get("seed", spec["seed"]))
        control_source = str(variant.get("control_source", "layout"))
        if control_source == "layout":
            control_full = overlay_layout(
                reference,
                layout,
                [int(value) for value in variant.get("layout_color_rgb", [18, 18, 18])],
                float(variant["layout_opacity"]),
            )
        elif control_source == "semantic_proposal":
            if semantic_control_full is None:
                raise ValueError("semantic_proposal control requested without proposal config")
            control_full = semantic_control_full
        else:
            raise ValueError("control_source must be layout or semantic_proposal")
        crop_box = roi_crop_box(
            bbox,
            reference.size,
            context_scale=float(variant["context_scale"]),
            target_size=(256, 512),
        )
        control = roi_control_image(control_full, crop_box, (256, 512))
        variant_root = output_root / name
        variant_root.mkdir(parents=True, exist_ok=True)
        control_path = variant_root / "control.png"
        control.save(control_path, compress_level=2)

        torch.cuda.empty_cache()
        started = time.perf_counter()
        generated, geometry = generator.generate_views(
            control_path,
            [prompt],
            [seed],
            modality="rgb",
            batch_size=1,
            added_prompt=added_prompt,
            negative_prompts=[negative_prompt],
            guidance_scale=float(variant["guidance_scale"]),
            conditioning_scale=float(variant["conditioning_scale"]),
        )
        elapsed = time.perf_counter() - started
        generated_crop = generated[0].convert("RGB")
        left, top, right, bottom = crop_box
        proposal = reference.copy()
        proposal.paste(
            generated_crop.resize((right - left, bottom - top), Image.Resampling.LANCZOS),
            (left, top),
        )
        blur_px = float(variant.get("proposal_blur_px", 0.0))
        if blur_px > 0.0:
            proposal = proposal.filter(ImageFilter.GaussianBlur(blur_px))
        proposal = harmonize_proposal(
            reference,
            proposal,
            edit_region,
            strength=float(variant.get("harmonize_strength", 0.75)),
        )
        alpha = feathered_region_map(
            edit_region, float(variant.get("region_feather_px", 3.0))
        )
        max_alpha = float(variant.get("region_max_alpha", 0.8))
        alpha = Image.fromarray(
            np.round(np.asarray(alpha, dtype=np.float32) * max_alpha).astype(np.uint8),
            mode="L",
        )
        composite = blend(reference, proposal, alpha)

        generated_path = variant_root / f"seed_{seed}_pasd.png"
        composite_path = variant_root / f"seed_{seed}_composite.png"
        generated_crop.save(generated_path, compress_level=2)
        alpha.save(variant_root / f"seed_{seed}_writeback_alpha.png", compress_level=2)
        composite.save(composite_path, compress_level=2)
        outside = Image.fromarray(
            np.round((1.0 - mask_array(edit_region)) * 255.0).astype(np.uint8), mode="L"
        )
        metrics = {
            "layout_mean_abs_change": weighted_delta(reference, composite, layout),
            "edit_region_mean_abs_change": weighted_delta(reference, composite, edit_region),
            "outside_edit_region_mean_abs_change": weighted_delta(reference, composite, outside),
        }
        records.append(
            {
                "variant": variant,
                "seed": seed,
                "seconds": elapsed,
                "crop_box_xyxy": list(crop_box),
                "geometry": geometry,
                "control": str(control_path),
                "generated": str(generated_path),
                "composite": str(composite_path),
                "metrics": metrics,
            }
        )
        print(
            json.dumps(
                {
                    "variant": name,
                    "seconds": round(elapsed, 3),
                    "layout_change": round(metrics["layout_mean_abs_change"], 3),
                    "outside_change": round(
                        metrics["outside_edit_region_mean_abs_change"], 4
                    ),
                },
                separators=(",", ":"),
            ),
            flush=True,
        )

    del generator
    gc.collect()
    torch.cuda.empty_cache()
    atomic_json(
        output_root / "metrics.json",
        {
            "schema_version": 1,
            "benchmark": spec["experiment_id"],
            "backend": "pasd-layout-control",
            "model_load_seconds": load_seconds,
            "positive_prompt_tokens": positive_tokens,
            "records": records,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
