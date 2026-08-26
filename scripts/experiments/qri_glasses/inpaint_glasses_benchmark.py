#!/usr/bin/env python3
"""Benchmark a dedicated SD1.5 inpainting UNet for localized eyewear recovery."""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image, ImageFilter
from diffusers import (
    AutoencoderKL,
    PNDMScheduler,
    StableDiffusionInpaintPipeline,
    UNet2DConditionModel,
)
from transformers import CLIPTextModel, CLIPTokenizer


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def component_variant(component_root: Path, stem: str) -> str | None:
    return "fp16" if (component_root / f"{stem}.fp16.safetensors").is_file() else None


def soft_mask(mask: np.ndarray, dilation_px: int, feather_px: float) -> Image.Image:
    image = Image.fromarray(np.where(mask, 255, 0).astype(np.uint8), mode="L")
    if dilation_px > 0:
        image = image.filter(ImageFilter.MaxFilter(2 * dilation_px + 1))
    if feather_px > 0:
        image = image.filter(ImageFilter.GaussianBlur(float(feather_px)))
    return image


def mask_metrics(reference: Image.Image, candidate: Image.Image, mask: Image.Image) -> dict:
    reference_array = np.asarray(reference, dtype=np.float32)
    candidate_array = np.asarray(candidate, dtype=np.float32)
    delta = np.abs(candidate_array - reference_array).mean(axis=2)
    alpha = np.asarray(mask, dtype=np.float32) / 255.0
    inside_weight = max(float(alpha.sum()), 1e-8)
    outside = 1.0 - alpha
    outside_weight = max(float(outside.sum()), 1e-8)
    return {
        "inside_mean_abs_change": float((delta * alpha).sum() / inside_weight),
        "outside_mean_abs_change": float((delta * outside).sum() / outside_weight),
        "inside_changed_fraction_gt10": float(
            (((delta > 10).astype(np.float32) * alpha).sum()) / inside_weight
        ),
        "max_abs_change": float(delta.max()),
    }


def lr_cycle_energy(candidate: Image.Image, lr: Image.Image) -> float:
    cycled = candidate.resize(lr.size, Image.Resampling.LANCZOS)
    candidate_array = np.asarray(cycled, dtype=np.float32) / 255.0
    lr_array = np.asarray(lr, dtype=np.float32) / 255.0
    return float(np.mean(np.square(candidate_array - lr_array)))


def load_pipeline(base_root: Path, inpaint_root: Path, device: str):
    """Reuse SALT's SD1.5 components and load only the dedicated 9-channel UNet."""
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    tokenizer = CLIPTokenizer.from_pretrained(base_root / "tokenizer", local_files_only=True)
    text_root = base_root / "text_encoder"
    vae_root = base_root / "vae"
    text_encoder = CLIPTextModel.from_pretrained(
        text_root,
        variant=component_variant(text_root, "model"),
        torch_dtype=dtype,
        local_files_only=True,
    )
    vae = AutoencoderKL.from_pretrained(
        vae_root,
        variant=component_variant(vae_root, "diffusion_pytorch_model"),
        torch_dtype=dtype,
        local_files_only=True,
    )
    scheduler = PNDMScheduler.from_pretrained(
        base_root / "scheduler", local_files_only=True
    )
    unet_root = inpaint_root / "unet"
    unet = UNet2DConditionModel.from_pretrained(
        unet_root,
        variant=component_variant(unet_root, "diffusion_pytorch_model"),
        torch_dtype=dtype,
        local_files_only=True,
    )
    pipe = StableDiffusionInpaintPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        safety_checker=None,
        feature_extractor=None,
        image_encoder=None,
        requires_safety_checker=False,
    )
    pipe.set_progress_bar_config(disable=False)
    pipe.enable_attention_slicing()
    return pipe.to(device)


def square_crop_box(size: tuple[int, int], side: int) -> tuple[int, int, int, int]:
    width, height = size
    side = min(int(side), width, height)
    return (0, 0, side, side)


def make_pipeline_mask(mask_crop: Image.Image, dilation_px: int) -> Image.Image:
    mask = mask_crop.convert("L")
    if dilation_px > 0:
        kernel = 2 * dilation_px + 1
        mask = mask.filter(ImageFilter.MaxFilter(kernel))
    return mask.point(lambda value: 255 if value > 8 else 0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-config", required=True, type=Path)
    parser.add_argument("--base-root", required=True, type=Path)
    parser.add_argument("--inpaint-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--section", default="sd15_inpaint")
    args = parser.parse_args()

    spec = yaml.safe_load(args.benchmark_config.read_text(encoding="utf-8"))
    inpaint_spec = spec[args.section]
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    reference = Image.open(spec["inputs"]["swin_reference"]).convert("RGB")
    archived_mask = Image.open(spec["inputs"]["eye_mask"]).convert("L")
    lr = Image.open(spec["inputs"]["source"]).convert("RGB").resize(
        (reference.width // 2, reference.height // 2), Image.Resampling.BICUBIC
    )
    crop_box = square_crop_box(reference.size, int(inpaint_spec["crop_side_px"]))
    reference_crop = reference.crop(crop_box)
    mask_crop = archived_mask.crop(crop_box)
    pipeline_mask = make_pipeline_mask(
        mask_crop, int(inpaint_spec["mask_dilation_px"])
    )
    pipeline_size = int(inpaint_spec["process_size"])
    pipeline_image = reference_crop.resize(
        (pipeline_size, pipeline_size), Image.Resampling.LANCZOS
    )
    pipeline_mask = pipeline_mask.resize(
        (pipeline_size, pipeline_size), Image.Resampling.NEAREST
    )
    pipeline_image.save(output_root / "input_crop.png", compress_level=2)
    pipeline_mask.save(output_root / "pipeline_mask.png", compress_level=2)

    final_mask = soft_mask(
        np.asarray(archived_mask) > 0,
        dilation_px=int(spec["mask"]["dilation_px"]),
        feather_px=float(spec["mask"]["feather_px"]),
    )
    final_alpha = np.asarray(final_mask, dtype=np.float32)
    if final_alpha.max() > 1.0:
        final_alpha /= 255.0
    final_alpha = final_alpha[..., None]

    load_started = time.perf_counter()
    pipe = load_pipeline(args.base_root, args.inpaint_root, args.device)
    load_seconds = time.perf_counter() - load_started

    prompt = f'{spec["prompt"]["positive"]}, {spec["prompt"]["added"]}'
    negative = str(spec["prompt"]["negative"])
    records = []
    for variant in inpaint_spec["variants"]:
        variant_root = output_root / str(variant["name"])
        variant_root.mkdir(parents=True, exist_ok=True)
        for seed_value in inpaint_spec.get("seeds", spec["seeds"]):
            seed = int(seed_value)
            generator = torch.Generator(device=args.device).manual_seed(seed)
            torch.cuda.empty_cache()
            tick = time.perf_counter()
            generated_crop = pipe(
                prompt=prompt,
                negative_prompt=negative,
                image=pipeline_image,
                mask_image=pipeline_mask,
                strength=float(variant["strength"]),
                num_inference_steps=int(variant["steps"]),
                guidance_scale=float(variant["guidance_scale"]),
                generator=generator,
                height=pipeline_size,
                width=pipeline_size,
            ).images[0].convert("RGB")
            elapsed = time.perf_counter() - tick

            generated_native = generated_crop.resize(
                reference_crop.size, Image.Resampling.LANCZOS
            )
            candidate = reference.copy()
            candidate.paste(generated_native, crop_box[:2])
            reference_array = np.asarray(reference, dtype=np.float32)
            candidate_array = np.asarray(candidate, dtype=np.float32)
            composite_array = np.clip(
                reference_array * (1.0 - final_alpha)
                + candidate_array * final_alpha,
                0,
                255,
            ).astype(np.uint8)
            composite = Image.fromarray(composite_array, mode="RGB")

            raw_path = variant_root / f"seed_{seed}_raw_crop.png"
            composite_path = variant_root / f"seed_{seed}_composite.png"
            generated_crop.save(raw_path, compress_level=2)
            composite.save(composite_path, compress_level=2)
            metrics = mask_metrics(reference, composite, final_mask)
            metrics["lr_cycle_energy"] = lr_cycle_energy(composite, lr)
            records.append(
                {
                    "backend": "stable-diffusion-v1-5-inpainting",
                    "variant": variant,
                    "seed": seed,
                    "seconds": elapsed,
                    "crop_box_xyxy": list(crop_box),
                    "metrics": metrics,
                    "raw_crop": str(raw_path),
                    "composite": str(composite_path),
                }
            )
            print(
                json.dumps(
                    {
                        "variant": variant["name"],
                        "seed": seed,
                        "seconds": round(elapsed, 3),
                        "inside_change": round(metrics["inside_mean_abs_change"], 3),
                    },
                    separators=(",", ":"),
                ),
                flush=True,
            )

    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    payload = {
        "schema_version": 1,
        "benchmark": "qri-glasses-gpu0-20260821",
        "backend": "stable-diffusion-v1-5-inpainting",
        "config_section": args.section,
        "model_load_seconds": load_seconds,
        "base_root": str(args.base_root),
        "inpaint_root": str(args.inpaint_root),
        "prompt": spec["prompt"],
        "records": records,
    }
    metrics_path = output_root / "metrics.json"
    atomic_json(metrics_path, payload)
    print(json.dumps({"metrics": str(metrics_path), "record_count": len(records)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
