#!/usr/bin/env python3
"""Run a generic tri-map controlled local imagination benchmark."""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
import yaml
from PIL import Image, ImageFilter
from diffusers import (
    AutoencoderKL,
    PNDMScheduler,
    StableDiffusionInpaintPipeline,
    UNet2DConditionModel,
)
from transformers import CLIPTextModel, CLIPTokenizer

from edit_program import (
    adaptive_writeback_map,
    build_trimap,
    feathered_region_map,
    harmonize_proposal,
    load_edit_program,
    rasterize_creation_map,
    rasterize_edit_region,
    rasterize_preservation_hint,
    weighted_map,
)


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def component_variant(component_root: Path, stem: str) -> str | None:
    return "fp16" if (component_root / f"{stem}.fp16.safetensors").is_file() else None


def load_pipeline(base_root: Path, inpaint_root: Path, device: str):
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
    scheduler = PNDMScheduler.from_pretrained(base_root / "scheduler", local_files_only=True)
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


def crop_box(value: list[int], image_size: tuple[int, int]) -> tuple[int, int, int, int]:
    if len(value) != 4:
        raise ValueError("crop_box_xyxy must contain four integers")
    x0, y0, x1, y1 = (int(item) for item in value)
    if not (0 <= x0 < x1 <= image_size[0] and 0 <= y0 < y1 <= image_size[1]):
        raise ValueError("crop_box_xyxy lies outside the reference image")
    return (x0, y0, x1, y1)


def image_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float32)


def mask_array(mask: Image.Image) -> np.ndarray:
    return np.asarray(mask.convert("L"), dtype=np.float32) / 255.0


def weighted_delta(reference: Image.Image, candidate: Image.Image, mask: Image.Image) -> float:
    delta = np.abs(image_array(candidate) - image_array(reference)).mean(axis=2)
    weight = mask_array(mask)
    return float((delta * weight).sum() / max(float(weight.sum()), 1e-8))


def crop_mask(full_mask: Image.Image, box: tuple[int, int, int, int]) -> Image.Image:
    return full_mask.crop(box)


def masked_style_energy(image: Image.Image, mask: Image.Image) -> float:
    array = image_array(image) / 255.0
    gray = array.mean(axis=2)
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:] = np.abs(gray[:, 1:] - gray[:, :-1])
    gy[1:, :] = np.abs(gray[1:, :] - gray[:-1, :])
    energy = gx + gy
    weight = mask_array(mask)
    return float((energy * weight).sum() / max(float(weight.sum()), 1e-8))


def build_source_lock(
    pipe: StableDiffusionInpaintPipeline,
    pipeline_image: Image.Image,
    edit_weight: Image.Image,
    seed: int,
    device: str,
    lock_start_fraction: float = 0.0,
    lock_ramp_fraction: float = 0.0,
):
    dtype = pipe.vae.dtype
    image_tensor = pipe.image_processor.preprocess(pipeline_image).to(device=device, dtype=dtype)
    source_generator = torch.Generator(device=device).manual_seed(int(seed) + 1_000_003)
    noise_generator = torch.Generator(device=device).manual_seed(int(seed) + 2_000_003)
    with torch.no_grad():
        source_latents = pipe._encode_vae_image(image_tensor, generator=source_generator)
    source_noise = torch.randn(
        source_latents.shape,
        generator=noise_generator,
        device=source_latents.device,
        dtype=source_latents.dtype,
    )
    weight_tensor = torch.from_numpy(mask_array(edit_weight)).to(
        device=source_latents.device, dtype=source_latents.dtype
    )[None, None]
    weight_tensor = functional.interpolate(
        weight_tensor,
        size=source_latents.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).clamp(0.0, 1.0)

    def callback(pipeline, step_index: int, timestep: int, callback_kwargs: dict[str, Any]):
        latents = callback_kwargs["latents"]
        total_steps = max(int(getattr(pipeline, "_num_timesteps", 0)), step_index + 1)
        progress = float(step_index + 1) / float(max(total_steps, 1))
        start = float(lock_start_fraction)
        ramp = float(lock_ramp_fraction)
        if progress <= start:
            return callback_kwargs
        if ramp > 0.0:
            lock_amount = min(1.0, max(0.0, (progress - start) / ramp))
        else:
            lock_amount = 1.0
        scheduler_timesteps = pipeline.scheduler.timesteps
        if step_index + 1 < len(scheduler_timesteps):
            next_timestep = scheduler_timesteps[step_index + 1]
            source_state = pipeline.scheduler.add_noise(
                source_latents, source_noise, next_timestep.reshape(1)
            )
        else:
            source_state = source_latents
        preservation_weight = (1.0 - weight_tensor) * float(lock_amount)
        callback_kwargs["latents"] = (
            (1.0 - preservation_weight) * latents + preservation_weight * source_state
        )
        return callback_kwargs

    return callback


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--base-root", required=True, type=Path)
    parser.add_argument("--inpaint-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    spec = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if int(spec.get("schema_version", 0)) != 1:
        raise ValueError("benchmark config requires schema_version=1")
    program = load_edit_program(Path(spec["edit_program"]))
    reference = Image.open(spec["inputs"]["reference"]).convert("RGB")
    box = crop_box(spec["crop_box_xyxy"], reference.size)
    reference_crop = reference.crop(box)
    process_size = int(spec["process_size"])

    creation_full = rasterize_creation_map(program, reference.size)
    edit_region_full = rasterize_edit_region(program, reference.size)
    preservation_hint_full = rasterize_preservation_hint(program, reference.size)
    trimap_full = build_trimap(
        creation_full,
        transition_dilation_px=int(spec["transition_dilation_px"]),
        transition_feather_px=float(spec["transition_feather_px"]),
    )
    initialization = dict(spec.get("initialization", {"type": "source"}))
    initialization_type = str(initialization.get("type", "source"))
    pipeline_source_full = reference
    if initialization_type == "source":
        pipeline_input_full = reference
    elif initialization_type == "layout_draft":
        color = initialization.get("color_rgb", [20, 20, 20])
        if not isinstance(color, list) or len(color) != 3:
            raise ValueError("layout_draft color_rgb must contain three values")
        opacity = float(initialization.get("opacity", 0.85))
        if not 0.0 <= opacity <= 1.0:
            raise ValueError("layout_draft opacity must be in [0, 1]")
        source_array = image_array(reference)
        draft_color = np.asarray(color, dtype=np.float32)[None, None, :]
        draft_alpha = mask_array(creation_full)[..., None] * opacity
        draft_array = np.clip(
            source_array * (1.0 - draft_alpha) + draft_color * draft_alpha,
            0.0,
            255.0,
        ).astype(np.uint8)
        pipeline_input_full = Image.fromarray(draft_array, mode="RGB")
    else:
        raise ValueError("initialization.type must be source or layout_draft")
    pipeline_source_image = pipeline_source_full.crop(box).resize(
        (process_size, process_size), Image.Resampling.LANCZOS
    )
    pipeline_image = pipeline_input_full.crop(box).resize(
        (process_size, process_size), Image.Resampling.LANCZOS
    )
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    pipeline_image.save(output_root / "input_crop.png", compress_level=2)
    pipeline_source_image.save(output_root / "source_crop.png", compress_level=2)
    for name, image in trimap_full.items():
        image.save(output_root / f"{name}_map.png", compress_level=2)
    edit_region_full.save(output_root / "edit_region_map.png", compress_level=2)
    preservation_hint_full.save(output_root / "preservation_hint_map.png", compress_level=2)

    trimap_crop = {name: crop_mask(image, box) for name, image in trimap_full.items()}
    inpaint_source = str(spec.get("inpaint_source", "trimap"))
    if inpaint_source == "trimap":
        inpaint_full = trimap_full["inpaint"]
    elif inpaint_source == "edit_region":
        inpaint_full = edit_region_full
    else:
        raise ValueError("inpaint_source must be trimap or edit_region")
    pipeline_mask = crop_mask(inpaint_full, box).resize(
        (process_size, process_size), Image.Resampling.NEAREST
    )
    pipeline_mask.save(output_root / "pipeline_mask.png", compress_level=2)

    load_started = time.perf_counter()
    pipe = load_pipeline(args.base_root, args.inpaint_root, args.device)
    load_seconds = time.perf_counter() - load_started
    prompt = ", ".join(
        value for value in [program["target_semantics"], program.get("appearance"), spec["prompt"]["added"]] if value
    )
    negative_prompt = str(spec["prompt"]["negative"])
    prompt_token_count = len(pipe.tokenizer(prompt, add_special_tokens=True).input_ids)
    if prompt_token_count > int(pipe.tokenizer.model_max_length):
        raise ValueError(
            f"positive prompt has {prompt_token_count} tokens; "
            f"maximum is {pipe.tokenizer.model_max_length}"
        )
    records = []
    reference_array = image_array(reference)

    for variant in spec["variants"]:
        seed = int(variant.get("seed", spec["seed"]))
        transition_weight = float(variant["transition_latent_weight"])
        edit_weight_full = weighted_map(trimap_full, transition_weight)
        edit_weight_crop = crop_mask(edit_weight_full, box).resize(
            (process_size, process_size), Image.Resampling.BILINEAR
        )
        final_alpha_full = weighted_map(
            trimap_full, float(variant["final_transition_alpha"])
        )
        final_alpha = mask_array(final_alpha_full)[..., None]
        callback = None
        if bool(variant["step_source_lock"]):
            source_lock_source = str(variant.get("source_lock_map", "trimap"))
            if source_lock_source == "trimap":
                source_lock_full = edit_weight_full
            elif source_lock_source == "edit_region":
                source_lock_full = edit_region_full
            else:
                raise ValueError("source_lock_map must be trimap or edit_region")
            source_lock_crop = crop_mask(source_lock_full, box).resize(
                (process_size, process_size), Image.Resampling.BILINEAR
            )
            callback = build_source_lock(
                pipe,
                pipeline_source_image,
                source_lock_crop,
                seed,
                args.device,
                lock_start_fraction=float(variant.get("lock_start_fraction", 0.0)),
                lock_ramp_fraction=float(variant.get("lock_ramp_fraction", 0.0)),
            )

        generator = torch.Generator(device=args.device).manual_seed(seed)
        torch.cuda.empty_cache()
        started = time.perf_counter()
        generated_crop = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=pipeline_image,
            mask_image=pipeline_mask,
            strength=float(variant["strength"]),
            num_inference_steps=int(variant["steps"]),
            guidance_scale=float(variant["guidance_scale"]),
            generator=generator,
            height=process_size,
            width=process_size,
            callback_on_step_end=callback,
            callback_on_step_end_tensor_inputs=["latents"],
        ).images[0].convert("RGB")
        elapsed = time.perf_counter() - started

        generated_native = generated_crop.resize(reference_crop.size, Image.Resampling.LANCZOS)
        proposal = reference.copy()
        proposal.paste(generated_native, box[:2])
        final_compositor = str(variant.get("final_compositor", "trimap"))
        writeback_proposal = proposal
        if final_compositor == "trimap":
            final_alpha_image = final_alpha_full
        elif final_compositor == "region":
            final_alpha_image = feathered_region_map(
                edit_region_full, float(variant.get("region_feather_px", 3.0))
            )
            region_max_alpha = float(variant.get("region_max_alpha", 1.0))
            if not 0.0 <= region_max_alpha <= 1.0:
                raise ValueError("region_max_alpha must be in [0, 1]")
            if region_max_alpha < 1.0:
                final_alpha_image = Image.fromarray(
                    np.round(
                        np.asarray(final_alpha_image, dtype=np.float32) * region_max_alpha
                    ).astype(np.uint8),
                    mode="L",
                )
            proposal_blur_px = float(variant.get("proposal_blur_px", 0.0))
            region_proposal = proposal
            if proposal_blur_px > 0.0:
                region_proposal = proposal.filter(ImageFilter.GaussianBlur(proposal_blur_px))
            writeback_proposal = harmonize_proposal(
                reference,
                region_proposal,
                edit_region_full,
                strength=float(variant.get("harmonize_strength", 0.0)),
            )
        elif final_compositor == "adaptive":
            adaptive_parameters = dict(variant.get("adaptive_writeback", {}))
            adaptive_maps = adaptive_writeback_map(
                reference,
                proposal,
                creation_full,
                preservation_hint_full,
                **adaptive_parameters,
            )
            final_alpha_image = adaptive_maps["alpha"]
        else:
            raise ValueError("final_compositor must be trimap, region or adaptive")
        final_alpha = mask_array(final_alpha_image)[..., None]
        proposal_array = image_array(writeback_proposal)
        composite_array = np.clip(
            reference_array * (1.0 - final_alpha) + proposal_array * final_alpha,
            0.0,
            255.0,
        ).astype(np.uint8)
        composite = Image.fromarray(composite_array, mode="RGB")

        variant_root = output_root / str(variant["name"])
        variant_root.mkdir(parents=True, exist_ok=True)
        raw_path = variant_root / f"seed_{seed}_raw_crop.png"
        composite_path = variant_root / f"seed_{seed}_composite.png"
        generated_crop.save(raw_path, compress_level=2)
        composite.save(composite_path, compress_level=2)
        final_alpha_image.save(variant_root / f"seed_{seed}_writeback_alpha.png", compress_level=2)

        evaluation_roi = Image.new("L", reference.size, 0)
        roi = program.get("evaluation_roi_xyxy")
        if roi:
            roi_box = crop_box(roi, reference.size)
            roi_array = np.zeros((reference.height, reference.width), dtype=np.uint8)
            roi_array[roi_box[1] : roi_box[3], roi_box[0] : roi_box[2]] = 255
            evaluation_roi = Image.fromarray(roi_array, mode="L")
        preservation_roi = Image.fromarray(
            np.round(mask_array(trimap_full["preservation"]) * mask_array(evaluation_roi) * 255.0).astype(np.uint8),
            mode="L",
        )
        reference_style = masked_style_energy(reference, trimap_full["transition"])
        candidate_style = masked_style_energy(composite, trimap_full["transition"])
        outside_edit_region = Image.fromarray(
            np.round((1.0 - mask_array(edit_region_full)) * 255.0).astype(np.uint8),
            mode="L",
        )
        metrics = {
            "creation_mean_abs_change": weighted_delta(reference, composite, trimap_full["creation"]),
            "transition_mean_abs_change": weighted_delta(reference, composite, trimap_full["transition"]),
            "preservation_roi_mean_abs_change": weighted_delta(reference, composite, preservation_roi),
            "global_preservation_mean_abs_change": weighted_delta(reference, composite, trimap_full["preservation"]),
            "outside_edit_region_mean_abs_change": weighted_delta(reference, composite, outside_edit_region),
            "transition_style_energy_ratio": float(
                candidate_style / max(reference_style, 1e-8)
            ),
        }
        records.append(
            {
                "variant": variant,
                "seed": seed,
                "seconds": elapsed,
                "raw_crop": str(raw_path),
                "composite": str(composite_path),
                "metrics": metrics,
            }
        )
        print(
            json.dumps(
                {
                    "variant": variant["name"],
                    "seconds": round(elapsed, 3),
                    "creation_change": round(metrics["creation_mean_abs_change"], 3),
                    "preservation_roi_change": round(
                        metrics["preservation_roi_mean_abs_change"], 3
                    ),
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
        "benchmark": spec["experiment_id"],
        "backend": "sd15-inpaint-trimap-source-lock",
        "inpaint_source": inpaint_source,
        "initialization": initialization,
        "config": str(args.config),
        "edit_program": str(spec["edit_program"]),
        "model_load_seconds": load_seconds,
        "prompt_token_count": prompt_token_count,
        "prompt": {"positive": prompt, "negative": negative_prompt},
        "records": records,
    }
    atomic_json(output_root / "metrics.json", payload)
    print(json.dumps({"metrics": str(output_root / "metrics.json"), "record_count": len(records)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
