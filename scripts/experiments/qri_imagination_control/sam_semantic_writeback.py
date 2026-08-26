#!/usr/bin/env python3
"""Use EditSpec prompts and existing SAM weights for precise semantic write-back."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image, ImageFilter

from edit_program import (
    harmonize_proposal,
    load_edit_program,
    rasterize_creation_map,
    rasterize_edit_region,
    rasterize_preservation_hint,
)


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def mask_array(mask: Image.Image) -> np.ndarray:
    return np.asarray(mask.convert("L"), dtype=np.float32) / 255.0


def image_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float32)


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


def map_point(
    point: list[float],
    crop_box: tuple[int, int, int, int],
    target_size: tuple[int, int],
) -> tuple[float, float]:
    left, top, right, bottom = crop_box
    scale_x = float(target_size[0]) / float(right - left)
    scale_y = float(target_size[1]) / float(bottom - top)
    return ((float(point[0]) - left) * scale_x, (float(point[1]) - top) * scale_y)


def map_box(
    box: list[float],
    crop_box: tuple[int, int, int, int],
    target_size: tuple[int, int],
) -> np.ndarray:
    x0, y0 = map_point(box[:2], crop_box, target_size)
    x1, y1 = map_point(box[2:], crop_box, target_size)
    return np.asarray([x0, y0, x1, y1], dtype=np.float32)


def resize_crop_mask(
    full_mask: Image.Image,
    crop_box: tuple[int, int, int, int],
    target_size: tuple[int, int],
    *,
    dilation_px: int = 0,
) -> np.ndarray:
    mask = full_mask.crop(crop_box)
    if int(dilation_px) > 0:
        mask = mask.filter(ImageFilter.MaxFilter(2 * int(dilation_px) + 1))
    mask = mask.resize(target_size, Image.Resampling.NEAREST)
    return np.asarray(mask, dtype=np.uint8) > 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    spec = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    reference = Image.open(spec["reference"]).convert("RGB")
    candidate = Image.open(spec["candidate"]).convert("RGB")
    crop_box = tuple(int(value) for value in spec["candidate_crop_box_xyxy"])
    program = load_edit_program(Path(spec["edit_program"]))
    creation = rasterize_creation_map(program, reference.size)
    edit_region = rasterize_edit_region(program, reference.size)
    preservation = rasterize_preservation_hint(program, reference.size)

    sam_source = Path(spec["sam"]["source_root"])
    sam_weights = Path(spec["sam"]["weights"])
    if not sam_source.is_dir() or not sam_weights.is_file():
        raise FileNotFoundError("configured SAM source or weights are missing")
    sys.path.insert(0, str(sam_source))
    from segment_anything import SamPredictor, sam_model_registry

    sam = sam_model_registry[str(spec["sam"].get("model_type", "vit_b"))](
        checkpoint=str(sam_weights)
    ).to(args.device)
    predictor = SamPredictor(sam)
    candidate_array = np.asarray(candidate, dtype=np.uint8)
    predictor.set_image(candidate_array)

    point_coords = []
    point_labels = []
    for point in spec["sam_prompts"]["positive_points"]:
        point_coords.append(map_point(point, crop_box, candidate.size))
        point_labels.append(1)
    for point in spec["sam_prompts"]["negative_points"]:
        point_coords.append(map_point(point, crop_box, candidate.size))
        point_labels.append(0)
    masks, sam_scores, _ = predictor.predict(
        point_coords=np.asarray(point_coords, dtype=np.float32),
        point_labels=np.asarray(point_labels, dtype=np.int32),
        box=map_box(spec["sam_prompts"]["box_xyxy"], crop_box, candidate.size),
        multimask_output=True,
    )

    layout_band = resize_crop_mask(
        creation,
        crop_box,
        candidate.size,
        dilation_px=int(spec["selection"].get("layout_dilation_px", 3)),
    )
    protected = resize_crop_mask(preservation, crop_box, candidate.size)
    box_mask = np.zeros((candidate.height, candidate.width), dtype=bool)
    x0, y0, x1, y1 = np.round(
        map_box(spec["sam_prompts"]["box_xyxy"], crop_box, candidate.size)
    ).astype(int)
    x0, x1 = max(0, x0), min(candidate.width, x1)
    y0, y1 = max(0, y0), min(candidate.height, y1)
    box_mask[y0:y1, x0:x1] = True
    expected_area = float(spec["selection"].get("expected_box_area_ratio", 0.25))
    candidates = []
    for index, mask in enumerate(masks):
        coverage = float(mask[layout_band].mean()) if layout_band.any() else 0.0
        protected_overlap = float(mask[protected].mean()) if protected.any() else 0.0
        area_ratio = float(mask[box_mask].sum() / max(int(box_mask.sum()), 1))
        objective = (
            2.0 * coverage
            - 2.0 * protected_overlap
            - abs(area_ratio - expected_area)
            + 0.2 * float(sam_scores[index])
        )
        candidates.append(
            {
                "index": index,
                "sam_score": float(sam_scores[index]),
                "layout_coverage": coverage,
                "protected_overlap": protected_overlap,
                "box_area_ratio": area_ratio,
                "objective": objective,
            }
        )
    selected = max(candidates, key=lambda item: item["objective"])
    selected_mask = masks[int(selected["index"])]

    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    for index, mask in enumerate(masks):
        Image.fromarray(np.where(mask, 255, 0).astype(np.uint8), mode="L").save(
            output_root / f"sam_candidate_{index}.png", compress_level=2
        )

    native_mask = Image.fromarray(
        np.where(selected_mask, 255, 0).astype(np.uint8), mode="L"
    ).resize(
        (crop_box[2] - crop_box[0], crop_box[3] - crop_box[1]),
        Image.Resampling.NEAREST,
    )
    full_mask = Image.new("L", reference.size, 0)
    full_mask.paste(native_mask, crop_box[:2])
    constrained = np.minimum(mask_array(full_mask), mask_array(edit_region))
    alpha = Image.fromarray(
        np.round(constrained * 255.0).astype(np.uint8), mode="L"
    )
    dilation_px = int(spec["writeback"].get("mask_dilation_px", 0))
    if dilation_px > 0:
        alpha = alpha.filter(ImageFilter.MaxFilter(2 * dilation_px + 1))
    feather_px = float(spec["writeback"].get("feather_px", 0.7))
    if feather_px > 0.0:
        alpha = alpha.filter(ImageFilter.GaussianBlur(feather_px))
    max_alpha = float(spec["writeback"].get("max_alpha", 1.0))
    alpha = Image.fromarray(
        np.round(np.asarray(alpha, dtype=np.float32) * max_alpha).astype(np.uint8),
        mode="L",
    )

    proposal = reference.copy()
    proposal.paste(
        candidate.resize(
            (crop_box[2] - crop_box[0], crop_box[3] - crop_box[1]),
            Image.Resampling.LANCZOS,
        ),
        crop_box[:2],
    )
    proposal = harmonize_proposal(
        reference,
        proposal,
        edit_region,
        strength=float(spec["writeback"].get("harmonize_strength", 0.7)),
    )
    composite = blend(reference, proposal, alpha)
    alpha.save(output_root / "selected_writeback_alpha.png", compress_level=2)
    composite.save(output_root / "composite.png", compress_level=2)
    creation.save(output_root / "layout_anchor.png", compress_level=2)
    preservation.save(output_root / "preservation_hint.png", compress_level=2)
    outside = Image.fromarray(
        np.round((1.0 - mask_array(edit_region)) * 255.0).astype(np.uint8), mode="L"
    )
    payload = {
        "schema_version": 1,
        "selected": selected,
        "candidates": candidates,
        "metrics": {
            "layout_mean_abs_change": weighted_delta(reference, composite, creation),
            "preservation_hint_mean_abs_change": weighted_delta(
                reference, composite, preservation
            ),
            "outside_edit_region_mean_abs_change": weighted_delta(
                reference, composite, outside
            ),
            "writeback_alpha_mass": float(mask_array(alpha).sum()),
        },
        "composite": str(output_root / "composite.png"),
    }
    atomic_json(output_root / "metrics.json", payload)
    print(json.dumps(payload, separators=(",", ":")), flush=True)
    del predictor, sam
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
