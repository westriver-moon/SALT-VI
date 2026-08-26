#!/usr/bin/env python3
"""Recompose an existing local-imagination candidate with adaptive write-back."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageFilter

from edit_program import (
    adaptive_writeback_map,
    feathered_region_map,
    harmonize_proposal,
    load_edit_program,
    rasterize_creation_map,
    rasterize_edit_region,
    rasterize_preservation_hint,
)


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def parse_box(value: list[int], size: tuple[int, int]) -> tuple[int, int, int, int]:
    if len(value) != 4:
        raise ValueError("crop_box_xyxy must contain four integers")
    x0, y0, x1, y1 = (int(item) for item in value)
    if not (0 <= x0 < x1 <= size[0] and 0 <= y0 < y1 <= size[1]):
        raise ValueError("crop_box_xyxy lies outside the reference image")
    return (x0, y0, x1, y1)


def blend(reference: Image.Image, proposal: Image.Image, alpha: Image.Image) -> Image.Image:
    source = np.asarray(reference.convert("RGB"), dtype=np.float32)
    target = np.asarray(proposal.convert("RGB"), dtype=np.float32)
    weight = np.asarray(alpha.convert("L"), dtype=np.float32)[..., None] / 255.0
    output = np.clip(source * (1.0 - weight) + target * weight, 0.0, 255.0)
    return Image.fromarray(np.round(output).astype(np.uint8), mode="RGB")


def masked_delta(reference: Image.Image, candidate: Image.Image, mask: Image.Image) -> float:
    source = np.asarray(reference.convert("RGB"), dtype=np.float32)
    target = np.asarray(candidate.convert("RGB"), dtype=np.float32)
    delta = np.abs(target - source).mean(axis=2)
    weight = np.asarray(mask.convert("L"), dtype=np.float32) / 255.0
    return float((delta * weight).sum() / max(float(weight.sum()), 1e-8))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--edit-program", type=Path)
    args = parser.parse_args()

    spec = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    reference_path = args.reference or Path(spec["reference"])
    candidate_path = args.candidate or Path(spec["candidate"])
    program_path = args.edit_program or Path(spec["edit_program"])
    reference = Image.open(reference_path).convert("RGB")
    candidate_crop = Image.open(candidate_path).convert("RGB")
    program = load_edit_program(program_path)
    box = parse_box(spec["crop_box_xyxy"], reference.size)
    proposal = reference.copy()
    proposal.paste(candidate_crop.resize((box[2] - box[0], box[3] - box[1]), Image.Resampling.LANCZOS), box[:2])
    anchor = rasterize_creation_map(program, reference.size)
    edit_region = rasterize_edit_region(program, reference.size)
    preservation = rasterize_preservation_hint(program, reference.size)

    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    inspection_box = None
    if spec.get("inspection_roi_xyxy"):
        inspection_box = parse_box(spec["inspection_roi_xyxy"], reference.size)
        proposal.crop(inspection_box).resize(
            ((inspection_box[2] - inspection_box[0]) * 4, (inspection_box[3] - inspection_box[1]) * 4),
            Image.Resampling.NEAREST,
        ).save(output_root / "proposal_inspection_x4.png", compress_level=2)
    records = []
    for variant in spec["variants"]:
        name = str(variant["name"])
        strategy = str(variant.get("strategy", "evidence"))
        recorded_parameters = {
            key: value for key, value in variant.items() if key not in {"name", "strategy"}
        }
        parameters = dict(recorded_parameters)
        writeback_proposal = proposal
        if strategy == "evidence":
            maps = adaptive_writeback_map(
                reference,
                proposal,
                anchor,
                preservation,
                **parameters,
            )
        elif strategy == "region":
            feather_px = float(parameters.pop("feather_px"))
            harmonize_strength = float(parameters.pop("harmonize_strength", 0.0))
            proposal_blur_px = float(parameters.pop("proposal_blur_px", 0.0))
            max_alpha = float(parameters.pop("max_alpha", 1.0))
            if not 0.0 <= max_alpha <= 1.0:
                raise ValueError("max_alpha must be in [0, 1]")
            if parameters:
                raise ValueError(f"unused region parameters: {sorted(parameters)}")
            region_proposal = proposal
            if proposal_blur_px > 0.0:
                region_proposal = proposal.filter(ImageFilter.GaussianBlur(proposal_blur_px))
            writeback_proposal = harmonize_proposal(
                reference,
                region_proposal,
                edit_region,
                strength=harmonize_strength,
            )
            alpha = feathered_region_map(edit_region, feather_px)
            if max_alpha < 1.0:
                alpha = Image.fromarray(
                    np.round(np.asarray(alpha, dtype=np.float32) * max_alpha).astype(np.uint8),
                    mode="L",
                )
            maps = {"alpha": alpha, "support": edit_region}
        else:
            raise ValueError(f"unsupported write-back strategy: {strategy}")
        composite = blend(reference, writeback_proposal, maps["alpha"])
        variant_root = output_root / name
        variant_root.mkdir(parents=True, exist_ok=True)
        for map_name, image in maps.items():
            image.save(variant_root / f"{map_name}.png", compress_level=2)
        composite_path = variant_root / "composite.png"
        composite.save(composite_path, compress_level=2)
        if inspection_box is not None:
            composite.crop(inspection_box).resize(
                (
                    (inspection_box[2] - inspection_box[0]) * 4,
                    (inspection_box[3] - inspection_box[1]) * 4,
                ),
                Image.Resampling.NEAREST,
            ).save(variant_root / "inspection_x4.png", compress_level=2)
        alpha_array = np.asarray(maps["alpha"], dtype=np.float32) / 255.0
        metrics = {
            "alpha_mass": float(alpha_array.sum()),
            "alpha_nonzero_pixels": int((alpha_array > 1e-4).sum()),
            "anchor_mean_abs_change": masked_delta(reference, composite, anchor),
            "preservation_hint_mean_abs_change": masked_delta(reference, composite, preservation),
        }
        records.append(
            {
                "name": name,
                "strategy": strategy,
                "parameters": recorded_parameters,
                "composite": str(composite_path),
                "metrics": metrics,
            }
        )
        print(json.dumps({"name": name, **metrics}, separators=(",", ":")), flush=True)

    atomic_json(
        output_root / "metrics.json",
        {
            "schema_version": 1,
            "reference": str(reference_path),
            "candidate": str(candidate_path),
            "edit_program": str(program_path),
            "records": records,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
