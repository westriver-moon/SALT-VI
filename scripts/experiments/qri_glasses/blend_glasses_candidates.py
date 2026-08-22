#!/usr/bin/env python3
"""Create cheap opacity blends from one Qwen-validated eyewear candidate."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from inpaint_glasses_benchmark import lr_cycle_energy, mask_metrics, soft_mask  # noqa: E402


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def opacity_name(value: float) -> str:
    return f"blend_t{int(round(value * 100)):03d}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-config", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--opacities", nargs="+", type=float, default=[0.25, 0.40, 0.55, 0.70, 0.85]
    )
    parser.add_argument("--seed", type=int, default=20260821)
    args = parser.parse_args()

    spec = yaml.safe_load(args.benchmark_config.read_text(encoding="utf-8"))
    reference = Image.open(args.reference).convert("RGB")
    candidate = Image.open(args.candidate).convert("RGB")
    if candidate.size != reference.size:
        raise ValueError(f"candidate size {candidate.size} != reference size {reference.size}")
    lr = Image.open(spec["inputs"]["source"]).convert("RGB").resize(
        (reference.width // 2, reference.height // 2), Image.Resampling.BICUBIC
    )
    archived_mask = np.asarray(Image.open(spec["inputs"]["eye_mask"]).convert("L")) > 0
    audit_mask = soft_mask(
        archived_mask,
        dilation_px=int(spec["mask"]["dilation_px"]),
        feather_px=float(spec["mask"]["feather_px"]),
    )
    reference_array = np.asarray(reference, dtype=np.float32)
    candidate_array = np.asarray(candidate, dtype=np.float32)
    args.output_root.mkdir(parents=True, exist_ok=True)

    records = []
    for opacity in args.opacities:
        if not 0.0 < opacity <= 1.0:
            raise ValueError(f"opacity must be in (0, 1], got {opacity}")
        variant = opacity_name(opacity)
        variant_root = args.output_root / variant
        variant_root.mkdir(parents=True, exist_ok=True)
        blended_array = np.clip(
            reference_array * (1.0 - opacity) + candidate_array * opacity,
            0,
            255,
        ).astype(np.uint8)
        blended = Image.fromarray(blended_array, mode="RGB")
        path = variant_root / f"seed_{args.seed}_composite.png"
        blended.save(path, compress_level=2)
        metrics = mask_metrics(reference, blended, audit_mask)
        metrics["lr_cycle_energy"] = lr_cycle_energy(blended, lr)
        records.append(
            {
                "backend": "sd15-inpaint-opacity-blend",
                "variant": {
                    "name": variant,
                    "opacity": opacity,
                    "source_candidate": str(args.candidate),
                },
                "seed": args.seed,
                "seconds": 0.0,
                "metrics": metrics,
                "composite": str(path),
            }
        )
        print(
            json.dumps(
                {
                    "variant": variant,
                    "inside_change": round(metrics["inside_mean_abs_change"], 3),
                    "outside_change": round(metrics["outside_mean_abs_change"], 6),
                },
                separators=(",", ":"),
            ),
            flush=True,
        )

    payload = {
        "schema_version": 1,
        "benchmark": "qri-glasses-gpu0-20260821",
        "backend": "sd15-inpaint-opacity-blend",
        "reference": str(args.reference),
        "source_candidate": str(args.candidate),
        "records": records,
    }
    output = args.output_root / "metrics.json"
    atomic_json(output, payload)
    print(json.dumps({"metrics": str(output), "record_count": len(records)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
