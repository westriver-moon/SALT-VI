#!/usr/bin/env python3
"""Run the production QRI-v1 ROI stack on one image without using a GPU."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from semantic_imagination.regional.config import load_regional_config
from semantic_imagination.regional.roi import (
    HumanROIGenerator,
    SCHPLIPBackend,
    SegmentAnythingBackend,
    UltralyticsPoseBackend,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--modality", choices=("rgb", "ir"), required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.monotonic()
    config = load_regional_config(args.config)
    height, width = config.output_size_hw
    image = Image.open(args.image).convert("RGB").resize(
        (width, height), Image.Resampling.BICUBIC
    )
    generator = HumanROIGenerator(
        pose=UltralyticsPoseBackend(
            config.assets["yolo_pose"].path,
            device=args.device,
        ),
        parsing=SCHPLIPBackend(
            config.roi["schp_root"],
            config.assets["schp_lip"].path,
            device=args.device,
        ),
        sam=SegmentAnythingBackend(
            config.assets["sam_vit_b"].path,
            device=args.device,
            repository=config.roi["sam_root"],
        ),
    )
    regions = generator.regions(image, args.modality)
    result = {
        "valid": True,
        "image": str(args.image.resolve()),
        "image_size": list(image.size),
        "modality": args.modality,
        "device": args.device,
        "region_count": len(regions),
        "regions": [
            {
                "region_id": region.region_id,
                "category": region.category,
                "side": region.side,
                "bbox_xyxy": list(region.bbox_xyxy),
                "mask_pixels": int(np.asarray(region.mask, dtype=bool).sum()),
            }
            for region in regions
        ],
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
