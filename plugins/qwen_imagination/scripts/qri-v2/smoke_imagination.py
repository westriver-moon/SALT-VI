#!/usr/bin/env python3
"""Verify that the real QRI-v2 model proposes imaginative ROI worlds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEMANTIC_ROOT = PROJECT_ROOT
if str(SEMANTIC_ROOT) not in sys.path:
    sys.path.insert(0, str(SEMANTIC_ROOT))

from qwen_imagination.regional.qwen_v2 import (  # noqa: E402
    V2_NON_EDIT_STATES,
    ImaginativeQwenReasoner,
)
from qwen_imagination.regional.schema import Region  # noqa: E402


def _bbox(value: str) -> tuple[int, int, int, int]:
    try:
        result = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("bbox must contain four integers") from error
    if len(result) != 4 or result[0] >= result[2] or result[1] >= result[3]:
        raise argparse.ArgumentTypeError("bbox must be left,top,right,bottom")
    return result


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lr", required=True, type=Path)
    parser.add_argument("--swin", required=True, type=Path)
    parser.add_argument("--bbox", type=_bbox, default=(69, 20, 201, 71))
    parser.add_argument("--region-id", default="eyes")
    parser.add_argument("--category", default="eyewear")
    parser.add_argument(
        "--endpoint", default="http://127.0.0.1:8080/v1/chat/completions"
    )
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    lr = Image.open(args.lr.expanduser().resolve()).convert("RGB")
    swin = Image.open(args.swin.expanduser().resolve()).convert("RGB")
    if swin.size != (256, 512):
        raise ValueError(f"Swin input must be the canonical 256x512 canvas, got {swin.size}")
    left, top, right, bottom = args.bbox
    mask = np.zeros((swin.height, swin.width), dtype=bool)
    mask[top:bottom, left:right] = True
    region = Region(args.region_id, args.category, args.bbox, mask)
    reasoner = ImaginativeQwenReasoner(
        endpoint=args.endpoint,
        timeout_seconds=args.timeout_seconds,
        proposal_rounds=args.rounds,
    )
    candidates = reasoner.propose(lr, swin, [region])[region.region_id]
    generated_positive = [
        candidate
        for candidate in candidates
        if candidate.state not in V2_NON_EDIT_STATES
        and "inserted by the imagination coverage contract" not in candidate.evidence
    ]
    contract = {
        "positive_from_model": bool(generated_positive),
        "absent_present": any(item.state == "absent" for item in candidates),
        "unresolved_present": any(item.state == "unresolved" for item in candidates),
    }
    payload = {
        "valid": contract["positive_from_model"] and contract["unresolved_present"],
        "contract": contract,
        "lr_size": list(lr.size),
        "swin_size": list(swin.size),
        "bbox_xyxy": list(args.bbox),
        "candidates": [
            {
                "state": item.state,
                "value": item.value,
                "evidence": item.evidence,
                "evidence_source": item.evidence_source,
            }
            for item in candidates
        ],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not contract["positive_from_model"]:
        raise RuntimeError("Qwen proposed no positive interpretation; only fallback survived")
    if not contract["unresolved_present"]:
        raise RuntimeError("QRI-v2 unresolved control world is missing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
