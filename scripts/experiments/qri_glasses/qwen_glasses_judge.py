#!/usr/bin/env python3
"""Use the project Qwen-VL service to judge localized eyewear candidates."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from PIL import Image, ImageOps


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
PLUGIN_ROOT = PROJECT_ROOT / "plugins" / "qwen_imagination"
for path in (PROJECT_ROOT, PROJECT_ROOT / "src", PLUGIN_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from qwen_imagination.regional.qwen import (  # noqa: E402
    LlamaServerQwenReasoner,
    _data_url,
)


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def square_panel(image: Image.Image, box: tuple[int, int, int, int], size: int) -> Image.Image:
    crop = image.crop(box).convert("RGB")
    fitted = ImageOps.contain(crop, (size, size), Image.Resampling.LANCZOS)
    panel = Image.new("RGB", (size, size), (127, 127, 127))
    panel.paste(fitted, ((size - fitted.width) // 2, (size - fitted.height) // 2))
    return panel


def comparison_board(image: Image.Image, *, source: bool = False) -> Image.Image:
    if source:
        tight = (20, 0, 72, 52)
        context = (0, 0, min(89, image.width), min(100, image.height))
    else:
        tight = (72, 0, 209, 76)
        context = (40, 0, 236, 160)
    left = square_panel(image, tight, 384)
    right = square_panel(image, context, 384)
    board = Image.new("RGB", (768, 384), (127, 127, 127))
    board.paste(left, (0, 0))
    board.paste(right, (384, 0))
    return board


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--endpoint", default="http://127.0.0.1:8080/v1/chat/completions"
    )
    parser.add_argument(
        "--model-id", default="third-party-qwen3.8-27b-ud-q4-k-xl"
    )
    args = parser.parse_args()

    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    source = Image.open(args.source).convert("RGB")
    reference = Image.open(args.reference).convert("RGB")
    source_board = comparison_board(source, source=True)
    reference_board = comparison_board(reference)
    reasoner = LlamaServerQwenReasoner(
        endpoint=args.endpoint,
        model_id=args.model_id,
        timeout_seconds=300,
        enable_thinking=False,
        reasoning_effort="none",
    )
    instruction = (
        "Act as a strict visual quality judge for low-resolution person re-identification. "
        "Image A is the authoritative source eye/head crop, Image B is the Swin reference, "
        "and Image C is a localized inpainting candidate. Each image is a board with a tight "
        "eye crop on the left and wider face context on the right. Judge visible pixels only; "
        "do not trust the generation prompt. A valid result must show two plausible optical "
        "lenses or frames, a coherent bridge across the nose, and no duplicated loops, goggles, "
        "mosaic, or changed facial anatomy. Return one JSON object with exactly these keys: "
        "candidate_eyewear (valid_glasses|ambiguous|artifact|absent), source_support "
        "(strong|weak|none), two_lenses (visible|uncertain|not_visible), bridge "
        "(visible|uncertain|not_visible), identity_preservation (pass|fail|uncertain), "
        "confidence (number 0 to 1), short_reason (under 25 words)."
    )

    records = []
    for row in metrics["records"]:
        candidate = Image.open(row["composite"]).convert("RGB")
        candidate_board = comparison_board(candidate)
        content = [
            {"type": "text", "text": "Image A: authoritative low-resolution source."},
            {"type": "image_url", "image_url": {"url": _data_url(source_board)}},
            {"type": "text", "text": "Image B: Swin reference before inpainting."},
            {"type": "image_url", "image_url": {"url": _data_url(reference_board)}},
            {"type": "text", "text": "Image C: candidate after localized inpainting."},
            {"type": "image_url", "image_url": {"url": _data_url(candidate_board)}},
        ]
        result = reasoner._complete(
            content,
            instruction,
            seed=int(row["seed"]),
            temperature=0.0,
            max_tokens=1024,
        )
        record = {
            "variant": row["variant"]["name"],
            "seed": row["seed"],
            "composite": row["composite"],
            "judgment": result,
        }
        records.append(record)
        print(json.dumps(record, ensure_ascii=False, separators=(",", ":")), flush=True)

    payload = {
        "schema_version": 1,
        "model_id": args.model_id,
        "source": str(args.source),
        "reference": str(args.reference),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, payload)
    print(json.dumps({"metrics": str(args.output), "record_count": len(records)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
