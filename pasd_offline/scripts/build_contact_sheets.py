from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


def tile(image: Image.Image, size=(128, 256)) -> Image.Image:
    return ImageOps.pad(image.convert("RGB"), size, method=Image.Resampling.LANCZOS, color=(20, 20, 20))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build source + five-view pilot contact sheets")
    parser.add_argument("--records", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--rows-per-sheet", type=int, default=10)
    args = parser.parse_args()
    records = [json.loads(line) for line in Path(args.records).read_text(encoding="utf-8").splitlines() if line.strip()]
    root = Path(args.output_root).expanduser().resolve()
    sheets = root / "contact-sheets"
    sheets.mkdir(parents=True, exist_ok=True)
    tile_width, tile_height, label_height = 128, 256, 24
    for start in range(0, len(records), args.rows_per_sheet):
        chunk = records[start : start + args.rows_per_sheet]
        canvas = Image.new("RGB", (tile_width * 6, (tile_height + label_height) * len(chunk)), "white")
        draw = ImageDraw.Draw(canvas)
        for row, record in enumerate(chunk):
            y = row * (tile_height + label_height)
            images = [Path(record["image"]), *(root / view["output"] for view in record["views"])]
            for column, path in enumerate(images):
                with Image.open(path) as image:
                    canvas.paste(tile(image), (column * tile_width, y))
                label = "source" if column == 0 else f"view_{column - 1}"
                draw.text((column * tile_width + 4, y + tile_height + 4), label, fill="black")
        canvas.save(sheets / f"pilot-{start // args.rows_per_sheet:02d}.png", format="PNG")
    print(json.dumps({"records": len(records), "sheets": len(list(sheets.glob('*.png'))), "output": str(sheets)}))


if __name__ == "__main__":
    main()
