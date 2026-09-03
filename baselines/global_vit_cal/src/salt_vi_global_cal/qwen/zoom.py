from __future__ import annotations

from PIL import Image, ImageDraw, ImageOps

from .schema import Region


RESAMPLING = getattr(Image, "Resampling", Image)


def clip_bbox(
    bbox: tuple[int, int, int, int], size: tuple[int, int]
) -> tuple[int, int, int, int]:
    left, top, right, bottom = bbox
    width, height = size
    left = max(0, min(int(left), width - 1))
    top = max(0, min(int(top), height - 1))
    right = max(left + 1, min(int(right), width))
    bottom = max(top + 1, min(int(bottom), height))
    return left, top, right, bottom


def expanded_bbox(
    bbox: tuple[int, int, int, int],
    size: tuple[int, int],
    margin_ratio: float = 0.75,
) -> tuple[int, int, int, int]:
    left, top, right, bottom = bbox
    width, height = max(1, right - left), max(1, bottom - top)
    margin_x = width * float(margin_ratio) / 2.0
    margin_y = height * float(margin_ratio) / 2.0
    return clip_bbox(
        (
            round(left - margin_x),
            round(top - margin_y),
            round(right + margin_x),
            round(bottom + margin_y),
        ),
        size,
    )


def crop_on_canvas(
    image: Image.Image,
    bbox: tuple[int, int, int, int],
    canvas_size: tuple[int, int],
    *,
    label: str,
) -> Image.Image:
    crop = image.convert("RGB").crop(clip_bbox(bbox, image.size))
    label_height = 18
    available = (int(canvas_size[0]), max(1, int(canvas_size[1]) - label_height))
    fitted = ImageOps.contain(crop, available, method=RESAMPLING.LANCZOS)
    canvas = Image.new("RGB", canvas_size, (112, 112, 112))
    left = (canvas.width - fitted.width) // 2
    top = label_height + (available[1] - fitted.height) // 2
    canvas.paste(fitted, (left, top))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, canvas.width, label_height), fill=(24, 24, 24))
    draw.text((4, 3), label, fill=(255, 255, 255))
    return canvas


def swin_roi_board(
    swin: Image.Image,
    region: Region,
    size_px: int = 512,
    margin_ratio: float = 0.75,
) -> Image.Image:
    """512x512 tight/context board from the authoritative SwinIR view."""

    size_px = int(size_px)
    if size_px < 256 or size_px % 2:
        raise ValueError("ROI board size must be an even integer >= 256")
    swin = swin.convert("RGB")
    context = expanded_bbox(region.bbox_xyxy, swin.size, margin_ratio)
    half = size_px // 2
    board = Image.new("RGB", (size_px, size_px), (64, 64, 64))
    tight = crop_on_canvas(
        swin, region.bbox_xyxy, (half, size_px), label="SwinIR tight"
    )
    contextual = crop_on_canvas(
        swin, context, (half, size_px), label="SwinIR context"
    )
    board.paste(tight, (0, 0))
    board.paste(contextual, (half, 0))
    return board
