from __future__ import annotations

from PIL import Image, ImageDraw, ImageOps

from .schema import Region


RESAMPLING = getattr(Image, "Resampling", Image)


def _clip_bbox(
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
    width = max(1, right - left)
    height = max(1, bottom - top)
    margin_x = width * float(margin_ratio) / 2.0
    margin_y = height * float(margin_ratio) / 2.0
    return _clip_bbox(
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
    resample=RESAMPLING.NEAREST,
    label: str | None = None,
) -> Image.Image:
    crop = image.convert("RGB").crop(_clip_bbox(bbox, image.size))
    label_height = 18 if label else 0
    available = (int(canvas_size[0]), max(1, int(canvas_size[1]) - label_height))
    fitted = ImageOps.contain(crop, available, method=resample)
    canvas = Image.new("RGB", canvas_size, (112, 112, 112))
    left = (canvas.width - fitted.width) // 2
    top = label_height + (available[1] - fitted.height) // 2
    canvas.paste(fitted, (left, top))
    if label:
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((0, 0, canvas.width, label_height), fill=(24, 24, 24))
        draw.text((4, 3), label, fill=(255, 255, 255))
    return canvas


def normalized_tight_crop(
    image: Image.Image,
    region: Region,
    size_px: int,
    *,
    resample=RESAMPLING.NEAREST,
) -> Image.Image:
    return crop_on_canvas(
        image,
        region.bbox_xyxy,
        (int(size_px), int(size_px)),
        resample=resample,
    )


def roi_comparison_board(
    lr: Image.Image,
    swin: Image.Image,
    region: Region,
    size_px: int = 512,
) -> Image.Image:
    """Create one bounded vision item containing tight and contextual A/B crops.

    The LR measurement is enlarged with nearest-neighbour interpolation so the
    board does not manufacture smooth edges.  SwinIR is shown separately and
    remains a non-authoritative proposal.
    """

    size_px = int(size_px)
    if size_px < 256 or size_px % 2:
        raise ValueError("ROI board size must be an even integer >= 256")
    lr_aligned = lr.convert("RGB").resize(swin.size, RESAMPLING.NEAREST)
    context = expanded_bbox(region.bbox_xyxy, swin.size)
    half = size_px // 2
    board = Image.new("RGB", (size_px, size_px), (64, 64, 64))
    tiles = (
        crop_on_canvas(
            lr_aligned,
            region.bbox_xyxy,
            (half, half),
            resample=RESAMPLING.NEAREST,
            label="A tight (LR)",
        ),
        crop_on_canvas(
            lr_aligned,
            context,
            (half, half),
            resample=RESAMPLING.NEAREST,
            label="A context (LR)",
        ),
        crop_on_canvas(
            swin,
            region.bbox_xyxy,
            (half, half),
            resample=RESAMPLING.LANCZOS,
            label="B tight (SwinIR)",
        ),
        crop_on_canvas(
            swin,
            context,
            (half, half),
            resample=RESAMPLING.LANCZOS,
            label="B context (SwinIR)",
        ),
    )
    for tile, xy in zip(tiles, ((0, 0), (half, 0), (0, half), (half, half))):
        board.paste(tile, xy)
    return board
