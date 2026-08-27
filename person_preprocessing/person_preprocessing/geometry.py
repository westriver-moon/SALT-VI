"""Two explicit image geometries; coordinates are always (x, y)."""

import math

import numpy as np
from PIL import Image, ImageFilter, ImageOps


RESAMPLING = Image.Resampling


def render(image, mode, size_hw, bbox=None, margin=0.05, blur_radius=24.0):
    image = image.convert("RGB")
    height, width = map(int, size_hw)
    sw, sh = image.size
    if mode == "resize":
        output = image.resize((width, height), RESAMPLING.BICUBIC)
        geometry = {
            "crop_box": [0, 0, sw, sh],
            "foreground_box": [0, 0, width, height],
            "source_to_target": [[width / sw, 0, 0], [0, height / sh, 0], [0, 0, 1]],
        }
    elif mode == "person_fit":
        if bbox is None:
            raise ValueError("person_fit requires a detected person box")
        x1, y1, x2, y2 = map(float, bbox)
        if not (0 <= x1 < x2 <= sw and 0 <= y1 < y2 <= sh):
            raise ValueError("person box must be nonempty and inside the source")
        dx, dy = (x2 - x1) * margin, (y2 - y1) * margin
        crop = [max(0, math.floor(x1 - dx)), max(0, math.floor(y1 - dy)),
                min(sw, math.ceil(x2 + dx)), min(sh, math.ceil(y2 + dy))]
        cw, ch = crop[2] - crop[0], crop[3] - crop[1]
        scale = min(width / cw, height / ch)
        left, top = (width - scale * cw) / 2, (height - scale * ch) / 2
        foreground = [left, top, left + scale * cw, top + scale * ch]
        # A single affine scale preserves aspect exactly, including subpixel placement.
        layer = image.transform(
            (width, height), Image.Transform.AFFINE,
            (1 / scale, 0, crop[0] - left / scale,
             0, 1 / scale, crop[1] - top / scale), RESAMPLING.BICUBIC,
        )
        background = ImageOps.fit(image, (width, height), RESAMPLING.BICUBIC)
        background = background.filter(ImageFilter.GaussianBlur(blur_radius))
        mask = content_mask(foreground, (height, width))
        output = Image.composite(layer, background, Image.fromarray(mask * 255))
        geometry = {
            "person_bbox": [float(value) for value in bbox], "crop_box": crop,
            "foreground_box": foreground, "person_margin": margin,
            "background_blur_radius": blur_radius,
            "source_to_target": [[scale, 0, left - scale * crop[0]],
                                 [0, scale, top - scale * crop[1]], [0, 0, 1]],
        }
    else:
        raise ValueError("unknown geometry: " + mode)
    geometry.update(mode=mode, source_size_hw=[sh, sw], target_size_hw=[height, width])
    return output, geometry


def content_mask(box, size_hw):
    height, width = size_hw
    yy, xx = np.mgrid[:height, :width]
    return ((xx + 0.5 >= box[0]) & (xx + 0.5 < box[2]) &
            (yy + 0.5 >= box[1]) & (yy + 0.5 < box[3])).astype(np.uint8)
