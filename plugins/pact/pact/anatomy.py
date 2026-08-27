"""COCO-17 -> four soft anatomical masks on the final image and ViT patch grid."""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from person_preprocessing.geometry import content_mask


PARTS = ("head", "torso", "arms", "legs")


def patch_average(masks, patch_hw=(16, 16), stride_hw=(12, 12)):
    """Average the exact anchor-patch footprint, not a resized heatmap."""
    _, height, width = masks.shape
    ph, pw = patch_hw
    sy, sx = stride_hw
    integral = np.pad(masks.astype(np.float32), ((0, 0), (1, 0), (1, 0)))
    integral = integral.cumsum(1).cumsum(2)
    y, x = np.meshgrid(np.arange(0, height - ph + 1, sy),
                       np.arange(0, width - pw + 1, sx), indexing="ij")
    pooled = (integral[:, y + ph, x + pw] - integral[:, y, x + pw]
              - integral[:, y + ph, x] + integral[:, y, x]) / (ph * pw)
    return np.clip(pooled, 0, 1).astype(np.float32)


def anatomical_masks(points, size_hw, foreground_box, threshold=0.3,
                     patch_hw=(16, 16), stride_hw=(12, 12)):
    height, width = size_hw
    points = np.asarray(points, dtype=np.float32)
    if points.shape != (17, 3) or not np.isfinite(points).all():
        raise ValueError("expected finite COCO-17 x,y,confidence keypoints")
    content = content_mask(foreground_box, size_hw)
    valid = ((points[:, 2] >= threshold) & (points[:, 0] >= foreground_box[0]) &
             (points[:, 0] < foreground_box[2]) & (points[:, 1] >= foreground_box[1]) &
             (points[:, 1] < foreground_box[3]))
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    masks = np.zeros((4, height, width), np.float32)
    quality = np.zeros(4, np.float32)
    sigma = max(2.0, (foreground_box[2] - foreground_box[0]) * 0.035)

    def tube(a, b, radius):
        p, q = points[a, :2], points[b, :2]
        dx, dy = q - p
        denominator = max(float(dx * dx + dy * dy), 1e-6)
        t = np.clip(((xx - p[0]) * dx + (yy - p[1]) * dy) / denominator, 0, 1)
        distance = (xx - p[0] - t * dx) ** 2 + (yy - p[1] - t * dy) ** 2
        return np.where(distance <= 9 * radius ** 2,
                        np.exp(-distance / (2 * radius ** 2)), 0)

    head = np.flatnonzero(valid[:5])
    if len(head):
        center = np.average(points[head, :2], axis=0, weights=points[head, 2])
        radius = sigma * 2.2
        distance = (xx - center[0]) ** 2 + (yy - center[1]) ** 2
        masks[0] = np.where(distance <= 9 * radius ** 2,
                            np.exp(-distance / (2 * radius ** 2)), 0)
        quality[0] = points[head, 2].mean() * len(head) / 5
    torso = [5, 6, 12, 11]
    if valid[torso].all():
        polygon = Image.new("L", (width, height))
        ImageDraw.Draw(polygon).polygon([tuple(points[i, :2]) for i in torso], fill=255)
        masks[1] = np.asarray(polygon.filter(ImageFilter.GaussianBlur(sigma / 2)), dtype=np.float32) / 255
        quality[1] = points[torso, 2].min()
    for part, segments in ((2, ((5, 7), (7, 9), (6, 8), (8, 10))),
                           (3, ((11, 13), (13, 15), (12, 14), (14, 16)))):
        for a, b in segments:
            if valid[a] and valid[b]:
                masks[part] = np.maximum(masks[part], tube(a, b, sigma * (1.3 if part == 3 else 1)))
                quality[part] += min(points[a, 2], points[b, 2]) / len(segments)
    masks *= content[None]
    masks /= np.maximum(masks.sum(0, keepdims=True), 1)
    # Pose does not segment true background. Only padding is known background;
    # uncovered pixels inside the foreground remain explicitly unassigned.
    background = 1 - content.astype(np.float32)
    residual = np.clip(content - masks.sum(0), 0, 1)
    token_masks = patch_average(masks, patch_hw, stride_hw)
    return {"keypoints": points, "pixel_masks": masks.astype(np.float16),
            "background_mask": background.astype(np.float16), "content_mask": content,
            "residual_mask": residual.astype(np.float16),
            "token_masks": token_masks.astype(np.float16),
            "token_background": patch_average(background[None], patch_hw, stride_hw)[0].astype(np.float16),
            "token_residual": patch_average(residual[None], patch_hw, stride_hw)[0].astype(np.float16),
            "part_confidence": quality, "part_valid": (quality > 0) & (token_masks.sum((1, 2)) > 0),
            "part_names": np.asarray(PARTS), "size_hw": np.asarray(size_hw),
            "patch_hw": np.asarray(patch_hw), "stride_hw": np.asarray(stride_hw)}
