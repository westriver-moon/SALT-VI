from __future__ import annotations

from dataclasses import dataclass
from typing import Any


CATEGORY_STATES = {
    "eyewear": ("absent", "eyewear_type", "frame_style", "lens_detail", "no_additional_detail"),
    "wrist_accessory": ("absent", "watch", "bracelet", "wristband", "no_additional_detail"),
    "headwear": ("absent", "cap", "hat", "hood", "other_headwear", "no_additional_detail"),
    "body_marking": ("absent", "tattoo", "scar", "no_additional_detail"),
    "clothing_detail": (
        "graphic", "pattern", "color_detail", "sleeve_detail",
        "other_clothing_detail", "no_additional_detail",
    ),
    "carried_object": (
        "absent", "backpack", "shoulder_bag", "bottle", "bag_accessory",
        "other_carried_object", "no_additional_detail",
    ),
    "pocket_item": (
        "absent", "zipper", "phone", "keys", "wallet",
        "other_pocket_item", "no_additional_detail",
    ),
    "footwear_detail": (
        "laces", "strap", "toe_style", "sock_detail", "color_detail",
        "texture", "other_footwear_detail", "no_additional_detail",
    ),
}


@dataclass(frozen=True)
class Region:
    region_id: str
    category: str
    bbox_xyxy: tuple[int, int, int, int]
    allowed_states: tuple[str, ...]
    uncertainty: float = 0.0
    blur_uncertainty: float = 0.0
    mask: Any | None = None

    def validate(self) -> "Region":
        left, top, right, bottom = self.bbox_xyxy
        if right <= left or bottom <= top:
            raise ValueError("ROI bbox must have positive area")
        if not self.region_id or not self.category or not self.allowed_states:
            raise ValueError("ROI id, category and allowed states are required")
        if self.category not in CATEGORY_STATES:
            raise ValueError(f"unsupported ROI category: {self.category}")
        if not set(self.allowed_states).issubset(CATEGORY_STATES[self.category]):
            raise ValueError("ROI allowed states do not match the category taxonomy")
        return self
