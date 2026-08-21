from __future__ import annotations

from collections.abc import Mapping


CATEGORY_STATES: Mapping[str, frozenset[str]] = {
    "eyewear": frozenset(
        {"absent", "eyewear_type", "frame_style", "lens_detail", "no_additional_detail"}
    ),
    "wrist_accessory": frozenset(
        {"absent", "watch", "bracelet", "wristband", "no_additional_detail"}
    ),
    "headwear": frozenset(
        {"absent", "cap", "hat", "hood", "other_headwear", "no_additional_detail"}
    ),
    "body_marking": frozenset(
        {"absent", "tattoo", "scar", "no_additional_detail"}
    ),
    "clothing_detail": frozenset(
        {
            "graphic",
            "pattern",
            "color_detail",
            "sleeve_detail",
            "other_clothing_detail",
            "no_additional_detail",
        }
    ),
    "carried_object": frozenset(
        {
            "absent",
            "backpack",
            "shoulder_bag",
            "bottle",
            "bag_accessory",
            "other_carried_object",
            "no_additional_detail",
        }
    ),
    "pocket_item": frozenset(
        {"absent", "zipper", "phone", "keys", "wallet", "other_pocket_item", "no_additional_detail"}
    ),
    "footwear_detail": frozenset(
        {
            "laces",
            "strap",
            "toe_style",
            "sock_detail",
            "color_detail",
            "texture",
            "other_footwear_detail",
            "no_additional_detail",
        }
    ),
}

DEFAULT_SAMPLING_STRATA = tuple(CATEGORY_STATES)
SENTINEL_STATES = frozenset({"absent", "no_additional_detail"})

COLOR_TOKENS = frozenset(
    {"black", "white", "gray", "grey", "red", "blue", "green", "yellow", "brown", "beige", "dark", "light"}
)

STATE_VALUE_EVIDENCE: Mapping[str, Mapping[str, frozenset[str]]] = {
    "eyewear": {
        "eyewear_type": frozenset({"glass", "glasses", "sunglass", "sunglasses", "spectacle", "goggle"}),
        "frame_style": frozenset({"frame", "rim", "round", "rectangular", "square", "oval"}),
        "lens_detail": frozenset({"lens", "tinted", "clear", "reflective"}),
    },
    "wrist_accessory": {
        "watch": frozenset({"watch", "timepiece", "band", "strap"}),
        "bracelet": frozenset({"bracelet", "bangle"}),
        "wristband": frozenset({"wristband", "band"}),
    },
    "headwear": {
        "cap": frozenset({"cap", "baseball"}),
        "hat": frozenset({"hat", "beanie"}),
        "hood": frozenset({"hood", "hooded"}),
        "other_headwear": frozenset({"helmet", "headband", "turban"}),
    },
    "body_marking": {
        "tattoo": frozenset({"tattoo", "ink"}),
        "scar": frozenset({"scar"}),
    },
    "clothing_detail": {
        "graphic": frozenset({"graphic", "logo", "heart", "letter", "print", "symbol"}),
        "pattern": frozenset({"pattern", "stripe", "striped", "plaid", "check", "checked", "dot"}),
        "color_detail": COLOR_TOKENS,
        "sleeve_detail": frozenset({"sleeve", "sleeveless", "cuff"}),
    },
    "carried_object": {
        "backpack": frozenset({"backpack", "rucksack", "bag"}),
        "shoulder_bag": frozenset({"bag", "purse", "satchel", "tote"}),
        "bottle": frozenset({"bottle", "flask"}),
        "bag_accessory": frozenset({"strap", "tag", "charm", "accessory"}),
        "other_carried_object": frozenset({"umbrella", "book", "parcel", "phone"}),
    },
    "pocket_item": {
        "zipper": frozenset({"zipper", "zip"}),
        "phone": frozenset({"phone", "mobile"}),
        "keys": frozenset({"key", "keys"}),
        "wallet": frozenset({"wallet"}),
        "other_pocket_item": frozenset({"card", "pen", "earphone", "tissue"}),
    },
    "footwear_detail": {
        "laces": frozenset({"lace", "laces", "shoelace"}),
        "strap": frozenset({"strap", "straps"}),
        "toe_style": frozenset({"toe", "open", "closed"}),
        "sock_detail": frozenset({"sock", "socks"}),
        "color_detail": COLOR_TOKENS,
        "texture": frozenset({"texture", "leather", "suede", "canvas", "mesh"}),
        "other_footwear_detail": frozenset({"sole", "heel", "buckle"}),
    },
}

def normalize_symbol(value: str) -> str:
    import re

    return re.sub(r"[^a-z0-9_]+", "_", value.strip().casefold()).strip("_")


def evidence_for(category: str, state: str) -> frozenset[str]:
    return STATE_VALUE_EVIDENCE.get(category, {}).get(state, frozenset())
