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

STATE_VALUE_EVIDENCE: Mapping[str, frozenset[str]] = {
    "eyewear_type": frozenset({"glass", "glasses", "sunglass", "sunglasses", "spectacle", "goggle"}),
    "frame_style": frozenset({"frame", "rim", "round", "rectangular", "square", "oval"}),
    "lens_detail": frozenset({"lens", "tinted", "clear", "reflective"}),
    "watch": frozenset({"watch", "timepiece"}),
    "bracelet": frozenset({"bracelet", "bangle"}),
    "wristband": frozenset({"wristband", "band"}),
    "cap": frozenset({"cap", "baseball"}),
    "hat": frozenset({"hat", "beanie"}),
    "hood": frozenset({"hood", "hooded"}),
    "other_headwear": frozenset({"helmet", "headband", "turban"}),
    "tattoo": frozenset({"tattoo", "ink"}),
    "scar": frozenset({"scar"}),
    "graphic": frozenset({"graphic", "logo", "heart", "letter", "print", "symbol"}),
    "pattern": frozenset({"pattern", "stripe", "striped", "plaid", "check", "checked", "dot"}),
    "sleeve_detail": frozenset({"sleeve", "sleeveless", "cuff"}),
    "backpack": frozenset({"backpack", "rucksack"}),
    "shoulder_bag": frozenset({"bag", "purse", "satchel", "tote"}),
    "bottle": frozenset({"bottle", "flask"}),
    "bag_accessory": frozenset({"strap", "tag", "charm", "accessory"}),
    "other_carried_object": frozenset({"umbrella", "book", "parcel", "phone"}),
    "zipper": frozenset({"zipper", "zip"}),
    "phone": frozenset({"phone", "mobile"}),
    "keys": frozenset({"key", "keys"}),
    "wallet": frozenset({"wallet"}),
    "other_pocket_item": frozenset({"card", "pen", "earphone", "tissue"}),
    "laces": frozenset({"lace", "laces", "shoelace"}),
    "strap": frozenset({"strap", "straps"}),
    "toe_style": frozenset({"toe", "open", "closed"}),
    "sock_detail": frozenset({"sock", "socks"}),
    "texture": frozenset({"texture", "leather", "suede", "canvas", "mesh"}),
    "other_footwear_detail": frozenset({"sole", "heel", "buckle"}),
}

COLOR_TOKENS = frozenset(
    {"black", "white", "gray", "grey", "red", "blue", "green", "yellow", "brown", "beige", "dark", "light"}
)


def normalize_symbol(value: str) -> str:
    import re

    return re.sub(r"[^a-z0-9_]+", "_", value.strip().casefold()).strip("_")


def state_value_compatible(state: str, value: str) -> bool:
    normalized_value = normalize_symbol(value)
    if state in SENTINEL_STATES:
        return normalized_value == state
    if normalized_value in SENTINEL_STATES:
        return False

    tokens = set(normalized_value.split("_"))
    if state == "color_detail":
        return bool(tokens & COLOR_TOKENS)
    evidence = STATE_VALUE_EVIDENCE.get(state)
    return evidence is None or bool(tokens & evidence)
