import math

import torch


CROSS_MODAL_PAIR_NAMES = (
    "RGB-IR",
    "RGB-Fusion",
    "RGB-Text",
    "IR-Fusion",
    "IR-Text",
    "Fusion-Text",
)

IMAGE_TEXT_FUSION_WAYS = frozenset(
    {"norm_add", "add", "parameter_add", "adaptive_add", "cross_attention"}
)
JOINT_MODE_FUSION_WAYS = {
    "uni": IMAGE_TEXT_FUSION_WAYS,
    "ir_crossfusion": IMAGE_TEXT_FUSION_WAYS,
}


def validate_fusion_compatibility(training_mode, joint_mode, fusion_way):
    if fusion_way not in IMAGE_TEXT_FUSION_WAYS:
        raise ValueError(
            f"Unsupported fusion_way {fusion_way!r}; expected one of "
            f"{sorted(IMAGE_TEXT_FUSION_WAYS)}"
        )
    if training_mode != "RGB_IR_Text":
        return
    if joint_mode not in JOINT_MODE_FUSION_WAYS:
        raise ValueError(
            f"Unsupported joint_mode {joint_mode!r}; expected one of "
            f"{sorted(JOINT_MODE_FUSION_WAYS)}"
        )


def extract_text_token_feat(text_map, caption_ids):
    indices = torch.arange(text_map.shape[0], device=text_map.device)
    return text_map[indices, caption_ids.argmax(dim=-1)].float()


def ensure_matching_feature_shape(**named_tensors):
    shapes = {name: tuple(tensor.shape) for name, tensor in named_tensors.items()}
    if len(set(shapes.values())) != 1:
        raise ValueError(f"SFF requires matching feature shapes, got {shapes}")


def resolve_cross_modal_pair_weights(raw_weights=None):
    if raw_weights is None:
        return {name: 1.0 for name in CROSS_MODAL_PAIR_NAMES}
    if not isinstance(raw_weights, dict):
        raise TypeError("cross_modal_pair_weights must be a YAML mapping")
    unknown = sorted(set(raw_weights) - set(CROSS_MODAL_PAIR_NAMES))
    missing = sorted(set(CROSS_MODAL_PAIR_NAMES) - set(raw_weights))
    if unknown or missing:
        raise ValueError(
            f"cross_modal_pair_weights keys mismatch; missing={missing}, unknown={unknown}"
        )
    result = {name: float(raw_weights[name]) for name in CROSS_MODAL_PAIR_NAMES}
    for name, value in result.items():
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"Invalid cross-modal pair weight {name}={value!r}")
    if sum(result.values()) <= 0.0:
        raise ValueError("cross_modal_pair_weights must have a positive total")
    return result


def weighted_cross_modal_pair_loss(pair_losses, raw_weights=None):
    weights = resolve_cross_modal_pair_weights(raw_weights)
    missing = sorted(set(CROSS_MODAL_PAIR_NAMES) - set(pair_losses))
    unknown = sorted(set(pair_losses) - set(CROSS_MODAL_PAIR_NAMES))
    if missing or unknown:
        raise ValueError(f"pair loss keys mismatch; missing={missing}, unknown={unknown}")
    ordered = torch.stack([pair_losses[name] for name in CROSS_MODAL_PAIR_NAMES])
    if all(value == 1.0 for value in weights.values()):
        return ordered.mean()
    weight_tensor = ordered.new_tensor([weights[name] for name in CROSS_MODAL_PAIR_NAMES])
    return (ordered * weight_tensor).sum() / weight_tensor.sum()
