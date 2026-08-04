"""Small, side-effect-free validation for supported runtime configurations."""

from __future__ import annotations

from collections.abc import Mapping


# The loader materializes these two text-batch contracts.  The other historic
# names are deliberately not accepted as public runtime modes.
SUPPORTED_JOINT_MODES = ("image_only", "ir_crossfusion", "uni")
SUPPORTED_TEXT_JOINT_MODES = ("ir_crossfusion", "uni")


def _value(config, name, default=None):
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _loss_names(value):
    if value is None:
        return set()
    if isinstance(value, str):
        return {item.strip() for item in value.split(",") if item.strip()}
    return {str(item).strip() for item in value if str(item).strip()}


def validate_runtime_config(config):
    """Reject unsupported or structurally inconsistent training recipes early.

    This intentionally validates only contracts implemented by the canonical
    loader/model.  It does not prescribe research hyperparameters.
    """
    training_mode = str(_value(config, "training_mode", ""))
    joint_mode = str(_value(config, "joint_mode", "image_only"))
    uses_text = "Text" in training_mode

    if joint_mode not in SUPPORTED_JOINT_MODES:
        raise ValueError(
            f"Unsupported joint_mode {joint_mode!r}; supported modes are "
            f"{list(SUPPORTED_JOINT_MODES)}"
        )
    if uses_text and joint_mode not in SUPPORTED_TEXT_JOINT_MODES:
        raise ValueError(
            f"training_mode {training_mode!r} requires one of "
            f"{list(SUPPORTED_TEXT_JOINT_MODES)}, got {joint_mode!r}"
        )

    uni_bn = bool(_value(config, "uni_BN", False))
    losses = _loss_names(_value(config, "loss_names", ""))
    if uni_bn and joint_mode != "uni":
        raise ValueError("uni_BN requires joint_mode='uni'")
    if uni_bn and not uses_text:
        raise ValueError("uni_BN requires a text-enabled training_mode")
    if uni_bn and "id_woir" in losses:
        raise ValueError(
            "uni_BN is incompatible with id_woir: the classifier requires "
            "five modality groups while id_woir produces four"
        )

    if bool(_value(config, "fixed_visual_data_parallel", False)) and not bool(
        _value(config, "Fix_Visual", False)
    ):
        raise ValueError("fixed_visual_data_parallel requires Fix_Visual=true")
    if bool(_value(config, "fixed_visual_data_parallel", False)) and int(
        _value(config, "visual_unfreeze_last_n_blocks", 0) or 0
    ) > 0:
        raise ValueError(
            "fixed_visual_data_parallel cannot be combined with visual branch unfreezing"
        )
    return config
