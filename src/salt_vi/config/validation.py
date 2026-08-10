"""Small, side-effect-free validation for supported runtime configurations."""

from __future__ import annotations

from collections.abc import Mapping

from salt_vi.retrieval import get_retrieval_protocol


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
    sr_backend = str(_value(config, "sysu_sr_backend", "array") or "array").lower()
    if sr_backend not in ("array", "pasd_multiview"):
        raise ValueError(f"Unsupported sysu_sr_backend {sr_backend!r}")
    if sr_backend == "pasd_multiview":
        if str(_value(config, "dataset", "")).lower() != "sysu":
            raise ValueError("pasd_multiview is supported only for SYSU-MM01")
        modalities = {
            str(value).lower() for value in (_value(config, "sysu_sr_modalities", []) or [])
        }
        if not modalities or not modalities.issubset({"rgb", "ir"}):
            raise ValueError("pasd_multiview requires rgb and/or ir SR modalities")
        if not bool(_value(config, "sysu_sr_exact_size", False)):
            raise ValueError("pasd_multiview requires sysu_sr_exact_size=true")
        views = int(_value(config, "sysu_sr_views_per_image", 0))
        if views not in (1, 5):
            raise ValueError("pasd_multiview requires one or five views per image")
        if not _value(config, "sysu_sr_view_manifest"):
            raise ValueError("pasd_multiview requires sysu_sr_view_manifest")
        sampling = str(_value(config, "sysu_sr_view_sampling", "independent")).lower()
        if sampling not in ("independent", "paired"):
            raise ValueError("sysu_sr_view_sampling must be independent or paired")
        eval_index = int(_value(config, "sysu_sr_eval_view_index", 0))
        if not 0 <= eval_index < views:
            raise ValueError(f"sysu_sr_eval_view_index must be in [0, {views - 1}]")
        if (int(_value(config, "img_h", 0)), int(_value(config, "img_w", 0))) != (512, 256):
            raise ValueError("pasd_multiview requires img_h=512 and img_w=256")

    retrieval_protocol = get_retrieval_protocol(
        _value(config, "retrieval_backend", "legacy")
    )
    retrieval_protocol.validate(
        config,
        sr_backend=sr_backend,
        sr_modalities=modalities if sr_backend == "pasd_multiview" else set(),
    )
    return config
