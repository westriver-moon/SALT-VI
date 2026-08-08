"""Small, side-effect-free validation for supported runtime configurations."""

from __future__ import annotations

from collections.abc import Mapping

from salt_vi.retrieval import get_retrieval_backend


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
        if modalities != {"rgb", "ir"}:
            raise ValueError("pasd_multiview requires sysu_sr_modalities=[rgb, ir]")
        if not bool(_value(config, "sysu_sr_exact_size", False)):
            raise ValueError("pasd_multiview requires sysu_sr_exact_size=true")
        if int(_value(config, "sysu_sr_views_per_image", 0)) != 5:
            raise ValueError("pasd_multiview requires exactly five views per image")
        if not _value(config, "sysu_sr_view_manifest"):
            raise ValueError("pasd_multiview requires sysu_sr_view_manifest")
        sampling = str(_value(config, "sysu_sr_view_sampling", "independent")).lower()
        if sampling not in ("independent", "paired"):
            raise ValueError("sysu_sr_view_sampling must be independent or paired")
        eval_index = int(_value(config, "sysu_sr_eval_view_index", 0))
        if not 0 <= eval_index < 5:
            raise ValueError("sysu_sr_eval_view_index must be in [0, 4]")
        if (int(_value(config, "img_h", 0)), int(_value(config, "img_w", 0))) != (512, 256):
            raise ValueError("pasd_multiview requires img_h=512 and img_w=256")

    retrieval_backend = get_retrieval_backend(
        _value(config, "retrieval_backend", "legacy")
    )
    if retrieval_backend:
        if str(_value(config, "dataset", "")).lower() != "sysu":
            raise ValueError("ir_to_rgb_text is supported only for SYSU-MM01")
        if training_mode != "RGB_IR_Text" or joint_mode != "uni":
            raise ValueError("ir_to_rgb_text requires training_mode=RGB_IR_Text and joint_mode=uni")
        if bool(_value(config, "Feat_Filter", False)):
            raise ValueError("ir_to_rgb_text does not use IR caption filtering")
        if uni_bn:
            raise ValueError("ir_to_rgb_text requires the shared classifier BN")
        if sr_backend != "array":
            raise ValueError("ir_to_rgb_text currently consumes the SwinIR array backend")
        if str(_value(config, "test_modality", "")) != retrieval_backend.RESULT_KEY:
            raise ValueError(
                f"ir_to_rgb_text requires test_modality={retrieval_backend.RESULT_KEY}"
            )
        if not _value(config, "gallery_caption_manifest"):
            raise ValueError("ir_to_rgb_text requires gallery_caption_manifest")
        text_dropout = float(_value(config, "gallery_text_dropout", 0.0))
        if not 0.0 <= text_dropout < 1.0:
            raise ValueError("gallery_text_dropout must be in [0, 1)")
    return config
