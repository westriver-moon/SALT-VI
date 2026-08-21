"""Small, side-effect-free validation for supported runtime configurations."""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Integral, Real
from pathlib import Path

from salt_vi.attention import validate_attention_backend_runtime
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


def _require_integer(config, name, *, minimum=None, maximum=None, default=None):
    value = _value(config, name, default)
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value!r}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}, got {value!r}")


def _require_real(
    config,
    name,
    *,
    minimum=None,
    maximum=None,
    exclusive_minimum=False,
    exclusive_maximum=False,
    default=None,
):
    value = _value(config, name, default)
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number, got {value!r}")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite real number, got {value!r}")
    if minimum is not None:
        if (exclusive_minimum and value <= minimum) or value < minimum:
            comparison = ">" if exclusive_minimum else ">="
            raise ValueError(
                f"{name} must be {comparison} {minimum}, got {value!r}"
            )
    if maximum is not None:
        if (exclusive_maximum and value >= maximum) or value > maximum:
            comparison = "<" if exclusive_maximum else "<="
            raise ValueError(
                f"{name} must be {comparison} {maximum}, got {value!r}"
            )


def _validate_numeric_ranges(config):
    integer_fields = {
        "batch_size": (1, None),
        "test_batch_size": (1, None),
        "num_pos": (1, None),
        "num_workers": (0, None),
        "img_h": (1, None),
        "img_w": (1, None),
        "total_train_epoch": (1, None),
        "checkpoint_epoch": (1, None),
        "eval_epoch": (1, None),
        "eval_start_epoch": (0, None),
        "eval_num_regdb": (1, 10),
        "trial": (1, 10),
        "seed": (0, 2**32 - 1),
        "eval_caption_seed": (0, 2**32 - 1),
        "gallery_trials": (1, None),
        "max_save_model_num": (0, None),
        "text_length": (1, None),
        "vocab_size": (1, None),
        "pmt_depth": (1, None),
        "pmt_num_heads": (1, None),
        "pmt_progressive_epoch": (1, None),
        "pmt_mscm_transition_epochs": (0, None),
        "prj_output_dim": (1, None),
        "pid_num": (1, None),
        "stride_size": (1, None),
        "warmup_epochs": (0, None),
        "visual_unfreeze_last_n_blocks": (0, None),
        "visual_unfreeze_start_epoch": (0, None),
        "qbn_freeze_running_stats_epoch": (-1, None),
        "cross_modal_hard_start_epoch": (0, None),
        "cross_modal_hard_ramp_epochs": (0, None),
    }
    for name, (minimum, maximum) in integer_fields.items():
        _require_integer(
            config, name, minimum=minimum, maximum=maximum
        )

    real_fields = (
        ("temperature", 0.0, None, True, False),
        ("alpha", 0.0, 1.0, True, True),
        ("beta", 0.0, 1.0, True, True),
        ("momentum", 0.0, 1.0, False, True),
        ("gamma", 0.0, 1.0, True, False),
        ("pa", 0.0, 1.0, False, False),
        ("warmup_factor", 0.0, 1.0, False, False),
        ("target_lr", 0.0, None, False, False),
        ("target_lr_factor", 0.0, None, False, False),
        ("llm_aug_prob", 0.0, 1.0, False, False),
        ("lr_visual", 0.0, None, False, False),
        ("lr_txt", 0.0, None, False, False),
        ("lr_factor", 0.0, None, False, False),
        ("classifier_lr_factor", 0.0, None, False, False),
        ("power", 0.0, None, True, False),
        ("cross_modal_hard_weight", 0.0, None, False, False),
        ("rgb_consistency_weight", 0.0, None, False, False),
        ("ir_rgb_text_pair_weight", 0.0, None, False, False),
        ("ir_rgb_aux_weight", 0.0, None, False, False),
        ("cmm_loss_weight", 0.0, None, False, False),
        ("id_loss_weight", 0.0, None, False, False),
        ("wrt_loss_weight", 0.0, None, False, False),
        ("gradient_clip_norm", 0.0, None, False, False),
        ("patch_gem_p", 0.0, None, True, False),
        ("pmt_mlp_ratio", 0.0, None, True, False),
        ("pmt_triplet_margin", 0.0, None, False, False),
        ("pmt_mscm_qct_margin", 0.0, None, False, False),
        ("pmt_mscm_qct_weight", 0.0, None, False, False),
        ("pmt_mscm_qct_branch_weight", 0.0, None, False, False),
        ("pmt_backbone_lr_factor", 0.0, None, False, False),
        ("visual_layer_decay", 0.0, 1.0, True, False),
        ("hetero_center_margin", 0.0, None, False, False),
        ("hetero_center_weight", 0.0, None, False, False),
        ("rfa_probability", 0.0, 1.0, False, False),
        ("rfa_gaussian_sigma", 0.0, None, True, False),
        ("ema_decay", 0.0, 1.0, True, True),
        ("cosine_softmax_scale", 0.0, None, True, False),
    )
    for name, minimum, maximum, exclusive_min, exclusive_max in real_fields:
        _require_real(
            config,
            name,
            minimum=minimum,
            maximum=maximum,
            exclusive_minimum=exclusive_min,
            exclusive_maximum=exclusive_max,
        )
    lr = _value(config, "lr", None)
    if lr is not None:
        _require_real(config, "lr", minimum=0.0)
    img_size = _value(config, "img_size", None)
    if img_size is not None:
        if (
            not isinstance(img_size, (tuple, list))
            or len(img_size) != 2
            or any(isinstance(item, bool) or not isinstance(item, Integral) for item in img_size)
            or any(item <= 0 for item in img_size)
        ):
            raise ValueError(
                f"img_size must be a pair of positive integers, got {img_size!r}"
            )
    for pair_name in ("pmt_patch_size", "pmt_stride_size"):
        pair = _value(config, pair_name, None)
        if pair is None:
            continue
        if (
            not isinstance(pair, (tuple, list))
            or len(pair) != 2
            or any(isinstance(item, bool) or not isinstance(item, Integral) for item in pair)
            or any(item <= 0 for item in pair)
        ):
            raise ValueError(
                f"{pair_name} must be a pair of positive integers, got {pair!r}"
            )


def validate_selected_config_schema(selected, defaults, path, project_root):
    """Reject unknown keys in canonical active configs.

    Reproduction snapshots remain immutable historical evidence and are not
    silently reinterpreted as current runtime configurations.
    """
    try:
        relative = Path(path).resolve().relative_to(Path(project_root).resolve())
    except ValueError:
        return selected
    parts = relative.parts
    is_active_config = (
        len(parts) >= 3
        and parts[0] == "configs"
        and parts[1] in {"stage_a", "stage_b"}
        and "reproduction" not in parts
    )
    if not is_active_config:
        return selected
    unknown = sorted(set(selected) - set(defaults))
    if unknown:
        raise KeyError(
            "Unknown or unsupported key(s) in active config {}: {}".format(
                relative, ", ".join(unknown)
            )
        )
    if selected.get("training_weight_init") and not selected.get(
        "training_weight_init_sha256"
    ):
        raise ValueError(
            "Active config {} sets training_weight_init but omits "
            "training_weight_init_sha256".format(relative)
        )
    return selected


def validate_runtime_config(config):
    """Reject unsupported or structurally inconsistent training recipes early.

    This intentionally validates only contracts implemented by the canonical
    loader/model.  It does not prescribe research hyperparameters.
    """
    _validate_numeric_ranges(config)
    if int(_value(config, "resume_train_epoch", -1)) >= 0:
        raise ValueError(
            "model-only resume via resume_train_epoch is retired; convert the "
            "checkpoint to the run-manifest full-state schema before resuming"
        )
    if int(_value(config, "metric_boost_resume_epoch", 0)) > 0:
        raise ValueError(
            "metric_boost_resume_epoch is retired; start a fresh run or use "
            "complete run-manifest resume"
        )
    if bool(_value(config, "Return_B4_BN", False)):
        raise ValueError(
            "Return_B4_BN was a no-op and has been removed; normalized BN features are always returned"
        )
    training_mode = str(_value(config, "training_mode", ""))
    joint_mode = str(_value(config, "joint_mode", "image_only"))
    visual_input_backend = str(
        _value(config, "visual_input_backend", "single") or "single"
    ).lower()
    pmt_recipe_variant = str(
        _value(config, "pmt_recipe_variant", "original") or "original"
    ).lower()
    if pmt_recipe_variant not in {"original", "mscm_phased"}:
        raise ValueError(
            f"Unsupported pmt_recipe_variant {pmt_recipe_variant!r}"
        )
    if visual_input_backend not in {"single", "quadruple_patch"}:
        raise ValueError(f"Unsupported visual_input_backend {visual_input_backend!r}")
    if visual_input_backend == "quadruple_patch":
        expected_order = [
            "visible_global",
            "visible_channel",
            "infrared_global",
            "infrared_channel",
        ]
        branch_order = list(_value(config, "quadruple_branch_order", expected_order))
        if branch_order != expected_order:
            raise ValueError(
                f"quadruple_branch_order must be {expected_order}, got {branch_order}"
            )
        if str(_value(config, "pretrain_choice", "")) != "PMT_VIT":
            raise ValueError("quadruple_patch requires pretrain_choice='PMT_VIT'")
        if _value(config, "pmt_patch_embed", None) is not None:
            raise ValueError("quadruple_patch cannot be combined with fused pmt_patch_embed")
        if not bool(_value(config, "pmt_recipe", False)):
            raise ValueError("quadruple_patch currently requires the Stage-A PMT recipe")
        if training_mode != "RGB_IR":
            raise ValueError("quadruple_patch currently requires training_mode='RGB_IR'")
        if str(_value(config, "dataset", "")).lower() != "sysu":
            raise ValueError("quadruple_patch is currently implemented for SYSU Stage A")
        if bool(_value(config, "Fix_Visual", False)):
            raise ValueError("quadruple_patch Stage A requires a trainable visual backbone")
        sr_modalities = {
            str(value).lower() for value in (_value(config, "sysu_sr_modalities", []) or [])
        }
        if sr_modalities != {"rgb", "ir"} or not _value(config, "sysu_sr_data_root"):
            raise ValueError(
                "quadruple_patch requires RGB and IR super-resolution assets; augmentations "
                "are applied only after the SR source returns each image"
            )
    if pmt_recipe_variant == "mscm_phased":
        if not bool(_value(config, "pmt_recipe", False)):
            raise ValueError("mscm_phased requires pmt_recipe=true")
        if visual_input_backend != "quadruple_patch":
            raise ValueError("mscm_phased requires visual_input_backend='quadruple_patch'")
        if not bool(_value(config, "quadruple_template_trainable", False)):
            raise ValueError(
                "mscm_phased requires quadruple_template_trainable=true for PMT warmup"
            )
        if str(_value(config, "pmt_metric_loss", "legacy")) != "legacy":
            raise ValueError("mscm_phased requires pmt_metric_loss='legacy'")
        if str(_value(config, "triplet_mining", "pmt_hard")) != "pmt_cross_modal_hard":
            raise ValueError(
                "mscm_phased post-switch stage requires "
                "triplet_mining='pmt_cross_modal_hard'"
            )
        if float(_value(config, "rfa_probability", 0.0)) != 0.0:
            raise ValueError(
                "mscm_phased requires rfa_probability=0 so post-switch inputs "
                "remain the exact MSCMNet augmentations"
            )
    uses_text = "Text" in training_mode
    metric_loss = str(_value(config, "pmt_metric_loss", "legacy"))
    if metric_loss not in {"legacy", "hetero_center"}:
        raise ValueError(f"Unsupported pmt_metric_loss {metric_loss!r}")
    sampler_type = str(_value(config, "sampler_type", "identity_current_replace"))
    supported_samplers = {
        "identity_current_replace",
        "identity_auto_replace",
        "identity_camera_diverse",
    }
    if sampler_type not in supported_samplers:
        raise ValueError(f"Unsupported sampler_type {sampler_type!r}")
    if sampler_type == "identity_camera_diverse" and str(
        _value(config, "dataset", "")
    ).lower() != "sysu":
        raise ValueError("identity_camera_diverse sampler requires dataset='sysu'")
    attention_backend = validate_attention_backend_runtime(
        _value(config, "pmt_attention_backend", "manual")
    )
    if attention_backend != "manual" and str(
        _value(config, "pretrain_choice", "")
    ) != "PMT_VIT":
        raise ValueError(
            "pmt_attention_backend is only implemented for pretrain_choice='PMT_VIT'"
        )

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
        if views not in (0, 1, 5):
            raise ValueError("pasd_multiview requires dynamic, one, or five views per image")
        if not _value(config, "sysu_sr_view_manifest"):
            raise ValueError("pasd_multiview requires sysu_sr_view_manifest")
        sampling = str(_value(config, "sysu_sr_view_sampling", "independent")).lower()
        if sampling not in ("independent", "paired"):
            raise ValueError("sysu_sr_view_sampling must be independent or paired")
        eval_index = int(_value(config, "sysu_sr_eval_view_index", 0))
        if eval_index < 0 or (views and eval_index >= views):
            raise ValueError(f"sysu_sr_eval_view_index must be in [0, {views - 1}]")
        if (int(_value(config, "img_h", 0)), int(_value(config, "img_w", 0))) != (512, 256):
            raise ValueError("pasd_multiview requires img_h=512 and img_w=256")

    retrieval_protocol = get_retrieval_protocol(
        _value(config, "retrieval_backend", "identity_text")
    )
    retrieval_protocol.validate(
        config,
        sr_backend=sr_backend,
        sr_modalities=modalities if sr_backend == "pasd_multiview" else set(),
    )
    return config
