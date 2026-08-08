from __future__ import annotations

from pathlib import Path


def _variant_directory(config) -> str:
    enabled = tuple(
        name
        for name, active in (
            ("FV", bool(config.Fix_Visual)),
            ("Filter", bool(config.Feat_Filter)),
            ("QBN", bool(config.uni_BN)),
        )
        if active
    )
    return "_".join(enabled) if enabled else "Base"


def build_experiment_name(config) -> str:
    name = "Baseline"
    if config.dataset == "regdb":
        name += f"_{config.trial}"
    name += f"_train[{config.training_mode}]"
    if len(config.training_mode.split("_")) == 3:
        name += f"_joint[{config.joint_mode}]"
    if "Text" in config.training_mode:
        name += f"_{config.captioner_name}"
        if "IR" in config.training_mode:
            name += f"_{config.fusion_way}"
        if config.llm_aug:
            name += f"_LLM_{config.llm_aug_prob}"
    if config.loss_names:
        name += f"_{config.loss_names}"
    if config.Return_B4_BN:
        name += "_Return_B4_BN"
    if config.uni_BN:
        name += "_uni_BN"
    if config.Fix_Visual:
        name += "_Fix_Visual"
    if config.Feat_Filter:
        name += "_Filtered"
    return name


def resolve_run_directory(config) -> str:
    if config.DEBUG:
        return str(Path(config.DEBUG_DIR).expanduser())
    is_resume = bool(config.auto_resume_training_from_lastest_step) or config.resume_train_epoch >= 0
    if config.mode == "test" or is_resume:
        if not getattr(config, "output_path", None):
            raise ValueError("test and resume require the final output_path")
        return str(Path(config.output_path).expanduser())
    root = getattr(config, "output_root", None) or getattr(config, "output_path", None)
    if not root:
        raise ValueError("fresh training requires output_root")
    experiment_name = getattr(config, "experiment_name", None) or build_experiment_name(config)
    return str(
        Path(root).expanduser()
        / str(config.dataset)
        / _variant_directory(config)
        / experiment_name
    )
