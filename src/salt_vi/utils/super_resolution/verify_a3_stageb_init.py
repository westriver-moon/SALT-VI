#!/usr/bin/env python3
"""Strictly audit a frozen-visual StageB initialization checkpoint."""
import argparse
import json
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from salt_vi.engine.build import CLIP2ReID
from salt_vi.entrypoints.train import _load_compatible_state_dict
from salt_vi.optim import build_optimizer
from salt_vi.utils.utils import load_train_configs


def verify(config_path):
    config = load_train_configs(str(config_path))
    expected = {
        "img_size": [512, 256],
        "sysu_source_size": [256, 128],
        "sysu_sr_modalities": ["rgb", "ir"],
        "sysu_sr_exact_size": True,
        "visual_forward_chunk_size": 8,
        "test_batch_size": 8,
        "Fix_Visual": True,
        "visual_unfreeze_last_n_blocks": 0,
        "fixed_visual_data_parallel": True,
        "fixed_visual_device_ids": [0, 1, 2, 3],
        "DataParallel": False,
        "fusion_way": "parameter_add",
        "pa": 0.5,
    }
    for key, expected_value in expected.items():
        value = getattr(config, key)
        if list(value) != expected_value if isinstance(expected_value, list) else value != expected_value:
            raise RuntimeError(f"StageB config mismatch for {key}: {value!r} != {expected_value!r}")
    data_root = Path(config.sysu_sr_data_root)
    for name in ("train_rgb_swinir_x2_img.npy", "train_ir_swinir_x2_img.npy", "manifest.json"):
        if not (data_root / name).is_file():
            raise FileNotFoundError(data_root / name)

    model = CLIP2ReID(config, num_classes=int(config.pid_num))
    _load_compatible_state_dict(model, config.training_weight_init, torch.device("cpu"))
    visual = [(name, parameter) for name, parameter in model.named_parameters() if name.startswith("base_model.visual")]
    if not visual or any(parameter.requires_grad for _, parameter in visual):
        raise RuntimeError("Visual backbone is not completely frozen")
    optimizer = build_optimizer(config, model)
    name_by_id = {id(parameter): name for name, parameter in model.named_parameters()}
    optimizer_names = [name_by_id[id(parameter)] for group in optimizer.param_groups for parameter in group["params"]]
    if any(name.startswith("base_model.visual") for name in optimizer_names):
        raise RuntimeError("Frozen visual parameter leaked into the optimizer")
    if not any(name.startswith("base_model.transformer") for name in optimizer_names):
        raise RuntimeError("Text encoder is missing from the optimizer")
    if not any(name.startswith("classifier.") for name in optimizer_names):
        raise RuntimeError("Classifier is missing from the optimizer")
    return {
        "config": str(Path(config_path).resolve()),
        "checkpoint": str(Path(config.training_weight_init).resolve()),
        "strict_load": True,
        "visual_parameter_tensors": len(visual),
        "visual_optimizer_tensors": 0,
        "optimizer_parameter_tensors": len(optimizer_names),
        "text_trainable": True,
        "classifier_trainable": True,
        "loss_names": config.loss_names,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.config), indent=2))


if __name__ == "__main__":
    main()
