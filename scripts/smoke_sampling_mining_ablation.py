#!/usr/bin/env python
import gc
import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch

from salt_vi.engine import build_model
from salt_vi.data.loader import Loader
from salt_vi.utils.utils import load_train_configs


CONFIGS = [
    "configs/stage_a/sampling_mining_ablation/s0_pk8x4_current_replace_hard.yaml",
    "configs/stage_a/sampling_mining_ablation/s1_pk8x4_auto_replace_hard.yaml",
    "configs/stage_a/sampling_mining_ablation/s2_pk16x2_auto_replace_hard.yaml",
    "configs/stage_a/sampling_mining_ablation/s3_pk4x8_auto_replace_hard.yaml",
    "configs/stage_a/sampling_mining_ablation/h1_pk8x4_auto_replace_wrt.yaml",
    "configs/stage_a/sampling_mining_ablation/h5_pk8x4_auto_replace_crossmodal_hard.yaml",
]


def set_pid_num(config):
    if config.dataset == "sysu":
        config.pid_num = 395
    elif config.dataset == "regdb":
        config.pid_num = 206
    elif config.dataset == "llcm":
        config.pid_num = 713
    else:
        raise ValueError(f"Unsupported dataset: {config.dataset}")


def assert_chunk_layout(labels, num_pos, name):
    if labels.numel() % num_pos != 0:
        raise AssertionError(f"{name} length {labels.numel()} is not divisible by num_pos={num_pos}")
    chunks = labels.view(-1, num_pos)
    if not torch.all(chunks.eq(chunks[:, :1])):
        raise AssertionError(f"{name} does not have one identity per consecutive num_pos chunk")


def move_batch(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def smoke_one(config_path):
    config = load_train_configs(config_path)
    set_pid_num(config)
    config.gpu_id = "0"
    config.CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[SMOKE] {Path(config_path).name} sampler={getattr(config, 'sampler_type', None)} mining={getattr(config, 'triplet_mining', None)} device={device}")

    loaders = Loader(config)
    train_loader = loaders.get_train_loader()
    batch = next(iter(train_loader))

    target_rgb = batch["target_rgb"].long()
    target_ir = batch["target_ir"].long()
    if target_rgb.shape != target_ir.shape:
        raise AssertionError(f"target_rgb shape {tuple(target_rgb.shape)} != target_ir shape {tuple(target_ir.shape)}")
    if bool(getattr(config, "pmt_recipe", False)) and not torch.equal(target_rgb, target_ir):
        raise AssertionError("PMT recipe requires target_rgb == target_ir")
    assert_chunk_layout(target_rgb, int(config.num_pos), "target_rgb")
    assert_chunk_layout(target_ir, int(config.num_pos), "target_ir")

    model = build_model(config).to(device)
    model.set_train()
    batch = move_batch(batch, device)
    rgb_stage_epoch = int(getattr(config, "pmt_progressive_epoch", 6))
    ret = model(batch, mode=None, current_epoch=rgb_stage_epoch)

    required = ["id_loss", "tri_loss", "msel_loss", "dcl_loss"]
    for key in required:
        if key not in ret:
            raise AssertionError(f"missing loss: {key}")
        value = ret[key]
        if not torch.is_tensor(value):
            raise AssertionError(f"{key} is not a tensor")
        if not torch.isfinite(value.detach()).all():
            raise AssertionError(f"{key} is not finite: {value}")

    total_loss = sum(value for key, value in ret.items() if "loss" in key)
    if not torch.isfinite(total_loss.detach()).all():
        raise AssertionError(f"total_loss is not finite: {total_loss}")
    total_loss.backward()
    print(f"[PASS] {Path(config_path).name}")

    del model, loaders, train_loader, batch, ret, total_loss
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    for config_path in CONFIGS:
        try:
            smoke_one(config_path)
        except Exception:
            print(f"[FAIL] {Path(config_path).name}")
            traceback.print_exc()
            return 1
    print("[ALL PASS] sampling/mining ablation smoke tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
