#!/usr/bin/env python3
"""Lossless-equivalence and throughput gate for A3 StageB frozen-visual parallelism."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import torch
from torch.cuda import amp

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from salt_vi.engine import build_model
from salt_vi.data.loader import Loader
from salt_vi.entrypoints.train import _load_fixed_visual_init, seed_torch
from salt_vi.optim import build_optimizer, build_lr_scheduler
from salt_vi.utils.super_resolution.verify_a3_stageb_init import verify
from salt_vi.utils.utils import load_train_configs


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_tree_equal(left, right):
    if torch.is_tensor(left):
        return torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            tensor_tree_equal(left[key], right[key]) for key in left
        )
    return left == right


def tensor_tree_max_abs(left, right):
    if torch.is_tensor(left):
        return float((left.float() - right.float()).abs().max().cpu())
    if isinstance(left, dict):
        return max(tensor_tree_max_abs(left[key], right[key]) for key in left)
    return 0.0


def tensor_digest(named_tensors):
    digest = hashlib.sha256()
    count = 0
    for name, tensor in named_tensors:
        digest.update(name.encode("utf-8"))
        value = tensor.detach().cpu().contiguous()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
        count += 1
    return digest.hexdigest(), count


def build_runtime(config, parallel):
    seed_torch(config.seed)
    device = torch.device("cuda:0")
    model = build_model(config)
    _load_fixed_visual_init(model, config, device)
    model = model.to(device)
    if parallel:
        model.configure_fixed_visual_data_parallel()
    optimizer = build_optimizer(config, model)
    scheduler = build_lr_scheduler(config, optimizer)
    scheduler.step(0)
    model.configure_epoch_trainability(0)
    model.set_train()
    return model, optimizer, device


def one_step(config, batch_cpu, parallel):
    model, optimizer, device = build_runtime(config, parallel)
    state_keys = tuple(model.state_dict().keys())
    optimizer.zero_grad()
    batch = {key: value.to(device) for key, value in batch_cpu.items()}
    scaler = amp.GradScaler(init_scale=4096.0)
    with amp.autocast(enabled=True):
        result = model(batch, None, current_epoch=0)
        losses = [(key, value) for key, value in result.items() if "loss" in key]
        total = sum(value for _, value in losses)
    scaler.scale(total).backward()
    scaler.unscale_(optimizer)
    loss_values = {key: float(value.detach().cpu()) for key, value in losses}
    loss_values["total_loss"] = float(total.detach().cpu())
    grad_hash, grad_count = tensor_digest(
        (name, parameter.grad)
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    )
    visual_gradients = [
        name for name, parameter in model.named_parameters()
        if name.startswith("base_model.visual") and parameter.grad is not None
    ]
    scaler.step(optimizer)
    scaler.update()
    parameter_hash, parameter_count = tensor_digest(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )
    replica_keys = [key for key in state_keys if "fixed_visual_parallel" in key]
    del batch, result, total, optimizer, model
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "losses": loss_values,
        "gradient_sha256": grad_hash,
        "gradient_tensors": grad_count,
        "trainable_parameter_sha256": parameter_hash,
        "trainable_parameter_tensors": parameter_count,
        "state_dict_keys": list(state_keys),
        "replica_state_keys": replica_keys,
        "visual_gradients": visual_gradients,
    }


def benchmark_visual(config, images, parallel, steps):
    model, _, device = build_runtime(config, parallel)
    images = images.to(device)
    for _ in range(2):
        with amp.autocast(enabled=True), torch.no_grad():
            model._encode_fixed_visual(images, None)
    torch.cuda.synchronize()
    for index in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(index)
    started = time.monotonic()
    for _ in range(steps):
        with amp.autocast(enabled=True), torch.no_grad():
            model._encode_fixed_visual(images, None)
    for index in range(torch.cuda.device_count()):
        torch.cuda.synchronize(index)
    elapsed = time.monotonic() - started
    peaks = {
        str(index): int(torch.cuda.max_memory_allocated(index))
        for index in range(torch.cuda.device_count())
    }
    del model, images
    gc.collect()
    torch.cuda.empty_cache()
    return elapsed, peaks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    config = load_train_configs(str(args.config.resolve()))
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(config.CUDA_VISIBLE_DEVICES):
        raise RuntimeError("Preflight must start with the config CUDA_VISIBLE_DEVICES already exported")
    if torch.cuda.device_count() != 4:
        raise RuntimeError(f"Expected exactly four visible CUDA devices, got {torch.cuda.device_count()}")
    initialization = verify(args.config)

    seed_torch(config.seed)
    loaders = Loader(config)
    batch_cpu = {key: value.clone() for key, value in next(iter(loaders.get_train_loader())).items()}
    images = torch.cat(
        (batch_cpu["img_rgb_ori"], batch_cpu["img_rgb_aug"], batch_cpu["img_ir"]), dim=0
    )

    # Direct frozen-visual equality with identical chunking.
    model, _, device = build_runtime(config, parallel=True)
    images_device = images.to(device)
    model._fixed_visual_parallel_enabled = False
    with amp.autocast(enabled=True), torch.no_grad():
        sequential_output = model._encode_fixed_visual(images_device, None)
    model._fixed_visual_parallel_enabled = True
    with amp.autocast(enabled=True), torch.no_grad():
        parallel_output = model._encode_fixed_visual(images_device, None)
    for index in range(torch.cuda.device_count()):
        torch.cuda.synchronize(index)
    exact_visual = tensor_tree_equal(sequential_output, parallel_output)
    visual_max_abs = tensor_tree_max_abs(sequential_output, parallel_output)
    del sequential_output, parallel_output, images_device, model
    gc.collect()
    torch.cuda.empty_cache()

    single_step = one_step(config, batch_cpu, parallel=False)
    parallel_step = one_step(config, batch_cpu, parallel=True)
    step_equal = (
        single_step["losses"] == parallel_step["losses"]
        and single_step["gradient_sha256"] == parallel_step["gradient_sha256"]
        and single_step["trainable_parameter_sha256"]
        == parallel_step["trainable_parameter_sha256"]
        and single_step["state_dict_keys"] == parallel_step["state_dict_keys"]
        and not parallel_step["replica_state_keys"]
        and not parallel_step["visual_gradients"]
    )

    single_seconds, single_peaks = benchmark_visual(config, images, False, args.steps)
    parallel_seconds, parallel_peaks = benchmark_visual(config, images, True, args.steps)
    speedup = single_seconds / parallel_seconds
    memory_limit = 22 * 1024 ** 3
    memory_ok = max(parallel_peaks.values()) < memory_limit
    valid = exact_visual and visual_max_abs == 0.0 and step_equal and speedup >= 2.0 and memory_ok
    implementation = {
        name: {"path": str(path), "sha256": sha256_file(path)}
        for name, path in {
            "runner": REPO_ROOT / "src/salt_vi/utils/super_resolution/run_a3_e4_hpt_l025.py",
            "verifier": REPO_ROOT / "src/salt_vi/utils/super_resolution/verify_a3_stageb_init.py",
            "converter": REPO_ROOT / "src/salt_vi/utils/super_resolution/convert_stage_a.py",
            "parallel_preflight": Path(__file__).resolve(),
            "core_build": REPO_ROOT / "src/salt_vi/engine/build.py",
            "main": REPO_ROOT / "scripts" / "train.py",
        }.items()
    }
    report = {
        "schema_version": 2,
        "formal_training_started": False,
        "valid": valid,
        "parallel_strategy": "frozen_visual_chunk_data_parallel",
        "fixed_visual_device_ids": [0, 1, 2, 3],
        "initialization": initialization,
        "forward_equivalence": {
            "exact_equal": exact_visual,
            "max_abs": visual_max_abs,
            "input_shape": list(images.shape),
        },
        "optimizer_step_equivalence": {
            "exact_equal": step_equal,
            "single": single_step,
            "parallel": parallel_step,
        },
        "gpu_smoke": {
            "both_forward_backward_optimizer_step": step_equal,
            "steps": args.steps,
            "single_seconds": single_seconds,
            "parallel_seconds": parallel_seconds,
            "speedup": speedup,
            "single_peak_bytes": single_peaks,
            "parallel_peak_bytes": parallel_peaks,
            "memory_limit_bytes": memory_limit,
            "memory_ok": memory_ok,
        },
        "implementation": implementation,
        "a3_checkpoint_sha256": sha256_file(config.training_weight_init),
        "data_manifest_sha256": sha256_file(Path(config.sysu_sr_data_root) / "manifest.json"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps({
        "valid": valid,
        "visual_exact": exact_visual,
        "optimizer_step_exact": step_equal,
        "speedup": speedup,
        "parallel_peak_bytes": parallel_peaks,
        "report": str(args.output),
    }, indent=2))
    if not valid:
        raise SystemExit("Four-GPU lossless preflight failed")


if __name__ == "__main__":
    main()
