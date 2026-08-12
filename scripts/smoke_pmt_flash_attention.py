#!/usr/bin/env python3
"""Run a full-topology PMT forward/backward CUDA attention smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from salt_vi.attention import normalize_attention_backend
from salt_vi.models.vision_adapter import PMTViTVisual


PATCH_EMBED = {
    "anchor_branch": 0,
    "branches": [
        {"patch_size": [16, 16], "stride_size": [12, 12]},
        {"patch_size": [16, 8], "stride_size": [12, 6]},
    ],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("legacy", "sdpa", "flash"), required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the PMT attention smoke")
    backend = normalize_attention_backend(args.backend)
    device = torch.device("cuda:0")
    torch.manual_seed(20260813)
    torch.cuda.manual_seed_all(20260813)

    model = PMTViTVisual(
        input_resolution=(512, 256),
        patch_size=(16, 16),
        stride_size=(12, 12),
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        drop_rate=0.03,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        output_dim=768,
        patch_embed_config=PATCH_EMBED,
        gradient_checkpointing=args.gradient_checkpointing,
        attention_backend=backend,
    ).to(device).train()
    selected = {block.attn.attention_backend for block in model.vit.blocks}
    if selected != {backend}:
        raise RuntimeError(f"PMT blocks selected unexpected attention backends: {selected}")

    inputs = torch.randn(args.batch_size, 3, 512, 256, device=device)

    def step():
        model.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            output = model(inputs)["features"]
            loss = output.float().square().mean()
        loss.backward()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss: {loss}")
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
            raise FloatingPointError("Missing or non-finite PMT gradients")
        return float(loss.detach())

    for _ in range(args.warmup_steps):
        step()
    torch.cuda.synchronize(device)
    baseline_allocated = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)

    durations = []
    losses = []
    for _ in range(args.steps):
        started = time.perf_counter()
        losses.append(step())
        torch.cuda.synchronize(device)
        durations.append(time.perf_counter() - started)

    result = {
        "valid": True,
        "backend": backend,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": torch.cuda.get_device_name(device),
        "capability": list(torch.cuda.get_device_capability(device)),
        "batch_size": args.batch_size,
        "image_size": [512, 256],
        "tokens": int(model.vit.patch_embed.num_patches + 1),
        "embed_dim": 768,
        "depth": 12,
        "heads": 12,
        "head_dim": 64,
        "gradient_checkpointing": args.gradient_checkpointing,
        "steps": args.steps,
        "step_seconds": durations,
        "median_step_seconds": statistics.median(durations),
        "losses": losses,
        "baseline_allocated_bytes": baseline_allocated,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
