import argparse
import json

import torch

from salt_vi.config.validation import validate_runtime_config
from salt_vi.engine import build_model
from salt_vi.optim import build_optimizer
from salt_vi.utils.utils import load_train_configs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/stage_a/plugins/quadruple_patch_pasd_512x256_b32.yaml",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    config = load_train_configs(args.config)
    config.gpu_id = "0"
    config.clip_download_root = "/home/lab929/.cache/clip"
    validate_runtime_config(config)
    model = build_model(config).to(torch.device("cuda:0"))
    model.set_train()
    optimizer = build_optimizer(config, model)
    torch.cuda.reset_peak_memory_stats()

    batch_size = args.batch_size
    num_pos = int(config.num_pos)
    if batch_size % num_pos:
        raise ValueError("batch size must be divisible by num_pos")
    labels = torch.arange(batch_size // num_pos).repeat_interleave(num_pos).cuda()
    batch = {
        "img_rgb_ori": torch.randn(batch_size, 3, config.img_h, config.img_w, device="cuda"),
        "img_rgb_aug": torch.randn(batch_size, 3, config.img_h, config.img_w, device="cuda"),
        "img_ir": torch.randn(batch_size, 3, config.img_h, config.img_w, device="cuda"),
        "img_ir_aug": torch.randn(batch_size, 3, config.img_h, config.img_w, device="cuda"),
        "target_rgb": labels,
        "target_ir": labels.clone(),
    }
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda"):
        result = model(batch, mode=None, current_epoch=0)
        losses = [value for key, value in result.items() if "loss" in key]
        total_loss = sum(losses)
    total_loss.backward()

    patch_gradients = [
        branch.proj.weight.grad is not None
        for branch in model.base_model.visual.input_plugin.patch_embeds
    ]
    shared_gradient = model.base_model.visual.vit.blocks[0].attn.qkv.weight.grad is not None
    if not all(patch_gradients) or not shared_gradient:
        raise RuntimeError(
            f"Missing gradients: patch={patch_gradients}, shared={shared_gradient}"
        )
    print(
        json.dumps(
            {
                "loss_keys": sorted(key for key in result if "loss" in key),
                "patch_gradients": patch_gradients,
                "pmt_stage": result["pmt_stage"],
                "shared_gradient": shared_gradient,
                "peak_memory_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 2),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
