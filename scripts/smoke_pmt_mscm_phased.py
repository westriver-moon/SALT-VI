import argparse
import json

import torch

from salt_vi.config.validation import validate_runtime_config
from salt_vi.data.loader import Loader
from salt_vi.engine import build_model
from salt_vi.optim import build_optimizer
from salt_vi.utils.utils import load_train_configs


def run_phase(model, optimizer, loaders, epoch):
    transition = model.prepare_pmt_mscm_phase(epoch, optimizer)
    loaders.set_training_epoch(epoch)
    batch = next(iter(loaders.get_train_loader()))
    batch = {key: value.to(model.device) for key, value in batch.items()}
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda"):
        result = model(batch, mode=None, current_epoch=epoch)
        losses = [value for key, value in result.items() if "loss" in key]
        total_loss = sum(losses)
    if not torch.isfinite(total_loss):
        raise FloatingPointError(f"epoch {epoch} produced non-finite loss")
    total_loss.backward()
    visual = model.base_model.visual
    template_gradient = visual.vit.patch_embed.proj.weight.grad is not None
    branch_gradients = [
        branch.proj.weight.grad is not None
        for branch in visual.input_plugin.patch_embeds
    ]
    shared_gradient = visual.vit.blocks[0].attn.qkv.weight.grad is not None
    optimizer.step()
    return {
        "epoch": epoch,
        "batch_keys": sorted(batch),
        "loss_keys": sorted(key for key in result if "loss" in key),
        "pmt_stage": result["pmt_stage"],
        "template_gradient": template_gradient,
        "branch_gradients": branch_gradients,
        "shared_gradient": shared_gradient,
        "total_loss": round(float(total_loss.detach()), 6),
        "transition": transition,
        "transition_alpha": float(result.get("pmt_mscm_transition_alpha", 0.0)),
        "qct_effective_weight": float(result.get("pmt_mscm_qct_effective_weight", 0.0)),
        "intra_tri": float(result.get("pmt_mscm_intra_tri", 0.0)),
        "cross_tri": float(result.get("pmt_mscm_cross_tri", 0.0)),
        "qct_modality_compactness": float(result.get("qct_modality_compactness", 0.0)),
        "qct_branch_compactness": float(result.get("qct_branch_compactness", 0.0)),
        "qct_negative_margin": float(result.get("qct_negative_margin", 0.0)),
        "qct_hard_negative_distance": float(result.get("qct_hard_negative_distance", 0.0)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/stage_a/plugins/pmt_mscm_phased_pasd_512x256_b32.yaml",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    config = load_train_configs(args.config)
    config.gpu_id = "0"
    config.CUDA_VISIBLE_DEVICES = "0"
    config.batch_size = args.batch_size
    config.num_workers = 0
    config.clip_download_root = "/home/lab929/.cache/clip"
    validate_runtime_config(config)
    loaders = Loader(config)
    model = build_model(config).to(torch.device("cuda:0"))
    model.set_train()
    optimizer = build_optimizer(config, model)
    torch.cuda.reset_peak_memory_stats()

    warmup = run_phase(model, optimizer, loaders, epoch=5)
    switch = run_phase(model, optimizer, loaders, epoch=6)
    phased = run_phase(model, optimizer, loaders, epoch=10)
    if not warmup["template_gradient"] or any(warmup["branch_gradients"]):
        raise RuntimeError(f"Unexpected warmup gradients: {warmup}")
    if switch["template_gradient"] or not all(switch["branch_gradients"]):
        raise RuntimeError(f"Unexpected switch gradients: {switch}")
    if phased["template_gradient"] or not all(phased["branch_gradients"]):
        raise RuntimeError(f"Unexpected phased gradients: {phased}")
    if not all(item["shared_gradient"] for item in (warmup, switch, phased)):
        raise RuntimeError("Shared ViT trunk did not receive gradients in every phase")
    print(json.dumps({
        "warmup": warmup,
        "switch": switch,
        "phased": phased,
        "peak_memory_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 2),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
