import argparse
import json

import torch

from salt_vi.models.vision_adapter import PMTViTVisual


BRANCH_ORDER = (
    "visible_global",
    "visible_channel",
    "infrared_global",
    "infrared_channel",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=256)
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    model = PMTViTVisual(
        input_resolution=(args.height, args.width),
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
        gradient_checkpointing=True,
        attention_backend="flash",
        visual_input_backend="quadruple_patch",
        quadruple_branch_order=BRANCH_ORDER,
    ).to(device).train()
    views = torch.randn(
        args.batch_size,
        4,
        3,
        args.height,
        args.width,
        device=device,
    )
    with torch.amp.autocast("cuda"):
        output = model(views)
        loss = output["branch_features"].float().square().mean()
    loss.backward()

    patch_gradients = [
        branch.proj.weight.grad is not None
        for branch in model.input_plugin.patch_embeds
    ]
    shared_gradient = model.vit.blocks[0].attn.qkv.weight.grad is not None
    if not all(patch_gradients) or not shared_gradient:
        raise RuntimeError(
            f"Missing gradients: patch={patch_gradients}, shared={shared_gradient}"
        )
    print(
        json.dumps(
            {
                "branch_features": list(output["branch_features"].shape),
                "branch_tokens": list(output["branch_tokens"].shape),
                "patch_gradients": patch_gradients,
                "shared_gradient": shared_gradient,
                "peak_memory_mib": round(
                    torch.cuda.max_memory_allocated(device) / 1024**2, 2
                ),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
