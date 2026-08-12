import torch

from salt_vi.models.vision_adapter import PMTViTVisual


def test_pmt_gradient_checkpointing_backward():
    model = PMTViTVisual(
        input_resolution=(32, 16),
        patch_size=(8, 8),
        stride_size=(8, 8),
        embed_dim=16,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
        output_dim=16,
        gradient_checkpointing=True,
    ).train()
    loss = model(torch.randn(2, 3, 32, 16))["features"].square().mean()
    loss.backward()
    assert model.vit.blocks[0].attn.qkv.weight.grad is not None
    assert torch.isfinite(model.vit.blocks[0].attn.qkv.weight.grad).all()
