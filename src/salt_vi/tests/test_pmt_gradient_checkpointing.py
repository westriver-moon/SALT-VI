import torch

import salt_vi.models.vision_transformer as vision_transformer
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


def test_pmt_partial_gradient_checkpointing_uses_configured_block_count(monkeypatch):
    checkpoint_calls = []

    def direct_checkpoint(function, *args):
        checkpoint_calls.append(function)
        return function(*args)

    monkeypatch.setattr(
        vision_transformer, "checkpoint_forward", direct_checkpoint
    )
    model = PMTViTVisual(
        input_resolution=(32, 16),
        patch_size=(8, 8),
        stride_size=(8, 8),
        embed_dim=16,
        depth=4,
        num_heads=4,
        mlp_ratio=2.0,
        output_dim=16,
        gradient_checkpointing=True,
        gradient_checkpoint_blocks=2,
    ).train()
    loss = model(torch.randn(2, 3, 32, 16))["features"].square().mean()
    loss.backward()
    assert len(checkpoint_calls) == 2
    assert model.gradient_checkpoint_blocks == 2


def test_pmt_checkpoint_segments_group_contiguous_blocks(monkeypatch):
    checkpoint_calls = []

    def direct_checkpoint(function, *args):
        checkpoint_calls.append(function)
        return function(*args)

    monkeypatch.setattr(
        vision_transformer, "checkpoint_forward", direct_checkpoint
    )
    model = PMTViTVisual(
        input_resolution=(32, 16),
        patch_size=(8, 8),
        stride_size=(8, 8),
        embed_dim=16,
        depth=4,
        num_heads=4,
        mlp_ratio=2.0,
        output_dim=16,
        gradient_checkpointing=True,
        gradient_checkpoint_blocks=3,
        gradient_checkpoint_segments=1,
    ).train()
    model(torch.randn(2, 3, 32, 16))["features"].sum().backward()
    assert len(checkpoint_calls) == 1


def test_segmented_checkpointing_matches_per_block_output_and_gradients():
    kwargs = dict(
        input_resolution=(32, 16),
        patch_size=(8, 8),
        stride_size=(8, 8),
        embed_dim=16,
        depth=4,
        num_heads=4,
        mlp_ratio=2.0,
        drop_rate=0.1,
        drop_path_rate=0.2,
        output_dim=16,
        gradient_checkpointing=True,
        gradient_checkpoint_blocks=3,
    )
    per_block = PMTViTVisual(**kwargs, gradient_checkpoint_segments=3).train()
    segmented = PMTViTVisual(**kwargs, gradient_checkpoint_segments=1).train()
    segmented.load_state_dict(per_block.state_dict(), strict=True)
    source = torch.randn(2, 3, 32, 16)

    def run(model):
        model.zero_grad(set_to_none=True)
        inputs = source.clone().requires_grad_(True)
        torch.manual_seed(123)
        features = model(inputs)["features"]
        features.square().sum().backward()
        return (
            features.detach(),
            inputs.grad.detach(),
            model.vit.blocks[0].attn.qkv.weight.grad.detach(),
        )

    expected = run(per_block)
    actual = run(segmented)
    for expected_tensor, actual_tensor in zip(expected, actual):
        torch.testing.assert_close(
            actual_tensor, expected_tensor, rtol=0.0, atol=0.0
        )
