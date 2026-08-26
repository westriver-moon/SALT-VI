from __future__ import annotations

import pytest
import torch

from salt_vi.attention import (
    normalize_attention_backend,
    run_scaled_dot_product_attention,
)
from salt_vi.models.vision_adapter import PMTViTVisual
from salt_vi.models.vision_transformer import Attention


def _qkv(dtype=torch.float32):
    torch.manual_seed(7)
    return tuple(
        torch.randn(2, 4, 5, 8, dtype=dtype, requires_grad=True)
        for _ in range(3)
    )


def test_attention_backend_names_are_strict():
    assert normalize_attention_backend(None) == "manual"
    assert normalize_attention_backend("SDPA") == "sdpa"
    with pytest.raises(ValueError, match="Unsupported attention backend"):
        normalize_attention_backend("automatic")


def test_manual_backend_matches_explicit_attention_and_backward():
    query, key, value = _qkv()
    expected_weights = (query @ key.transpose(-2, -1)) * (8**-0.5)
    expected = expected_weights.softmax(dim=-1) @ value
    observed = run_scaled_dot_product_attention(
        query,
        key,
        value,
        scale=8**-0.5,
        dropout_p=0.0,
        training=True,
        backend="manual",
    )
    assert torch.allclose(observed, expected, atol=1e-7, rtol=1e-6)
    observed.square().mean().backward()
    assert all(tensor.grad is not None for tensor in (query, key, value))
    assert all(torch.isfinite(tensor.grad).all() for tensor in (query, key, value))


@pytest.mark.skipif(
    not hasattr(torch.nn.functional, "scaled_dot_product_attention"),
    reason="SDPA requires PyTorch 2.0 or newer",
)
def test_sdpa_backend_matches_manual_on_cpu():
    query, key, value = _qkv()
    manual = run_scaled_dot_product_attention(
        query,
        key,
        value,
        scale=8**-0.5,
        dropout_p=0.0,
        training=False,
        backend="manual",
    )
    sdpa = run_scaled_dot_product_attention(
        query,
        key,
        value,
        scale=8**-0.5,
        dropout_p=0.0,
        training=False,
        backend="sdpa",
    )
    assert torch.allclose(sdpa, manual, atol=1e-6, rtol=1e-5)


@pytest.mark.skipif(
    not hasattr(torch.nn.functional, "scaled_dot_product_attention"),
    reason="Flash SDPA requires PyTorch 2.0 or newer",
)
def test_flash_backend_rejects_cpu_instead_of_falling_back():
    query, key, value = _qkv(dtype=torch.float16)
    with pytest.raises(RuntimeError, match="requires CUDA tensors"):
        run_scaled_dot_product_attention(
            query,
            key,
            value,
            scale=8**-0.5,
            dropout_p=0.0,
            training=False,
            backend="flash",
        )


def test_backend_switch_preserves_checkpoint_parameter_contract():
    manual = Attention(dim=32, num_heads=4, attention_backend="manual")
    sdpa = Attention(dim=32, num_heads=4, attention_backend="sdpa")
    assert manual.state_dict().keys() == sdpa.state_dict().keys()
    sdpa.load_state_dict(manual.state_dict(), strict=True)


def test_pmt_visual_propagates_backend_to_every_block():
    model = PMTViTVisual(
        input_resolution=(32, 16),
        patch_size=(8, 8),
        stride_size=(8, 8),
        embed_dim=32,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
        output_dim=32,
        attention_backend="sdpa",
    )
    assert [block.attn.attention_backend for block in model.vit.blocks] == [
        "sdpa",
        "sdpa",
    ]


def test_pmt_visual_checkpoint_loads_strictly_across_backends():
    model_kwargs = dict(
        input_resolution=(32, 16),
        patch_size=(8, 8),
        stride_size=(8, 8),
        embed_dim=32,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
        output_dim=32,
    )
    manual = PMTViTVisual(**model_kwargs, attention_backend="manual")
    flash = PMTViTVisual(**model_kwargs, attention_backend="flash")
    assert manual.state_dict().keys() == flash.state_dict().keys()
    flash.load_state_dict(manual.state_dict(), strict=True)
