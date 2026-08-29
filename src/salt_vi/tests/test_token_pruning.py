from types import SimpleNamespace

import pytest
import torch

from salt_vi.config.validation import validate_runtime_config
from salt_vi.models.vision_adapter import (
    PMTViTVisual,
    build_rounded_rect_keep_indices,
)
from salt_vi.models.vision_transformer import ViT
from salt_vi.utils.utils import load_train_configs


def _small_visual(**overrides):
    kwargs = dict(
        input_resolution=(32, 16),
        patch_size=(8, 8),
        stride_size=(8, 8),
        embed_dim=16,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        output_dim=16,
    )
    kwargs.update(overrides)
    return PMTViTVisual(**kwargs)


@pytest.mark.parametrize(
    ("fraction", "expected_keep"),
    ((0.10, 794), (0.15, 752), (0.20, 708)),
)
def test_c3_grid_roundrect_sweep_counts_and_symmetry(fraction, expected_keep):
    indices = build_rounded_rect_keep_indices(
        (42, 21), prune_fraction=fraction, roundness=4.0
    )
    assert indices.shape == (expected_keep,)
    mask = torch.zeros(42 * 21, dtype=torch.bool)
    mask[indices] = True
    mask = mask.reshape(42, 21)
    assert torch.equal(mask, mask.flip(0))
    assert torch.equal(mask, mask.flip(1))


def test_position_embeddings_are_gathered_from_original_grid():
    vit = ViT(
        img_size=(16, 16),
        patch_size=(8, 8),
        stride_size=(8, 8),
        embed_dim=4,
        depth=1,
        num_heads=1,
        drop_rate=0.0,
        drop_path_rate=0.0,
    )
    with torch.no_grad():
        vit.cls_token.zero_()
        values = torch.arange(5, dtype=torch.float32).reshape(1, 5, 1)
        vit.pos_embed.copy_(values.expand(-1, -1, 4))
    patches = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4)
    keep = torch.tensor([0, 3], dtype=torch.long)

    tokens, grid = vit.prepare_embedded_tokens(
        patches, (2, 2), patch_indices=keep
    )

    assert grid == (2, 2)
    torch.testing.assert_close(tokens[:, :1], torch.zeros(1, 1, 4))
    torch.testing.assert_close(tokens[:, 1], patches[:, 0] + 1.0)
    torch.testing.assert_close(tokens[:, 2], patches[:, 3] + 4.0)


def test_roundrect_forward_backward_and_checkpoint_keys_are_compatible():
    baseline = PMTViTVisual(
        input_resolution=(512, 256),
        patch_size=(16, 16),
        stride_size=(12, 12),
        embed_dim=16,
        depth=1,
        num_heads=4,
        mlp_ratio=2.0,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        output_dim=16,
    )
    pruned = PMTViTVisual(
        input_resolution=(512, 256),
        patch_size=(16, 16),
        stride_size=(12, 12),
        embed_dim=16,
        depth=1,
        num_heads=4,
        mlp_ratio=2.0,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        output_dim=16,
        token_pruning_mode="rounded_rect",
        token_prune_fraction=0.10,
        token_roundness=4.0,
    )
    assert set(baseline.state_dict()) == set(pruned.state_dict())
    pruned.load_state_dict(baseline.state_dict(), strict=True)

    images = torch.randn(1, 3, 512, 256, requires_grad=True)
    output = pruned(images)
    assert output["tokens"].shape == (1, 795, 16)
    output["features"].square().mean().backward()
    assert images.grad is not None
    assert pruned.vit.blocks[0].attn.qkv.weight.grad is not None


def test_quadruple_branches_share_the_same_physical_mask():
    model = _small_visual(
        visual_input_backend="quadruple_patch",
        quadruple_branch_order=(
            "visible_global",
            "visible_channel",
            "infrared_global",
            "infrared_channel",
        ),
        token_pruning_mode="rounded_rect",
        token_prune_fraction=0.50,
        token_roundness=4.0,
    )
    output = model(torch.randn(2, 4, 3, 32, 16))
    assert output["tokens"].shape == (8, 5, 16)
    assert output["branch_tokens"].shape == (2, 4, 5, 16)


def test_token_pruning_rejects_soft_ellipse_attention():
    config = SimpleNamespace(
        pmt_token_pruning_mode="rounded_rect",
        pmt_token_prune_fraction=0.10,
        pmt_token_roundness=4.0,
        pretrain_choice="PMT_VIT",
        ellipse_attention_weight=0.10,
        ellipse_attention_layer=2,
        pmt_depth=12,
    )
    with pytest.raises(ValueError, match="cannot be enabled together"):
        validate_runtime_config(config)

    model = _small_visual(
        token_pruning_mode="rounded_rect",
        token_prune_fraction=0.10,
    )
    with pytest.raises(ValueError, match="cannot be enabled together"):
        model.configure_ellipse_attention(layer=1)


@pytest.mark.parametrize(
    ("suffix", "fraction"),
    (("10", 0.10), ("15", 0.15), ("20", 0.20)),
)
def test_c3_swin_person_fit_roundrect_configs_change_only_token_pruning(
    suffix, fraction
):
    baseline = load_train_configs(
        "configs/stage_a/person_assets/sysu_person_fit.yaml"
    )
    rounded = load_train_configs(
        f"configs/stage_a/person_assets/sysu_person_fit_rounded_rect_{suffix}.yaml"
    )
    validate_runtime_config(rounded)
    assert rounded.pmt_token_pruning_mode == "rounded_rect"
    assert rounded.pmt_token_prune_fraction == fraction
    assert rounded.pmt_token_roundness == 4.0
    for key in (
        "prepared_data_root",
        "pretrain_choice",
        "img_size",
        "pmt_patch_size",
        "pmt_stride_size",
        "sampler_type",
        "total_train_epoch",
        "lrscheduler",
    ):
        assert getattr(rounded, key) == getattr(baseline, key)


def test_c3_swin_resize_roundrect_config_uses_canonical_resize_asset():
    baseline = load_train_configs("configs/stage_a/person_assets/sysu_resize.yaml")
    rounded = load_train_configs(
        "configs/stage_a/person_assets/sysu_resize_rounded_rect_10.yaml"
    )
    validate_runtime_config(rounded)
    assert rounded.prepared_data_root.endswith("/person-assets-512x256/resize")
    assert rounded.prepared_data_root == baseline.prepared_data_root
    assert rounded.pmt_token_pruning_mode == "rounded_rect"
    assert rounded.pmt_token_prune_fraction == 0.10
