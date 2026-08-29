import torch

from salt_vi.config.validation import validate_runtime_config
from salt_vi.models.vision_adapter import PMTViTVisual
from salt_vi.utils.utils import load_train_configs


def test_broad_soft_ellipse_attention_backward_with_checkpointing():
    visual = PMTViTVisual(
        input_resolution=(64, 32),
        patch_size=(16, 16),
        stride_size=(16, 16),
        embed_dim=32,
        depth=4,
        num_heads=4,
        output_dim=32,
        gradient_checkpointing=True,
        gradient_checkpoint_blocks=4,
        gradient_checkpoint_segments=2,
        attention_backend="manual",
    )
    visual.configure_ellipse_attention(
        layer=2, radius_x=0.58, radius_y=0.55, temperature=0.12
    )
    visual.train()
    output = visual(torch.randn(2, 3, 64, 32, requires_grad=True))
    attention = output["ellipse_attention"]
    mask = output["ellipse_mask"]
    assert attention.shape == (2, 8)
    assert mask.shape == (8,)
    assert torch.allclose(attention.sum(dim=-1), torch.ones(2), atol=1e-6)
    mask_grid = mask.reshape(4, 2)
    assert mask_grid[1:3].mean() > mask_grid[[0, 3]].mean()
    outside_mass = (attention * (1.0 - mask)).sum(dim=-1).mean()
    outside_mass.backward()
    assert visual.vit.blocks[1].attn.qkv.weight.grad is not None
    assert torch.isfinite(visual.vit.blocks[1].attn.qkv.weight.grad).all()


def test_c3_ellipse_configs_only_change_requested_layer(monkeypatch, tmp_path):
    monkeypatch.setenv("SALT_SAFE_TRICKS_OUTPUT_ROOT", str(tmp_path / "unused"))
    monkeypatch.setenv("SALT_ELLIPSE_OUTPUT_ROOT", str(tmp_path / "ellipse"))
    baseline = load_train_configs(
        "configs/stage_a/safe_tricks/c3_camera_diverse_cosine.yaml"
    )
    layer2 = load_train_configs(
        "configs/stage_a/safe_tricks/e1_c3_ellipse_layer2.yaml"
    )
    layer4 = load_train_configs(
        "configs/stage_a/safe_tricks/e2_c3_ellipse_layer4.yaml"
    )
    for config, layer in ((layer2, 2), (layer4, 4)):
        validate_runtime_config(config)
        assert config.ellipse_attention_layer == layer
        assert config.ellipse_attention_weight == 0.10
        assert config.ellipse_attention_radius_x > 0.5
        assert config.ellipse_attention_radius_y > 0.5
        for key in (
            "pretrain_choice",
            "img_size",
            "batch_size",
            "total_train_epoch",
            "sampler_type",
            "normalized_classifier",
            "sysu_sr_data_root",
            "pmt_attention_backend",
            "pmt_gradient_checkpointing",
        ):
            assert getattr(config, key) == getattr(baseline, key)


def test_c3_x4_resize_control_and_ellipse_configs(monkeypatch, tmp_path):
    monkeypatch.setenv("SALT_ELLIPSE_X4_OUTPUT_ROOT", str(tmp_path / "x4-runs"))
    control = load_train_configs(
        "configs/stage_a/safe_tricks/x4/c3_swinir_x4_resize.yaml"
    )
    layer2 = load_train_configs(
        "configs/stage_a/safe_tricks/x4/e1_c3_swinir_x4_resize_ellipse_layer2.yaml"
    )
    layer4 = load_train_configs(
        "configs/stage_a/safe_tricks/x4/e2_c3_swinir_x4_resize_ellipse_layer4.yaml"
    )
    for config in (control, layer2, layer4):
        validate_runtime_config(config)
        assert config.sysu_sr_backend == "image_tree"
        assert config.sysu_sr_data_root.endswith("/outputs/sysu")
        assert set(config.sysu_sr_modalities) == {"rgb", "ir"}
        assert config.sysu_sr_exact_size is False
        assert config.img_size == [512, 256]
    assert control.ellipse_attention_weight == 0.0
    assert layer2.ellipse_attention_layer == 2
    assert layer4.ellipse_attention_layer == 4
