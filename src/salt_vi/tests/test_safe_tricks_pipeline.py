from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from salt_vi.config.validation import validate_runtime_config
from salt_vi.optim.build import build_optimizer, pmt_visual_layer_id
from salt_vi.training.recipes import cross_modal_hard_weight
from salt_vi.utils.utils import load_train_configs


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_safe_trick_configs_are_isolated_and_valid(monkeypatch, tmp_path):
    checkpoint = tmp_path / "stage_a.pth"
    checkpoint.write_bytes(b"stage-a")
    monkeypatch.setenv("SALT_STAGE_A_CHECKPOINT", str(checkpoint))
    monkeypatch.setenv("SALT_STAGE_A_SHA256", "f" * 64)
    monkeypatch.setenv("SALT_SAFE_TRICKS_OUTPUT_ROOT", str(tmp_path / "outputs"))
    configs = {
        name: load_train_configs(
            str(PROJECT_ROOT / f"configs/stage_b/safe_tricks/{name}.yaml")
        )
        for name in (
            "b0_baseline",
            "b1_cls_gem",
            "b2_unfreeze_last2",
            "b3_unfreeze_last2_llrd",
            "b4_qbn_freeze6",
            "b5_hard_loss_ramp",
            "b6_rgb_consistency",
        )
    }
    for config in configs.values():
        validate_runtime_config(config)
        assert config.training_weight_init == str(checkpoint)
    assert configs["b0_baseline"].visual_pooling == "cls"
    assert configs["b1_cls_gem"].visual_pooling == "cls_gem"
    assert configs["b2_unfreeze_last2"].visual_unfreeze_last_n_blocks == 2
    assert configs["b3_unfreeze_last2_llrd"].visual_layer_decay == 0.85
    assert configs["b4_qbn_freeze6"].uni_BN is True
    assert configs["b5_hard_loss_ramp"].cross_modal_hard_start_epoch == 3
    assert configs["b6_rgb_consistency"].rgb_consistency_weight == 0.1


@pytest.mark.parametrize(
    ("epoch", "expected"),
    ((0, 0.0), (3, 0.0), (4, 0.3125), (7, 1.25), (8, 1.25)),
)
def test_cross_modal_hard_weight_ramps_without_test_state(epoch, expected):
    args = SimpleNamespace(
        cross_modal_hard_weight=1.25,
        cross_modal_hard_start_epoch=3,
        cross_modal_hard_ramp_epochs=5,
    )
    assert cross_modal_hard_weight(args, epoch) == pytest.approx(expected)


class _DummyVisual(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = nn.Module()
        self.vit.blocks = nn.ModuleList(nn.Linear(2, 2) for _ in range(12))
        self.vit.norm = nn.LayerNorm(2)
        self.projection = nn.Linear(2, 2, bias=False)


class _DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = nn.Module()
        self.base_model.visual = _DummyVisual()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    @staticmethod
    def is_scheduled_visual_parameter(name):
        return name.startswith((
            "base_model.visual.vit.blocks.10",
            "base_model.visual.vit.blocks.11",
            "base_model.visual.vit.norm",
            "base_model.visual.projection",
        ))


def test_llrd_optimizer_orders_last_blocks_and_skips_norm_decay():
    args = SimpleNamespace(
        lr_factor=5.0,
        lr_visual=3e-4,
        visual_lr=1e-6,
        visual_weight_decay=1e-4,
        visual_weight_decay_bias=1e-4,
        visual_bias_lr_factor=1.0,
        lr_txt=7.5e-6,
        text_weight_decay=4e-5,
        text_weight_decay_bias=0.0,
        text_bias_lr_factor=2.0,
        classifier_lr_factor=1.0,
        pmt_recipe=False,
        pmt_backbone_lr_factor=0.5,
        pmt_depth=12,
        visual_layer_decay=0.85,
        optimizer_no_weight_decay=True,
        optimizer="AdamW",
        alpha=0.9,
        beta=0.999,
        lr=None,
        momentum=0.9,
    )
    optimizer = build_optimizer(args, _DummyModel())
    by_layer = {}
    for group in optimizer.param_groups:
        by_layer.setdefault(group["layer_id"], group)
    assert pmt_visual_layer_id("base_model.visual.vit.blocks.10.attn.weight", 12) == 11
    assert by_layer[11]["lr"] == pytest.approx(1e-6 * 0.85**2)
    assert by_layer[12]["lr"] == pytest.approx(1e-6 * 0.85)
    assert by_layer[13]["lr"] == pytest.approx(1e-6)
    assert by_layer[13]["weight_decay"] == 0.0
