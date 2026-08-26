from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from salt_vi.config.validation import validate_runtime_config
from salt_vi.data.sampler import CameraDiverseIdentitySampler, GenIdx
from salt_vi.engine.build import Classifier
from salt_vi.engine.ema import ModelEMA
from salt_vi.optim.build import build_optimizer, pmt_visual_layer_id
from salt_vi.training.recipes import cross_modal_hard_weight, random_frequency_augmentation
from salt_vi.utils.loss import HeteroCenterTripletLoss
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


def test_stage_a_p1_configs_are_single_variable_and_valid(monkeypatch, tmp_path):
    monkeypatch.setenv("SALT_SAFE_TRICKS_OUTPUT_ROOT", str(tmp_path / "outputs"))
    configs = {
        name: load_train_configs(
            str(PROJECT_ROOT / f"configs/stage_a/safe_tricks/{name}.yaml")
        )
        for name in (
            "a0_resolution_aligned_512",
            "a1_ema",
            "a2_camera_diverse",
            "a3_hetero_center",
            "a4_rfa",
            "a5_cosine_softmax",
            "a6_p1_combined",
            "c1_ema_cosine",
            "c2_ema_camera_diverse",
            "c3_camera_diverse_cosine",
            "c3_b96_camera_diverse_cosine",
        )
    }
    for config in configs.values():
        validate_runtime_config(config)
    assert configs["a0_resolution_aligned_512"].ema_enabled is False
    assert configs["a0_resolution_aligned_512"].output_root == str(
        tmp_path / "outputs" / "a0_resolution_aligned_512"
    )
    assert configs["a1_ema"].ema_enabled is True
    assert configs["a2_camera_diverse"].sampler_type == "identity_camera_diverse"
    assert configs["a3_hetero_center"].pmt_metric_loss == "hetero_center"
    assert configs["a4_rfa"].rfa_probability == pytest.approx(0.5)
    assert configs["a5_cosine_softmax"].normalized_classifier is True
    assert configs["a6_p1_combined"].ema_enabled is True
    assert configs["c1_ema_cosine"].ema_enabled is True
    assert configs["c1_ema_cosine"].normalized_classifier is True
    assert configs["c2_ema_camera_diverse"].ema_enabled is True
    assert configs["c2_ema_camera_diverse"].sampler_type == "identity_camera_diverse"
    assert configs["c3_camera_diverse_cosine"].sampler_type == "identity_camera_diverse"
    assert configs["c3_camera_diverse_cosine"].normalized_classifier is True
    assert configs["c3_camera_diverse_cosine"].batch_size == 32
    assert configs["c3_b96_camera_diverse_cosine"].sampler_type == "identity_camera_diverse"
    assert configs["c3_b96_camera_diverse_cosine"].normalized_classifier is True
    assert configs["c3_b96_camera_diverse_cosine"].batch_size == 48
    assert configs["c3_b96_camera_diverse_cosine"].num_pos == 4


def test_camera_diverse_sampler_covers_available_cameras_per_identity():
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]).numpy()
    positions, _ = GenIdx(labels, labels)
    cameras = torch.tensor([1, 2, 3, 4, 1, 2, 3, 4]).numpy()
    sampler = CameraDiverseIdentitySampler(
        labels, labels, positions, positions, 4, 2, cameras, cameras
    )
    for offset in range(0, len(sampler.index1), 4):
        assert len(set(cameras[sampler.index1[offset : offset + 4]])) == 4


def test_hetero_center_triplet_uses_modality_centers():
    visible = torch.tensor([[0.0], [0.2], [3.0], [3.2]])
    infrared = torch.tensor([[0.1], [0.3], [3.1], [3.3]])
    labels = torch.tensor([0, 0, 1, 1])
    loss = HeteroCenterTripletLoss(margin=0.3)(visible, infrared, labels, labels)
    assert loss == pytest.approx(0.0)


def test_rfa_preserves_shape_and_finite_values():
    torch.manual_seed(4)
    mean = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    visible = (torch.rand(3, 3, 16, 8) - mean) / std
    infrared = (torch.rand(3, 3, 16, 8) - mean) / std
    augmented_visible, augmented_ir = random_frequency_augmentation(
        visible, infrared, 1.0, 0.1
    )
    assert augmented_visible.shape == visible.shape
    assert augmented_ir.shape == infrared.shape
    assert torch.isfinite(augmented_visible).all()
    assert not torch.equal(augmented_visible, visible)


def test_cosine_classifier_normalizes_features_and_weights():
    classifier = Classifier(2, dim=2, joint_mode="image_only", normalized=True, scale=2.0)
    classifier.train()
    classifier.BN.eval()
    with torch.no_grad():
        classifier.BN.weight.fill_(1.0)
        classifier.BN.bias.zero_()
        classifier.classifier.weight.copy_(torch.eye(2))
    _, logits = classifier(torch.tensor([[2.0, 0.0], [0.0, 3.0]]))
    assert logits.diag().tolist() == pytest.approx([2.0, 2.0], abs=1e-4)


def test_ema_updates_and_temporarily_swaps_model_weights():
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    ema = ModelEMA(model, decay=0.5)
    with torch.no_grad():
        model.weight.fill_(3.0)
    ema.update(model)
    with ema.average_parameters(model):
        assert model.weight.item() == pytest.approx(2.0)
    assert model.weight.item() == pytest.approx(3.0)


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
