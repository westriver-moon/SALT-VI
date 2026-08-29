from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torchvision.transforms as transforms

from salt_vi.config.validation import validate_runtime_config
from salt_vi.data.pose_tokens import (
    HumanTokenMaskStore,
    SynchronizedTokenTransform,
)
from salt_vi.engine.build import Classifier
from salt_vi.models.vision_adapter import (
    PMTViTVisual,
    sample_human_token_deletions,
)
from salt_vi.training.recipes import (
    PMTRecipe,
    cti_loss_weight,
    true_identity_evidence,
)
from salt_vi.utils.utils import load_train_configs


def _small_visual():
    return PMTViTVisual(
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


def test_fixed_k_sampling_is_unique_human_only_and_ordered():
    torch.manual_seed(4)
    masks = torch.tensor(
        [
            [1.0, 0.9, 0.0, 0.8, 0.0, 0.7, 0.6, 0.0],
            [0.0, 0.9, 0.8, 0.0, 0.7, 0.6, 0.0, 1.0],
        ]
    )
    kept, deleted, valid = sample_human_token_deletions(masks, 3, 0.05)

    assert valid.tolist() == [True, True]
    assert deleted.shape == (2, 3)
    assert kept.shape == (2, 5)
    assert torch.all(deleted[:, 1:] > deleted[:, :-1])
    assert torch.all(kept[:, 1:] > kept[:, :-1])
    assert torch.all(masks.gather(1, deleted) >= 0.05)
    for row in range(2):
        assert set(kept[row].tolist()).isdisjoint(deleted[row].tolist())


def test_too_few_human_tokens_marks_only_cti_invalid():
    mask = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    kept, deleted, valid = sample_human_token_deletions(mask, 2, 0.05)
    assert kept.shape == (1, 2)
    assert deleted.shape == (1, 2)
    assert valid.tolist() == [False]


def test_counterfactual_forward_physically_shortens_and_backpropagates():
    torch.manual_seed(7)
    model = _small_visual()
    images = torch.randn(2, 3, 32, 16, requires_grad=True)
    human_mask = torch.ones(2, 4, 2)

    output = model.forward_counterfactual(
        images,
        human_mask,
        intervention_layer=1,
        delete_count=2,
        threshold=0.05,
    )

    assert output["tokens"].shape == (2, 9, 16)
    assert output["cti_deleted_tokens"].shape == (2, 7, 16)
    assert output["cti_valid"].all()
    loss = output["features"].square().mean()
    loss = loss + output["cti_deleted_features"].square().mean()
    loss.backward()
    assert torch.isfinite(images.grad).all()
    assert model.vit.blocks[1].attn.qkv.weight.grad is not None


def test_true_identity_evidence_and_weight_ramp():
    scores = torch.tensor([[2.0, 1.0, 0.0], [0.0, 1.0, 3.0]])
    labels = torch.tensor([0, 2])
    expected = torch.tensor(
        [
            2.0 - torch.logsumexp(torch.tensor([1.0, 0.0]), dim=0),
            3.0 - torch.logsumexp(torch.tensor([0.0, 1.0]), dim=0),
        ]
    )
    torch.testing.assert_close(true_identity_evidence(scores, labels), expected)

    args = SimpleNamespace(
        cti_enabled=True,
        cti_weight=0.3,
        cti_start_epoch=6,
        cti_warmup_epochs=3,
    )
    assert cti_loss_weight(args, 5) == 0.0
    assert cti_loss_weight(args, 6) == pytest.approx(0.1)
    assert cti_loss_weight(args, 7) == pytest.approx(0.2)
    assert cti_loss_weight(args, 8) == pytest.approx(0.3)


def test_counterfactual_scoring_reuses_batch_stats_without_bn_mutation():
    torch.manual_seed(2)
    classifier = Classifier(pid_num=5, dim=4, joint_mode="image_only")
    classifier.train()
    features = torch.randn(8, 4)
    _, full_scores = classifier(features)
    running_mean = classifier.BN.running_mean.clone()
    running_var = classifier.BN.running_var.clone()

    counterfactual_scores = classifier.counterfactual_scores(features, features)

    torch.testing.assert_close(counterfactual_scores, full_scores, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(classifier.BN.running_mean, running_mean)
    torch.testing.assert_close(classifier.BN.running_var, running_var)


def test_horizontal_flip_is_synchronized_with_token_mask():
    image = np.zeros((2, 4, 3), dtype=np.uint8)
    image[:, 0] = 255
    mask = torch.tensor([[1.0, 0.5, 0.0, 0.0]])
    paired = SynchronizedTokenTransform(
        transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ToTensor(),
            ]
        )
    )

    flipped_image, flipped_mask = paired(image, mask)

    assert flipped_image[:, :, -1].mean() == pytest.approx(1.0)
    torch.testing.assert_close(flipped_mask, mask.flip(-1))


def test_human_mask_store_unions_only_valid_parts(tmp_path):
    anatomy = tmp_path / "sysu" / "anatomy" / "cam1" / "0001"
    anatomy.mkdir(parents=True)
    token_masks = np.zeros((3, 2, 2), dtype=np.float16)
    token_masks[0, 0, 0] = 0.7
    token_masks[1, 0, 1] = 0.9
    token_masks[2, 1, 1] = 0.8
    np.savez(
        anatomy / "0001.npz",
        token_masks=token_masks,
        part_valid=np.array([True, False, True]),
    )
    store = HumanTokenMaskStore(tmp_path, "sysu", expected_grid=(2, 2))

    assert store.require_keys(["cam1/0001/0001.jpg"]) == 1
    with pytest.raises(FileNotFoundError, match="coverage is incomplete"):
        store.require_keys(["cam1/0001/missing.jpg"])
    mask = store.load("cam1/0001/0001.jpg")

    torch.testing.assert_close(
        mask,
        torch.tensor([[0.7002, 0.0], [0.0, 0.7998]]),
        atol=1e-3,
        rtol=0,
    )


def test_cti_config_keeps_full_grid_and_rejects_pruning():
    config = load_train_configs(
        "configs/stage_a/person_assets/sysu_person_fit_cti.yaml"
    )
    validate_runtime_config(config)
    assert config.pmt_token_pruning_mode == "none"
    assert config.ellipse_attention_weight == 0.0
    assert config.cti_enabled is True

    config.pmt_token_pruning_mode = "rounded_rect"
    config.pmt_token_prune_fraction = 0.1
    with pytest.raises(ValueError, match="cannot be enabled together"):
        validate_runtime_config(config)


def test_pmt_recipe_adds_cti_to_original_c3_and_backpropagates():
    class TinyBase(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.visual = _small_visual()

        @property
        def dtype(self):
            return self.visual.input_dtype

        def encode_image(self, images, mode=None):
            return self.visual(images, mode)

    args = SimpleNamespace(
        pretrain_choice="PMT_VIT",
        training_mode="RGB_IR",
        visual_input_backend="single",
        rfa_probability=0.0,
        rfa_gaussian_sigma=0.1,
        pmt_progressive_epoch=0,
        Fix_Visual=False,
        id_loss_weight=1.0,
        pmt_metric_loss="legacy",
        triplet_mining="pmt_hard",
        pmt_msel_weight=0.5,
        pmt_dcl_weight=0.5,
        cti_enabled=True,
        cti_weight=0.4,
        cti_start_epoch=0,
        cti_warmup_epochs=0,
        cti_intervention_layer=1,
        cti_delete_count=2,
        cti_human_threshold=0.05,
        cti_margin=10.0,
        ellipse_attention_weight=0.0,
    )
    model = SimpleNamespace(
        args=args,
        base_model=TinyBase(),
        classifier=Classifier(
            pid_num=4,
            dim=16,
            joint_mode="image_only",
            normalized=True,
        ),
        pid_criterion=torch.nn.CrossEntropyLoss(),
        logit_scale=torch.tensor(1.0).log(),
        _visual_unfrozen=True,
    )
    model._assert_pmt_batch_layout = lambda *_args: None
    model._get_visual_embedding = lambda output: output["features"]

    def slice_visual(output, start, end):
        batch_size = output["features"].shape[0]
        return {
            key: (
                value[start:end]
                if torch.is_tensor(value)
                and value.ndim > 0
                and value.shape[0] == batch_size
                else value
            )
            for key, value in output.items()
        }

    model._slice_visual_output = slice_visual
    model.pmt_tri_criterion = lambda first, second, labels: (
        first.square().mean() + second.square().mean()
    ) * 0.01
    model.pmt_msel_criterion = lambda features, labels: features.square().mean() * 0.0
    model.pmt_dcl_criterion = lambda features, labels: features.square().mean() * 0.0

    batch = {
        "img_rgb_ori": torch.randn(2, 3, 32, 16),
        "img_rgb_aug": torch.randn(2, 3, 32, 16),
        "img_ir": torch.randn(2, 3, 32, 16),
        "cti_mask_rgb_ori": torch.ones(2, 4, 2),
        "cti_mask_rgb_aug": torch.ones(2, 4, 2),
        "cti_mask_ir": torch.ones(2, 4, 2),
        "target_rgb": torch.tensor([0, 1]),
        "target_ir": torch.tensor([0, 1]),
    }

    result = PMTRecipe().compute_losses(model, batch, current_epoch=1)
    total = sum(value for key, value in result.items() if "loss" in key)
    total.backward()

    assert "id_loss" in result
    assert "tri_loss" in result
    assert result["cti_loss"].item() > 0.0
    assert result["cti_valid_fraction"].item() == 1.0
    assert model.base_model.visual.vit.blocks[1].attn.qkv.weight.grad is not None
