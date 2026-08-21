import numpy as np
import torch
from types import SimpleNamespace
from types import MethodType
from unittest.mock import patch

from torchvision import transforms

from salt_vi.data.dataset import SYSU_Tri_Data
from salt_vi.data.loader import (
    ExactSize,
    build_mscmnet_exact_quadruple_transforms,
    collate_pmt_mscm_warmup,
)
from salt_vi.data.processing import (
    MSCMChannelAdapGray,
    MSCMChannelExchange,
    MSCMChannelT,
)
from salt_vi.config.validation import validate_runtime_config
from salt_vi.engine.build import CLIP2ReID
from salt_vi.models.vision_adapter import PMTViTVisual
from salt_vi.optim.build import pmt_visual_layer_id
from salt_vi.training.recipes import (
    PMTMSCMPhasedRecipe,
    PMTRecipe,
    build_training_recipe,
)
from salt_vi.utils.loss import PMTQuadrupleCenterTripletLoss
from salt_vi.utils.utils import load_train_configs


BRANCH_ORDER = (
    "visible_global",
    "visible_channel",
    "infrared_global",
    "infrared_channel",
)


def test_warmup_collate_drops_only_unused_rgb_view():
    samples = [
        {
            "img_rgb_ori": torch.full((3, 2, 2), float(index)),
            "img_rgb_aug": torch.full((3, 2, 2), float(index + 10)),
            "img_ir": torch.full((3, 2, 2), float(index + 20)),
            "target_rgb": index,
            "target_ir": index,
        }
        for index in range(2)
    ]
    collated = collate_pmt_mscm_warmup(samples)
    assert "img_rgb_ori" not in collated
    torch.testing.assert_close(
        collated["img_rgb_aug"],
        torch.stack([sample["img_rgb_aug"] for sample in samples]),
    )
    torch.testing.assert_close(
        collated["img_ir"],
        torch.stack([sample["img_ir"] for sample in samples]),
    )


def build_visual(backend="quadruple_patch", template_trainable=False):
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
        visual_input_backend=backend,
        quadruple_branch_order=BRANCH_ORDER,
        quadruple_template_trainable=template_trainable,
    )


def test_quadruple_patch_shapes_parameters_and_shared_gradient():
    model = build_visual()
    views = torch.randn(2, 4, 3, 32, 16)
    output = model(views)

    assert output["branch_order"] == BRANCH_ORDER
    assert output["tokens"].shape == (8, 9, 16)
    assert output["branch_tokens"].shape == (2, 4, 9, 16)
    assert output["branch_features"].shape == (2, 4, 16)

    patch_weights = [branch.proj.weight for branch in model.input_plugin.patch_embeds]
    assert len({weight.data_ptr() for weight in patch_weights}) == 4
    assert all(torch.equal(patch_weights[0], weight) for weight in patch_weights[1:])

    output["branch_features"].square().mean().backward()
    assert all(weight.grad is not None for weight in patch_weights)
    assert model.vit.blocks[0].attn.qkv.weight.grad is not None


def test_modality_forward_keeps_two_branches_but_returns_one_feature():
    model = build_visual().eval()
    images = torch.randn(3, 3, 32, 16)
    with torch.no_grad():
        output = model(images, mode="ir")

    assert output["branch_ids"] == (2, 3)
    assert output["branch_features"].shape == (3, 2, 16)
    assert output["features"].shape == (3, 16)
    torch.testing.assert_close(output["features"], output["branch_features"].mean(dim=1))


def test_default_single_input_path_is_unchanged():
    model = build_visual(backend="single")
    assert model.input_plugin is None
    output = model(torch.randn(2, 3, 32, 16))
    assert output["features"].shape == (2, 16)


def test_single_input_checkpoint_initializes_all_quadruple_branches():
    single = build_visual(backend="single")
    quadruple = build_visual(backend="quadruple_patch")
    result = quadruple.load_state_dict(single.state_dict(), strict=True)

    assert result.missing_keys == []
    assert result.unexpected_keys == []
    source = single.vit.patch_embed.proj.weight
    branch_weights = [branch.proj.weight for branch in quadruple.input_plugin.patch_embeds]
    assert all(torch.equal(source, weight) for weight in branch_weights)
    assert len({weight.data_ptr() for weight in branch_weights}) == 4


def test_phased_template_is_trainable_and_can_initialize_all_branches():
    model = build_visual(template_trainable=True)
    assert model.vit.patch_embed.proj.weight.requires_grad
    with torch.no_grad():
        model.vit.patch_embed.proj.weight.add_(1.0)
    model.sync_input_plugin_from_template()
    source = model.vit.patch_embed.proj.weight
    assert all(
        torch.equal(source, branch.proj.weight)
        for branch in model.input_plugin.patch_embeds
    )
    output = model(torch.randn(2, 3, 32, 16), mode="shared_template")
    assert output["features"].shape == (2, 16)


class _Source:
    def __init__(self, value):
        self.image = np.full((8, 4, 3), value, dtype=np.uint8)

    def sample(self, _index):
        return self.image, 0


class _RecordTransform:
    def __init__(self, calls):
        self.calls = calls

    def __call__(self, image):
        self.calls.append(int(np.asarray(image).mean()))
        return torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float()


def test_four_augmentations_receive_images_returned_by_sr_sources():
    calls = []
    dataset = SYSU_Tri_Data.__new__(SYSU_Tri_Data)
    dataset.cIndex = np.array([0])
    dataset.tIndex = np.array([0])
    dataset.train_color_label = np.array([7])
    dataset.train_thermal_label = np.array([7])
    dataset.rgb_visual_source = _Source(31)
    dataset.ir_visual_source = _Source(79)
    dataset.transform1 = _RecordTransform(calls)
    dataset.transform2 = _RecordTransform(calls)
    dataset.transform3 = _RecordTransform(calls)
    dataset.transform4 = _RecordTransform(calls)
    dataset.joint_mode = "image_only"

    batch = dataset[0]

    assert calls == [31, 31, 79, 79]
    assert set(batch) == {
        "img_rgb_ori",
        "img_rgb_aug",
        "img_ir",
        "img_ir_aug",
        "target_rgb",
        "target_ir",
    }


def test_phased_dataset_materializes_only_the_active_epoch_inputs():
    calls = []
    dataset = SYSU_Tri_Data.__new__(SYSU_Tri_Data)
    dataset.cIndex = np.array([0])
    dataset.tIndex = np.array([0])
    dataset.train_color_label = np.array([7])
    dataset.train_thermal_label = np.array([7])
    dataset.rgb_visual_source = _Source(31)
    dataset.ir_visual_source = _Source(79)
    dataset.transform1 = _RecordTransform(calls)
    dataset.transform2 = _RecordTransform(calls)
    dataset.transform3 = _RecordTransform(calls)
    dataset.transform4 = None
    dataset.phased_transforms = tuple(_RecordTransform(calls) for _ in range(4))
    dataset.training_phase = "pmt"
    dataset.joint_mode = "image_only"

    warmup = dataset[0]
    assert calls == [31, 31, 79]
    assert set(warmup) == {
        "img_rgb_ori", "img_rgb_aug", "img_ir", "target_rgb", "target_ir",
    }

    calls.clear()
    dataset.set_training_phase("mscm")
    quadruple = dataset[0]
    assert calls == [31, 31, 79, 79]
    assert set(quadruple) == {
        "img_mscm_rgb1", "img_mscm_rgb2", "img_mscm_ir1", "img_mscm_ir2",
        "target_rgb", "target_ir",
    }


def test_quadruple_pipeline_matches_original_mscmnet_operator_order():
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    exact_rgb = ExactSize((8, 4), field_name="RGB SR")
    exact_ir = ExactSize((8, 4), field_name="IR SR")
    pipelines = build_mscmnet_exact_quadruple_transforms(
        (8, 4), normalize, [exact_rgb], [exact_ir]
    )
    names = [
        [type(step).__name__ for step in pipeline.transforms]
        for pipeline in pipelines
    ]
    assert names == [
        [
            "ToPILImage", "ExactSize", "RandomGrayscale", "Pad", "RandomCrop",
            "RandomHorizontalFlip", "ToTensor", "Normalize", "ChannelRandomErasing",
        ],
        [
            "ToPILImage", "ExactSize", "Pad", "RandomCrop",
            "RandomHorizontalFlip", "ToTensor", "Normalize", "ChannelRandomErasing",
            "MSCMChannelExchange",
        ],
        [
            "ToPILImage", "ExactSize", "Pad", "RandomCrop",
            "RandomHorizontalFlip", "ToTensor", "Normalize", "ChannelRandomErasing",
            "MSCMChannelAdapGray",
        ],
        [
            "ToPILImage", "ExactSize", "ColorJitter", "Pad", "RandomCrop",
            "RandomHorizontalFlip", "ToTensor", "Normalize", "ChannelRandomErasing",
            "MSCMChannelT",
        ],
    ]


def test_exact_mscmnet_channel_operators_preserve_original_rng_semantics():
    image = torch.stack([
        torch.full((2, 2), 1.0),
        torch.full((2, 2), 2.0),
        torch.full((2, 2), 3.0),
    ])
    with patch("salt_vi.data.processing.random.randint", return_value=1):
        exchanged = MSCMChannelExchange(gray=2)(image.clone())
    torch.testing.assert_close(exchanged, torch.full_like(exchanged, 2.0))

    with patch("salt_vi.data.processing.random.randint", return_value=0):
        adapted = MSCMChannelAdapGray(probability=0.5)(image.clone())
    torch.testing.assert_close(adapted, torch.full_like(adapted, 2.0))

    with patch(
        "salt_vi.data.processing.random.uniform",
        side_effect=[0.0, 0.1, 0.2, 0.3],
    ):
        scaled = MSCMChannelT(probability=0.5)(image.clone())
    expected = torch.stack([
        torch.full((2, 2), 0.2),
        torch.full((2, 2), 0.8),
        torch.full((2, 2), 1.8),
    ])
    torch.testing.assert_close(scaled, expected)


def test_quadruple_patch_parameters_are_pmt_layer_zero():
    name = "base_model.visual.input_plugin.patch_embeds.2.proj.weight"
    assert pmt_visual_layer_id(name, depth=12) == 0


def test_quadruple_stage_a_config_resolves_and_requires_sr_inputs():
    config = load_train_configs(
        "configs/stage_a/plugins/quadruple_patch_pasd_512x256_b32.yaml"
    )
    validate_runtime_config(config)
    assert config.visual_input_backend == "quadruple_patch"
    assert config.pmt_patch_embed is None
    assert set(config.sysu_sr_modalities) == {"rgb", "ir"}
    assert config.sysu_sr_data_root


def test_recipe_variant_switch_preserves_original_and_selects_phased_recipe():
    original = SimpleNamespace(pmt_recipe=True, pmt_recipe_variant="original")
    phased = SimpleNamespace(pmt_recipe=True, pmt_recipe_variant="mscm_phased")
    assert isinstance(build_training_recipe(original, None), PMTRecipe)
    assert isinstance(build_training_recipe(phased, None), PMTMSCMPhasedRecipe)


def test_phased_stage_a_config_resolves_with_branch_level_qct():
    config = load_train_configs(
        "configs/stage_a/plugins/pmt_mscm_phased_pasd_512x256_b32.yaml"
    )
    validate_runtime_config(config)
    assert config.pmt_recipe_variant == "mscm_phased"
    assert config.quadruple_template_trainable is True
    assert config.pmt_progressive_epoch == 6
    assert config.pmt_mscm_transition_epochs == 4
    assert config.pmt_mscm_qct_margin == 1.2
    assert config.pmt_mscm_qct_weight == 0.1
    assert config.pmt_mscm_qct_branch_weight == 0.25
    assert config.pmt_gradient_checkpoint_blocks == 7
    assert config.pmt_gradient_checkpoint_blocks_warmup == 3
    assert config.pmt_gradient_checkpoint_segments == 3


def test_phased_evaluation_uses_template_only_before_switch():
    owner = SimpleNamespace(
        args=SimpleNamespace(
            pmt_recipe_variant="mscm_phased", pmt_progressive_epoch=6
        ),
        _evaluation_epoch=5,
    )
    assert (
        CLIP2ReID._resolve_evaluation_visual_mode(owner, "rgb")
        == "shared_template"
    )
    assert (
        CLIP2ReID._resolve_evaluation_visual_mode(owner, "ir")
        == "shared_template"
    )
    owner._evaluation_epoch = 6
    assert CLIP2ReID._resolve_evaluation_visual_mode(owner, "rgb") == "rgb"
    owner._evaluation_epoch = None
    assert CLIP2ReID._resolve_evaluation_visual_mode(owner, "ir") == "ir"


class _RecipeBase:
    def encode_image(self, views, mode):
        del mode
        branch_features = [
            views[:, index].mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1).repeat(1, 2)
            for index in range(4)
        ]
        return {"features": torch.cat(branch_features, dim=0)}


class _RecipeModel:
    def __init__(self):
        self.args = SimpleNamespace(
            pretrain_choice="PMT_VIT",
            training_mode="RGB_IR",
            visual_input_backend="quadruple_patch",
            Fix_Visual=False,
            rfa_probability=0.0,
            rfa_gaussian_sigma=0.1,
            pmt_metric_loss="legacy",
            triplet_mining="wrt",
            id_loss_weight=1.0,
            pmt_msel_weight=0.5,
            pmt_dcl_weight=0.5,
        )
        self.base_model = _RecipeBase()
        self.logit_scale = torch.tensor(1.0)
        self.pid_criterion = torch.nn.CrossEntropyLoss()
        self.tri_criterion = lambda features, labels: features.square().mean()
        self.pmt_msel_criterion = lambda features, labels: features.square().mean()
        self.pmt_dcl_criterion = lambda features, labels: features.square().mean()
        self.last_classifier_features = None

    def _assert_pmt_batch_layout(self, label_rgb, label_ir):
        assert torch.equal(label_rgb, label_ir)

    def _get_visual_embedding(self, visual):
        return visual["features"]

    def classifier(self, features):
        self.last_classifier_features = features
        scores = torch.stack((features[:, 0], -features[:, 0]), dim=1)
        return features, scores


def test_quadruple_recipe_reduces_views_before_existing_losses():
    model = _RecipeModel()
    batch = {
        "img_rgb_ori": torch.ones(2, 3, 4, 2),
        "img_rgb_aug": torch.full((2, 3, 4, 2), 3.0),
        "img_ir": torch.full((2, 3, 4, 2), 5.0),
        "img_ir_aug": torch.full((2, 3, 4, 2), 7.0),
        "target_rgb": torch.zeros(2, dtype=torch.long),
        "target_ir": torch.zeros(2, dtype=torch.long),
    }

    result = PMTRecipe().compute_losses(model, batch)

    assert result["pmt_stage"] == "quadruple_rgb_ir"
    assert model.last_classifier_features.shape == (4, 2)
    torch.testing.assert_close(
        model.last_classifier_features[:, 0], torch.tensor([2.0, 2.0, 6.0, 6.0])
    )


class _PhasedBase:
    def __init__(self):
        self.last_mode = None

    def encode_image(self, images, mode):
        self.last_mode = mode
        if images.ndim == 4:
            values = images.mean(dim=(1, 2, 3))
            return {"features": values.unsqueeze(1).repeat(1, 2)}
        values = images.mean(dim=(2, 3, 4))
        branch_features = values.unsqueeze(-1).repeat(1, 1, 2)
        return {"branch_features": branch_features}


class _FakeQCT:
    def __init__(self):
        self.last_shape = None

    def __call__(self, branch_features, labels, return_components=False):
        del labels
        assert return_components
        self.last_shape = tuple(branch_features.shape)
        value = branch_features.square().mean()
        components = {
            "modality_compactness": value,
            "branch_compactness": value,
            "negative_margin": value,
            "hard_negative_distance": value,
        }
        return value, components


class _PhasedRecipeModel:
    def __init__(self):
        self.args = SimpleNamespace(
            pretrain_choice="PMT_VIT",
            training_mode="RGB_IR",
            pmt_progressive_epoch=6,
            rfa_probability=0.0,
            rfa_gaussian_sigma=0.1,
            triplet_mining="pmt_cross_modal_hard",
            id_loss_weight=1.0,
            pmt_cross_modal_triplet_weight=1.0,
            pmt_mscm_qct_weight=0.1,
            pmt_mscm_transition_epochs=4,
        )
        self.base_model = _PhasedBase()
        self.logit_scale = torch.tensor(1.0)
        self.pid_criterion = torch.nn.CrossEntropyLoss()
        self.pmt_tri_criterion = lambda left, right, labels: (
            left.square().mean() + right.square().mean()
        )
        self.tri_criterion = lambda features, labels: features.square().mean()
        self.cross_calls = 0
        self.pmt_qct_criterion = _FakeQCT()
        self.last_classifier_features = None

    def _assert_pmt_batch_layout(self, label_rgb, label_ir):
        assert torch.equal(label_rgb, label_ir)

    def _get_visual_embedding(self, visual):
        return visual["features"]

    def _slice_visual_output(self, visual, start, end):
        return {"features": visual["features"][start:end]}

    def classifier(self, features):
        self.last_classifier_features = features
        return features, torch.stack((features[:, 0], -features[:, 0]), dim=1)

    def cross_modal_tri_criterion(self, visible, infrared, labels):
        del labels
        self.cross_calls += 1
        return (visible - infrared).square().mean()

def test_phased_recipe_keeps_original_gray_warmup_before_epoch_six():
    model = _PhasedRecipeModel()
    batch = {
        "img_rgb_aug": torch.full((2, 3, 4, 2), 2.0),
        "img_ir": torch.full((2, 3, 4, 2), 4.0),
        "target_rgb": torch.zeros(2, dtype=torch.long),
        "target_ir": torch.zeros(2, dtype=torch.long),
    }
    result = PMTMSCMPhasedRecipe().compute_losses(
        model, batch, current_epoch=5
    )
    assert result["pmt_stage"] == "gray_ir"
    assert model.base_model.last_mode == "shared_template"
    assert model.last_classifier_features.shape == (4, 2)
    assert "msel_loss" not in result
    assert result["qct_loss"].item() == 0.0


def test_phased_recipe_supervises_four_branches_without_feature_averaging():
    model = _PhasedRecipeModel()
    batch = {
        "img_mscm_rgb1": torch.full((2, 3, 4, 2), 1.0),
        "img_mscm_rgb2": torch.full((2, 3, 4, 2), 3.0),
        "img_mscm_ir1": torch.full((2, 3, 4, 2), 5.0),
        "img_mscm_ir2": torch.full((2, 3, 4, 2), 7.0),
        "target_rgb": torch.zeros(2, dtype=torch.long),
        "target_ir": torch.zeros(2, dtype=torch.long),
    }
    result = PMTMSCMPhasedRecipe().compute_losses(
        model, batch, current_epoch=6
    )
    assert result["pmt_stage"] == "mscm_quadruple"
    assert model.cross_calls == 4
    assert "msel_loss" not in result
    assert result["pmt_mscm_transition_alpha"] == 0.0
    assert result["pmt_mscm_qct_effective_weight"] == 0.0
    assert result["qct_loss"].item() == 0.0
    assert model.pmt_qct_criterion.last_shape == (2, 4, 2)
    torch.testing.assert_close(
        model.last_classifier_features[:, 0],
        torch.tensor([1.0, 1.0, 3.0, 3.0, 5.0, 5.0, 7.0, 7.0]),
    )


def test_branch_aware_qct_is_finite_and_backpropagates_to_all_four_views():
    torch.manual_seed(7)
    features = torch.randn(8, 4, 16, requires_grad=True)
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    criterion = PMTQuadrupleCenterTripletLoss(
        margin=0.7, branch_weight=1.0
    )
    loss, components = criterion(features, labels, return_components=True)
    assert torch.isfinite(loss)
    assert set(components) == {
        "modality_compactness", "branch_compactness", "negative_margin",
        "hard_negative_distance",
    }
    loss.backward()
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()
    assert torch.all(features.grad.abs().sum(dim=(0, 2)) > 0)


def test_phase_switch_reuses_optimizer_and_copies_template_adam_state():
    visual = build_visual(template_trainable=True)
    owner = SimpleNamespace(
        args=SimpleNamespace(
            pmt_recipe_variant="mscm_phased", pmt_progressive_epoch=6
        ),
        base_model=SimpleNamespace(visual=visual),
        _phased_quadruple_synced=False,
    )
    owner.sync_phased_quadruple_patch_embeddings = MethodType(
        CLIP2ReID.sync_phased_quadruple_patch_embeddings, owner
    )
    optimizer = torch.optim.AdamW(visual.parameters(), lr=1e-3)
    output = visual(
        torch.randn(2, 3, 32, 16), mode="shared_template"
    )["features"]
    output.square().mean().backward()
    optimizer.step()

    template_weight = visual.vit.patch_embed.proj.weight
    branch_weights = [
        branch.proj.weight for branch in visual.input_plugin.patch_embeds
    ]
    assert template_weight in optimizer.state
    assert all(weight not in optimizer.state for weight in branch_weights)

    summary = CLIP2ReID.prepare_pmt_mscm_phase(owner, 6, optimizer)

    assert summary["reused_optimizer"] is True
    assert summary["copied_parameter_states"] > 0
    assert summary["template_parameters_without_state"] == []
    for weight in branch_weights:
        assert weight in optimizer.state
        torch.testing.assert_close(weight, template_weight)
        torch.testing.assert_close(
            optimizer.state[weight]["exp_avg"],
            optimizer.state[template_weight]["exp_avg"],
        )
        assert (
            optimizer.state[weight]["exp_avg"].data_ptr()
            != optimizer.state[template_weight]["exp_avg"].data_ptr()
        )


def test_phased_recipe_selects_warmup_and_four_view_checkpoint_budgets():
    visual = build_visual(template_trainable=True)
    owner = SimpleNamespace(
        args=SimpleNamespace(
            pmt_recipe_variant="mscm_phased",
            pmt_progressive_epoch=6,
            pmt_gradient_checkpoint_blocks=7,
            pmt_gradient_checkpoint_blocks_warmup=3,
        ),
        base_model=SimpleNamespace(visual=visual),
        _phased_quadruple_synced=False,
    )
    owner.sync_phased_quadruple_patch_embeddings = MethodType(
        CLIP2ReID.sync_phased_quadruple_patch_embeddings, owner
    )
    optimizer = torch.optim.AdamW(visual.parameters(), lr=1e-3)
    CLIP2ReID.prepare_pmt_mscm_phase(owner, 5, optimizer)
    assert visual.gradient_checkpoint_blocks == 3
    CLIP2ReID.prepare_pmt_mscm_phase(owner, 6, optimizer)
    assert visual.gradient_checkpoint_blocks == 7
