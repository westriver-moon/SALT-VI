from types import SimpleNamespace

import numpy as np
import pytest
import torch

from salt_vi.config.validation import validate_runtime_config
from salt_vi.data.dataset import _lookup_text_description
from salt_vi.retrieval import get_retrieval_protocol
from salt_vi.retrieval import evaluator
from salt_vi.training import build_training_recipe


def _config(**overrides):
    values = {
        "retrieval_backend": "ir_to_rgb_text",
        "dataset": "sysu",
        "training_mode": "RGB_IR_Text",
        "joint_mode": "uni",
        "loss_names": "id,cross_modal_hard",
        "uni_BN": False,
        "Fix_Visual": True,
        "Feat_Filter": False,
        "fixed_visual_data_parallel": False,
        "visual_unfreeze_last_n_blocks": 0,
        "sysu_sr_backend": "array",
        "test_modality": "IR-RGBText",
        "gallery_caption_manifest": "/captions/rgb.json",
        "gallery_text_dropout": 0.3,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_backend_contract_and_runtime_config():
    backend = get_retrieval_protocol("ir_to_rgb_text")
    assert backend.TRAIN_TEXT_MODALITIES == ("rgb",)
    assert backend.QUERY_CAPTION_LOOKUP is None
    assert backend.GALLERY_CAPTION_LOOKUP == "image"
    assert validate_runtime_config(_config()).retrieval_backend == "ir_to_rgb_text"


def test_identity_text_is_an_explicit_protocol():
    protocol = get_retrieval_protocol("identity_text")
    config = SimpleNamespace(test_modality="IR,Fusion,Text")
    assert protocol.NAME == "identity_text"
    assert protocol.RESULT_KEYS == ("IR", "Fusion", "Text")
    assert protocol.RESULT_KEY == "Fusion"
    assert protocol.train_text_modalities(config) == ("rgb", "ir")
    assert protocol.query_caption_lookup(config) == "identity"


def test_legacy_alias_resolves_to_identity_text_protocol():
    protocol = get_retrieval_protocol("legacy")
    assert protocol.NAME == "identity_text"
    assert get_retrieval_protocol("identity_text") is protocol


def test_identity_text_protocol_rejects_invalid_runtime_combinations():
    protocol = get_retrieval_protocol("identity_text")
    assert protocol.validate(
        SimpleNamespace(dataset="sysu", test_modality="Fusion")
    ).test_modality == "Fusion"
    assert protocol.query_caption_lookup(
        SimpleNamespace(test_modality="IR")
    ) is None

    with pytest.raises(ValueError, match="supports only"):
        protocol.validate(SimpleNamespace(dataset="market1501", test_modality="IR"))
    with pytest.raises(ValueError, match="non-empty subset"):
        protocol.validate(SimpleNamespace(dataset="sysu", test_modality="IR-RGBText"))
    with pytest.raises(ValueError, match="direction"):
        protocol.validate(
            SimpleNamespace(
                dataset="regdb",
                test_modality="IR",
                regdb_test_mode="thermal-visible",
                trial=1,
                eval_num_regdb=1,
            )
        )
    with pytest.raises(ValueError, match="within 1-10"):
        protocol.validate(
            SimpleNamespace(
                dataset="regdb",
                test_modality="Fusion",
                regdb_test_mode="t-v",
                trial=9,
                eval_num_regdb=3,
            )
        )


def test_training_recipe_dispatch_is_owned_by_protocol():
    config = SimpleNamespace(pmt_recipe=False, training_mode="RGB_IR_Text")
    protocol = get_retrieval_protocol("ir_to_rgb_text")
    assert build_training_recipe(config, protocol).name == "ir_to_rgb_text"
    assert build_training_recipe(config, get_retrieval_protocol("identity_text")).name == (
        "identity_text_rgb_ir_text"
    )


def test_gallery_caption_lookup_uses_image_path():
    captions = {
        "datasets/sysu/cam1/0001/0001.jpg": {"description": "person in a light coat"}
    }
    result = _lookup_text_description(
        captions,
        "sysu",
        "/datasets/SYSU-MM01/",
        "/datasets/SYSU-MM01/cam1/0001/0001.jpg",
    )
    assert result == "person in a light coat"


class _BaseModel:
    @staticmethod
    def encode_text(text):
        return text.float()


class _TrainingModel:
    def __init__(self):
        self.base_model = _BaseModel()
        self.args = SimpleNamespace(
            fusion_way="add",
            gallery_text_dropout=0.0,
            id_loss_weight=1.0,
            cross_modal_hard_weight=1.0,
            ir_rgb_text_pair_weight=1.0,
            ir_rgb_aux_weight=0.5,
        )
        self.pid_criterion = torch.nn.CrossEntropyLoss()

    @staticmethod
    def _slice_visual_output(features, start, end):
        return features[start:end]

    @staticmethod
    def current_pa():
        return 0.5

    @staticmethod
    def fusion_layer(text_map, image_map, caption_ids, pa, way):
        return text_map + image_map

    @staticmethod
    def classifier(features):
        scores = torch.stack((features[:, 0], features[:, 1]), dim=1)
        return features, scores

    @staticmethod
    def cross_modal_tri_criterion(left, right, labels):
        return (left - right).square().mean()


def test_training_plugin_builds_only_protocol_losses():
    backend = get_retrieval_protocol("ir_to_rgb_text")
    model = _TrainingModel()
    labels = torch.tensor([0, 1])
    result = backend.training_losses(
        model,
        {"text_rgb": torch.tensor([[1, 0], [0, 1]])},
        torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.8, 0.2], [0.2, 0.8]]),
        torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.8, 0.2], [0.2, 0.8]]),
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        labels,
        labels,
        ["id", "cross_modal_hard"],
    )
    assert set(result) == {"id_loss", "cross_modal_hard_loss", "acc"}
    assert all(torch.isfinite(value) for value in result.values())


class _EvaluationModel:
    @staticmethod
    def set_eval():
        return None

    @staticmethod
    def encode_image_featmap(image, mode):
        assert mode == "ir"
        return image

    @staticmethod
    def extract_global_feat(features):
        return features

    @staticmethod
    def classifier(features, mode):
        return torch.nn.functional.normalize(features.float(), dim=1)

    @staticmethod
    def encode_fusion(text, image, mode):
        assert mode == "rgb"
        return image.float() + text.float()


def test_evaluator_keeps_text_on_gallery_side(monkeypatch):
    backend = get_retrieval_protocol("ir_to_rgb_text")
    loader = SimpleNamespace(
        query_loader=[{"img": torch.tensor([[1.0, 0.0]])}],
        gallery_loaders=[
            [{"img": torch.tensor([[1.0, 0.0]]), "text": torch.tensor([[0, 1]])}]
            for _ in range(10)
        ],
        query_label=np.array([0]),
        query_cam=np.array([3]),
        gallery_labels=[np.array([0]) for _ in range(10)],
        gallery_cams=[np.array([1]) for _ in range(10)],
    )
    monkeypatch.setattr(
        evaluator,
        "eval_sysu",
        lambda *args: (np.array([1.0]), 1.0, 1.0),
    )
    result = evaluator.evaluate_sysu(
        _EvaluationModel(), loader, torch.device("cpu"), backend
    )
    assert result["IR-RGBText"][0:2] == (1.0, 1.0)
