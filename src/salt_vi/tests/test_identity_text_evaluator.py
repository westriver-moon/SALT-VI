from contextlib import contextmanager
from importlib import import_module
from types import SimpleNamespace

import numpy as np
import torch

evaluator = import_module("salt_vi.engine.test")


def test_entrypoint_enables_cuda_autocast(monkeypatch):
    observed = {}

    class Base:
        @staticmethod
        def set_evaluation_epoch(current_epoch):
            observed["evaluation_epoch"] = current_epoch

    @contextmanager
    def fake_autocast(device_type, enabled):
        observed["autocast"] = (device_type, enabled)
        yield

    class Protocol:
        def evaluate(self, base, loader, config, device):
            observed["evaluated"] = True
            return {"IR": "ok"}

    monkeypatch.setattr(evaluator.torch.amp, "autocast", fake_autocast)
    monkeypatch.setattr(evaluator, "get_retrieval_protocol", lambda name: Protocol())
    result = evaluator.test(
        Base(),
        object(),
        SimpleNamespace(retrieval_backend="identity_text"),
        torch.device("cuda"),
        current_epoch=5,
    )
    assert observed == {
        "evaluation_epoch": 5,
        "autocast": ("cuda", True),
        "evaluated": True,
    }
    assert result == {"IR": "ok"}


class Model:
    @staticmethod
    def set_eval():
        return None

    @staticmethod
    def _uses_spatial_map_visual():
        return False

    @staticmethod
    def encode_image_featmap(image, mode):
        assert torch.is_inference_mode_enabled()
        return image.float()

    @staticmethod
    def encode_image_feat(image, mode):
        assert torch.is_inference_mode_enabled()
        return image.float()

    @staticmethod
    def extract_global_feat(features):
        return features

    @staticmethod
    def classifier(features, mode="RGB"):
        return features.float()

    @staticmethod
    def encode_text_feat(text):
        return text.float()

    @staticmethod
    def encode_fusion(text, image, mode):
        return text.float() + image.float()


def test_sysu_identity_text_modalities_share_one_trial_pipeline(monkeypatch):
    monkeypatch.setattr(
        evaluator,
        "eval_sysu",
        lambda distance, *unused: (
            np.asarray([distance[0, 0]]),
            float(distance[0, 0]),
            float(distance[0, 0]),
        ),
    )
    loader = SimpleNamespace(
        dataset="sysu",
        query_loader=[
            {"img": torch.tensor([[1.0, 0.0]]), "text": torch.tensor([[0, 1]])}
        ],
        gallery_loaders=[
            [{"img": torch.tensor([[1.0, 0.0]])}] for _ in range(2)
        ],
        query_label=np.asarray([0]),
        query_cam=np.asarray([3]),
        gallery_labels=[np.asarray([0]) for _ in range(2)],
        gallery_cams=[np.asarray([1]) for _ in range(2)],
    )
    config = SimpleNamespace(
        test_modality="IR,Fusion,Text",
        Fix_Visual=False,
        Feat_Filter=False,
        CAT_EVAL=True,
    )
    result = evaluator.evaluate_identity_text(Model(), loader, config, torch.device("cpu"))
    assert result["IR"][2][0] == -1.0
    assert result["Fusion"][2][0] == -2.0
    assert result["Text"][2][0] == 0.0


def test_regdb_reverse_mode_reverses_features_and_labels(monkeypatch):
    calls = []

    def metric(distance, query_label, gallery_label):
        calls.append((distance.shape, query_label.tolist(), gallery_label.tolist()))
        return np.asarray([1.0]), 1.0, 1.0

    monkeypatch.setattr(evaluator, "eval_regdb", metric)
    loader = SimpleNamespace(
        dataset="regdb",
        query_loaders=[[{"img": torch.tensor([[1.0, 0.0]])}]],
        gallery_loaders=[
            [{"img": torch.tensor([[1.0, 0.0], [0.0, 1.0]])}]
        ],
        query_labels=[np.asarray([7])],
        gallery_labels=[np.asarray([8, 9])],
    )
    config = SimpleNamespace(
        test_modality="IR",
        Fix_Visual=False,
        CAT_EVAL=False,
        regdb_test_mode="v-t",
    )
    evaluator.evaluate_identity_text(Model(), loader, config, torch.device("cpu"))
    assert calls == [((2, 1), [8, 9], [7])]
