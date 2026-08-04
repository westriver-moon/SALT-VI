from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from salt_vi.data.dataset import PIL_LANCZOS, _infer_dataset_name
from salt_vi.data.sampler import GenIdx, IdentitySampler, validate_identity_batch_config
from salt_vi.data.tokenizer import default_bpe
from salt_vi.engine.build import Classifier
from salt_vi.entrypoints.train import main
from salt_vi.models.model import Classifier as LegacyClassifier
from salt_vi.utils.utils import _expand_environment_values


def test_dataset_name_inference_accepts_trailing_separator():
    assert _infer_dataset_name("datasets/sysu") == "sysu"
    assert _infer_dataset_name("datasets/sysu/") == "sysu"
    assert _infer_dataset_name("datasets\\sysu\\") == "sysu"


def test_pillow_compatibility_uses_resampling_constant():
    assert PIL_LANCZOS is not None


def test_pk_batch_contract_rejects_non_divisible_batch_size():
    with pytest.raises(ValueError, match="divisible"):
        validate_identity_batch_config(batch_size=30, num_pos=4, number_of_identities=10)


def test_pk_batch_contract_rejects_too_many_identities():
    with pytest.raises(ValueError, match="only 3 are available"):
        validate_identity_batch_config(batch_size=8, num_pos=2, number_of_identities=3)


def test_identity_sampler_reports_actual_yield_length():
    labels = np.repeat(np.arange(4), 3)
    color_pos, thermal_pos = GenIdx(labels, labels)
    sampler = IdentitySampler(labels, labels, color_pos, thermal_pos, num_pos=2, batchSize=2)
    yielded = list(iter(sampler))
    assert len(sampler) == len(yielded) == len(sampler.index1)
    first_batch_labels = labels[sampler.index1[:4]]
    assert sorted(np.unique(first_batch_labels, return_counts=True)[1].tolist()) == [2, 2]


def test_uni_bn_rejects_incomplete_five_group_batch():
    classifier = Classifier(pid_num=3, dim=2, uni_BN=True, joint_mode="uni")
    classifier.train()
    with pytest.raises(ValueError, match="five equally sized modality groups"):
        classifier(torch.randn(11, 2))


def test_legacy_classifier_preserves_batch_dimension_for_singleton():
    classifier = LegacyClassifier(pid_num=3, dim=2)
    classifier.eval()
    output = classifier(torch.randn(1, 2))
    assert output.shape == (1, 2)


def test_packaged_bpe_resource_exists_in_source_tree():
    assert Path(default_bpe()).is_file()


def test_legacy_data_parallel_fails_before_loader_or_model_construction():
    with pytest.raises(RuntimeError, match="Legacy DataParallel is unsupported"):
        main(SimpleNamespace(DataParallel=True))


def test_config_environment_placeholders_expand_recursively(monkeypatch):
    monkeypatch.setenv("SALT_VI_TEST_ROOT", "/tmp/salt-vi")
    payload = {"path": "${SALT_VI_TEST_ROOT}/data", "items": ["${SALT_VI_TEST_ROOT}/a"]}
    assert _expand_environment_values(payload) == {
        "path": "/tmp/salt-vi/data",
        "items": ["/tmp/salt-vi/a"],
    }
