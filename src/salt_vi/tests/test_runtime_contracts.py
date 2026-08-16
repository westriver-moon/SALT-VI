from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
import hashlib

import numpy as np
import pytest
import torch

from salt_vi.config.validation import (
    validate_runtime_config,
    validate_selected_config_schema,
)
from salt_vi.data.dataset import PIL_LANCZOS, _infer_dataset_name, _resolve_text_dir
from salt_vi.data.sampler import GenIdx, IdentitySampler, validate_identity_batch_config
from salt_vi.data.tokenizer import default_bpe
from salt_vi.engine.build import Classifier
from salt_vi.engine.train import handle_nonfinite_gradients
from salt_vi.entrypoints import train as train_entry
from salt_vi.entrypoints.train import main
from salt_vi.models.clip_model.clip_model import CLIP
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


def test_identity_sampler_accepts_non_contiguous_labels():
    labels = np.repeat(np.array([10, 20, 30, 40]), 3)
    color_pos, thermal_pos = GenIdx(labels, labels)
    sampler = IdentitySampler(labels, labels, color_pos, thermal_pos, num_pos=2, batchSize=2)
    sampled_labels = labels[sampler.index1]
    assert set(sampled_labels).issubset({10, 20, 30, 40})


def test_identity_sampler_rejects_inconsistent_modal_identity_sets():
    with pytest.raises(ValueError, match="identity sets are inconsistent"):
        GenIdx(np.array([1, 1, 2, 2]), np.array([1, 1, 3, 3]))


def test_uni_bn_rejects_incomplete_five_group_batch():
    classifier = Classifier(pid_num=3, dim=2, uni_BN=True, joint_mode="uni")
    classifier.train()
    with pytest.raises(ValueError, match="five equally sized modality groups"):
        classifier(torch.randn(11, 2))


def test_packaged_bpe_resource_exists_in_source_tree():
    assert Path(default_bpe()).is_file()


def test_legacy_data_parallel_fails_before_loader_or_model_construction():
    with pytest.raises(RuntimeError, match="Legacy DataParallel is unsupported"):
        main(SimpleNamespace(DataParallel=True))


def test_runtime_validation_rejects_unimplemented_text_mode():
    with pytest.raises(ValueError, match="Unsupported joint_mode"):
        validate_runtime_config(
            {"training_mode": "RGB_IR_Text", "joint_mode": "dual_text"}
        )


def test_runtime_validation_rejects_qbn_id_woir_combo():
    with pytest.raises(ValueError, match="incompatible with id_woir"):
        validate_runtime_config(
            {
                "training_mode": "RGB_IR_Text",
                "joint_mode": "uni",
                "uni_BN": True,
                "loss_names": "id_woir",
            }
        )


def test_runtime_validation_rejects_removed_return_before_bn_flag():
    with pytest.raises(ValueError, match="Return_B4_BN was a no-op"):
        validate_runtime_config(
            {"training_mode": "RGB_IR", "joint_mode": "image_only", "Return_B4_BN": True}
        )


def test_runtime_validation_rejects_non_positive_batch_size():
    with pytest.raises(ValueError, match="batch_size must be >= 1"):
        validate_runtime_config({"batch_size": 0})


def test_runtime_validation_rejects_non_positive_temperature():
    with pytest.raises(ValueError, match="temperature must be > 0"):
        validate_runtime_config({"temperature": 0.0})


def test_runtime_validation_rejects_invalid_image_size_pair():
    with pytest.raises(ValueError, match="img_size must be a pair"):
        validate_runtime_config({"img_size": (288, 0)})


def test_runtime_validation_rejects_model_only_resume():
    with pytest.raises(ValueError, match="model-only resume"):
        validate_runtime_config(
            {"dataset": "sysu", "test_modality": "Fusion", "resume_train_epoch": 3}
        )


def test_runtime_validation_rejects_metric_boost_resume():
    with pytest.raises(ValueError, match="metric_boost_resume_epoch is retired"):
        validate_runtime_config(
            {
                "dataset": "sysu",
                "test_modality": "Fusion",
                "metric_boost_resume_epoch": 2,
            }
        )


def test_external_text_root_has_priority(tmp_path):
    external = tmp_path / "portable_text"
    expected = external / "Blip_RGB"
    expected.mkdir(parents=True)
    assert _resolve_text_dir(
        str(tmp_path / "regdb"), "regdb", "Blip", "RGB", str(external)
    ) == str(expected) + os.sep


def _disabled_scaler():
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=False)
    return torch.cuda.amp.GradScaler(enabled=False)


def test_cpu_nonfinite_gradient_does_not_step_optimizer():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    parameter.grad = torch.tensor([float("inf")])
    with pytest.raises(FloatingPointError, match="without an active AMP scaler"):
        handle_nonfinite_gradients(_disabled_scaler(), optimizer, ["parameter"])
    assert parameter.item() == 1.0
    assert parameter.grad is None


def test_training_checkpoint_round_trip_restores_full_state(tmp_path):
    train_entry._reset_best_metrics()
    train_entry.best_rank1_fusion = 0.8
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    scaler = _disabled_scaler()
    loss = model(torch.ones(1, 2)).sum()
    loss.backward()
    optimizer.step()
    scheduler.step()
    expected_weight = model.weight.detach().clone()
    checkpoint_path = tmp_path / "checkpoint_latest.pth"
    train_entry._save_training_checkpoint(
        str(checkpoint_path),
        4,
        model,
        optimizer,
        scheduler,
        scaler,
        run_uuid="run-1",
        run_manifest_sha256="manifest-hash-1",
    )
    with torch.no_grad():
        model.weight.zero_()
    assert train_entry._load_training_checkpoint(
        str(checkpoint_path),
        model,
        optimizer,
        scheduler,
        scaler,
        torch.device("cpu"),
        expected_run_uuid="run-1",
        expected_run_manifest_sha256="manifest-hash-1",
    ) == 5
    assert torch.equal(model.weight, expected_weight)
    assert train_entry.best_rank1_fusion == 0.8


def test_training_checkpoint_rejects_run_identity_mismatch_before_state_load(tmp_path):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    scaler = _disabled_scaler()
    checkpoint_path = tmp_path / "checkpoint_latest.pth"
    train_entry._save_training_checkpoint(
        str(checkpoint_path),
        3,
        model,
        optimizer,
        scheduler,
        scaler,
        run_uuid="expected-run",
        run_manifest_sha256="expected-manifest",
    )
    with torch.no_grad():
        model.weight.zero_()
    with pytest.raises(ValueError, match="run UUID mismatch"):
        train_entry._load_training_checkpoint(
            str(checkpoint_path),
            model,
            optimizer,
            scheduler,
            scaler,
            torch.device("cpu"),
            expected_run_uuid="different-run",
            expected_run_manifest_sha256="expected-manifest",
        )
    assert torch.equal(model.weight, torch.zeros_like(model.weight))


class _ProtocolStub:
    identifier = "stub-protocol"
    eval_caption_seed = 0

    def as_dict(self):
        return {"identifier": self.identifier}


def test_run_manifest_round_trip_hashes_config_data_and_protocol(tmp_path):
    view_manifest = tmp_path / "manifest.jsonl"
    view_manifest.write_text("source_1\nsource_2\n", encoding="utf-8")
    config = SimpleNamespace(
        output_path=str(tmp_path / "run"),
        mode="test",
        test_model_path=None,
        training_weight_init=None,
        sysu_sr_view_manifest=str(view_manifest),
        sysu_sr_data_root=None,
        dataset="sysu",
        batch_size=32,
        seed=1,
    )
    run_uuid = "fresh-run-uuid"
    path, manifest_hash = train_entry._write_run_manifest(
        config, run_uuid, _ProtocolStub()
    )
    assert Path(path).is_file()
    manifest = train_entry._load_run_manifest(config)
    assert manifest["run_uuid"] == run_uuid
    assert manifest["resolved_config_sha256"] == train_entry._resolved_config_digest(config)
    assert manifest["data_manifest"]["paths"] == [str(view_manifest)]
    assert manifest["protocol_identifier"] == "stub-protocol"
    assert manifest_hash == train_entry._sha256_file(path)
    train_entry._validate_run_manifest(config, _ProtocolStub(), manifest)


def test_golden_evaluation_writes_structured_metrics(tmp_path):
    checkpoint = tmp_path / "model_Fusion_epoch_6.pth"
    checkpoint.write_bytes(b"checkpoint-bytes")
    output_root = tmp_path / "golden-run"
    config = SimpleNamespace(
        output_path=str(output_root),
        mode="test",
        test_model_path=str(checkpoint),
        training_weight_init=None,
        sysu_sr_view_manifest=None,
        sysu_sr_data_root=None,
        dataset="sysu",
        batch_size=32,
        seed=1,
        golden_evaluation_path=str(tmp_path / "golden_evaluation.json"),
        run_manifest_sha256="manifest-hash",
        metric_experiment_id="GOLDEN-1",
    )
    path = train_entry._write_golden_evaluation(
        config,
        _ProtocolStub(),
        {"Fusion": (0.6862, 0.7960, [0.8293])},
    )
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["checkpoint_sha256"] == train_entry._sha256_file(str(checkpoint))
    assert payload["metrics"]["Fusion"]["Rank-1"] == 0.8293
    assert payload["metrics"]["Fusion"]["mAP"] == 0.7960
    assert payload["metrics"]["Fusion"]["mINP"] == 0.6862


def test_training_checkpoint_loader_requests_full_state_when_supported(monkeypatch):
    observed = {}

    def simulated_torch_load(path, map_location=None, weights_only=True):
        observed["path"] = path
        observed["map_location"] = map_location
        observed["weights_only"] = weights_only
        return {"loaded": True}

    monkeypatch.setattr(train_entry.torch, "load", simulated_torch_load)
    assert train_entry._load_trusted_training_checkpoint("trusted.pth") == {"loaded": True}
    assert observed["weights_only"] is False


class _ClipStateStub:
    visual_name = "ViT-B/16"

    def __init__(self):
        self._state = {
            "token_embedding.weight": torch.zeros(2, 2),
            "positional_embedding": torch.zeros(2, 2),
            "transformer.resblocks.0.attn.in_proj_weight": torch.zeros(2, 2),
            "ln_final.weight": torch.zeros(2),
            "ln_final.bias": torch.zeros(2),
            "text_projection": torch.zeros(2, 2),
        }
        self.positional_embedding = self._state["positional_embedding"]
        self.text_projection = self._state["text_projection"]

    def state_dict(self):
        return self._state


def test_clip_loader_requires_all_text_transformer_weights():
    stub = _ClipStateStub()
    incomplete = {
        key: torch.ones_like(value)
        for key, value in stub.state_dict().items()
        if not key.startswith("transformer.")
    }
    with pytest.raises(RuntimeError, match="transformer.resblocks"):
        CLIP.load_param(stub, incomplete)


def test_clip_loader_accepts_dataparallel_prefixed_full_text_state():
    stub = _ClipStateStub()
    prefixed = {
        "module." + key: torch.ones_like(value)
        for key, value in stub.state_dict().items()
    }
    summary = CLIP.load_param(stub, prefixed)
    assert len(summary["loaded_keys"]) == len(stub.state_dict())
    assert all(torch.equal(value, torch.ones_like(value)) for value in stub.state_dict().values())


def test_config_environment_placeholders_expand_recursively(monkeypatch):
    monkeypatch.setenv("SALT_VI_TEST_ROOT", "/tmp/salt-vi")
    payload = {"path": "${SALT_VI_TEST_ROOT}/data", "items": ["${SALT_VI_TEST_ROOT}/a"]}
    assert _expand_environment_values(payload) == {
        "path": "/tmp/salt-vi/data",
        "items": ["/tmp/salt-vi/a"],
    }


def test_active_config_schema_rejects_unknown_keys(tmp_path):
    project_root = tmp_path
    config_path = project_root / "configs" / "stage_b" / "bad.yaml"
    config_path.parent.mkdir(parents=True)
    with pytest.raises(KeyError, match="typo_field"):
        validate_selected_config_schema(
            {"dataset": "sysu", "typo_field": True},
            {"dataset": "sysu"},
            config_path,
            project_root,
        )


def test_active_config_schema_requires_checkpoint_hash(tmp_path):
    project_root = tmp_path
    config_path = project_root / "configs" / "stage_b" / "missing_hash.yaml"
    config_path.parent.mkdir(parents=True)
    with pytest.raises(ValueError, match="training_weight_init_sha256"):
        validate_selected_config_schema(
            {"training_weight_init": "warm-start.pth"},
            {"training_weight_init": None, "training_weight_init_sha256": None},
            config_path,
            project_root,
        )


def test_training_weight_init_sha256_gate(tmp_path):
    checkpoint = tmp_path / "warm-start.pth"
    checkpoint.write_bytes(b"checkpoint")
    config = SimpleNamespace(
        training_weight_init=str(checkpoint),
        training_weight_init_sha256="0" * 64,
    )
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        train_entry._verify_training_weight_init(config)
    expected = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    config.training_weight_init_sha256 = expected
    assert train_entry._verify_training_weight_init(config) == expected


def test_training_weight_init_requires_hash_after_runtime_overrides(tmp_path):
    checkpoint = tmp_path / "warm-start.pth"
    checkpoint.write_bytes(b"checkpoint")
    config = SimpleNamespace(
        training_weight_init=str(checkpoint),
        training_weight_init_sha256=None,
    )
    with pytest.raises(ValueError, match="required whenever training_weight_init is set"):
        train_entry._verify_training_weight_init(config)


def test_missing_weight_hash_fails_before_loader_or_model(monkeypatch, tmp_path):
    checkpoint = tmp_path / "warm-start.pth"
    checkpoint.write_bytes(b"checkpoint")
    config = SimpleNamespace(
        DataParallel=False,
        retrieval_backend="identity_text",
        CUDA_VISIBLE_DEVICES="0",
        gpu_id="0",
        mode="train",
        auto_resume_training_from_lastest_step=False,
        resume_train_epoch=-1,
        training_weight_init=str(checkpoint),
        training_weight_init_sha256=None,
    )
    protocol = SimpleNamespace(RESULT_KEY="Fusion")
    monkeypatch.setattr(train_entry, "validate_runtime_config", lambda value: value)
    monkeypatch.setattr(train_entry, "get_retrieval_protocol", lambda value: protocol)
    monkeypatch.setattr(train_entry, "build_protocol_spec", lambda *args: None)
    monkeypatch.setattr(train_entry, "resolve_run_directory", lambda value: str(tmp_path / "run"))
    monkeypatch.setattr(train_entry, "ensure_fresh_run_directory", lambda value: None)
    monkeypatch.setattr(
        train_entry,
        "Loader",
        lambda value: pytest.fail("Loader must not be constructed before SHA validation"),
    )
    monkeypatch.setattr(
        train_entry,
        "build_model",
        lambda value: pytest.fail("Model must not be constructed before SHA validation"),
    )
    with pytest.raises(ValueError, match="required whenever training_weight_init is set"):
        train_entry.main(config)


def test_set_override_cannot_remove_training_weight_init_hash(tmp_path):
    checkpoint = tmp_path / "warm-start.pth"
    checkpoint.write_bytes(b"checkpoint")
    expected = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    selected = tmp_path / "selected.yaml"
    selected.write_text(
        f"training_weight_init: {checkpoint}\n"
        f"training_weight_init_sha256: {expected}\n",
        encoding="utf-8",
    )
    cli = train_entry.get_args(
        [
            "--config_select",
            str(selected),
            "--set",
            "training_weight_init_sha256=null",
        ]
    )
    config = train_entry._merge_runtime_config(cli)
    with pytest.raises(ValueError, match="required whenever training_weight_init is set"):
        train_entry._verify_training_weight_init(config)


def test_cli_checkpoint_override_cannot_reuse_another_files_hash(tmp_path):
    first = tmp_path / "first.pth"
    second = tmp_path / "second.pth"
    first.write_bytes(b"first checkpoint")
    second.write_bytes(b"second checkpoint")
    expected = hashlib.sha256(first.read_bytes()).hexdigest()
    selected = tmp_path / "selected.yaml"
    selected.write_text(
        f"training_weight_init: {first}\n"
        f"training_weight_init_sha256: {expected}\n",
        encoding="utf-8",
    )
    cli = train_entry.get_args(
        [
            "--config_select",
            str(selected),
            "--training_weight_init",
            str(second),
        ]
    )
    config = train_entry._merge_runtime_config(cli)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        train_entry._verify_training_weight_init(config)


def test_metric_boost_uses_a_retained_canonical_baseline_config():
    repository_root = Path(__file__).resolve().parents[3]
    config = repository_root / "configs" / "stage_b" / "a3_e4_stageb.yaml"
    assert config.is_file()
    payload = config.read_text(encoding="utf-8")
    assert "training_weight_init_sha256:" in payload
    assert "metric_boost_" not in payload
    assert "vit_source_core_sysu_no_sff_parameter_add_pa05.yaml" not in payload
