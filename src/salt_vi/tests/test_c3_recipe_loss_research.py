import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from salt_vi.training.recipes import pmt_metric_auxiliary_scale
from salt_vi.utils.loss import PMTIdentityRelationLoss
from salt_vi.diagnostics.c3_recipe import (
    _numeric_deltas,
    _optional_summary,
    _within_identity_dispersion,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MANIFEST = PROJECT_ROOT / "configs/pipelines/c3_recipe_loss_research_20260829.yaml"
SCRIPT = PROJECT_ROOT / "scripts/experiments/run_c3_recipe_loss_research.py"


def _load_study_script():
    spec = importlib.util.spec_from_file_location("c3_recipe_loss_study", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_metric_auxiliary_schedule_is_decoupled_and_ramped():
    args = SimpleNamespace(
        pmt_progressive_epoch=4,
        pmt_metric_start_epoch=6,
        pmt_metric_ramp_epochs=3,
    )
    assert pmt_metric_auxiliary_scale(args, 5) == 0.0
    assert pmt_metric_auxiliary_scale(args, 6) == pytest.approx(1.0 / 3.0)
    assert pmt_metric_auxiliary_scale(args, 7) == pytest.approx(2.0 / 3.0)
    assert pmt_metric_auxiliary_scale(args, 8) == 1.0


def test_best_evaluated_epoch_is_selected_instead_of_final_epoch(tmp_path):
    study = _load_study_script()
    events_path = tmp_path / "events.jsonl"
    protocol = {
        "gallery_trials": 10,
        "trial_ids": list(range(10)),
    }
    events = [
        {
            "event_type": "eval_epoch",
            "epoch": 19,
            "protocol": "official-sysu",
            "protocol_spec": protocol,
            "metrics": {"Rank-1": 0.71, "mAP": 0.68, "mINP": 0.55},
        },
        {
            "event_type": "eval_epoch",
            "epoch": 23,
            "protocol": "official-sysu",
            "protocol_spec": protocol,
            "metrics": {"Rank-1": 0.70, "mAP": 0.69, "mINP": 0.57},
        },
    ]
    events_path.write_text(
        "\n".join(json.dumps(event) for event in events) + "\n",
        encoding="utf-8",
    )
    selection = {
        "policy": "best-evaluated-epoch",
        "primary_metric": "Rank-1",
        "mode": "maximize",
        "tie_breakers": [],
        "final_tie_breaker": "latest_epoch",
    }

    result = study.best_evaluated_metrics(
        {"name": "trial", "events_path": str(events_path)},
        selection,
    )

    assert result["epoch"] == 19
    assert result["Rank-1"] == pytest.approx(0.71)
    assert result["evaluated_epochs"] == [19, 23]


def test_metric_auxiliary_schedule_defaults_to_original_switch():
    args = SimpleNamespace(
        pmt_progressive_epoch=6,
        pmt_metric_start_epoch=None,
        pmt_metric_ramp_epochs=0,
    )
    assert pmt_metric_auxiliary_scale(args, 5) == 0.0
    assert pmt_metric_auxiliary_scale(args, 6) == 1.0


def test_relation_loss_matches_modality_internal_geometry_not_absolute_centers():
    visible = torch.tensor(
        [[1.0, 0.0], [1.1, 0.0], [0.0, 1.0], [0.0, 1.1]],
        requires_grad=True,
    )
    rotation = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
    infrared = (visible.detach() @ rotation).requires_grad_(True)
    labels = torch.tensor([0, 0, 1, 1])
    loss = PMTIdentityRelationLoss()(visible, infrared, labels, labels)
    assert loss.item() == pytest.approx(0.0, abs=1e-7)
    loss.backward()
    assert visible.grad is not None
    assert infrared.grad is not None
    assert torch.isfinite(visible.grad).all()
    assert torch.isfinite(infrared.grad).all()


def test_relation_loss_detects_different_identity_geometry():
    visible = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [-1.0, 0.0], [-1.0, 0.0]]
    )
    infrared = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [0.8, 0.2], [0.8, 0.2], [0.0, 1.0], [0.0, 1.0]]
    )
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    assert PMTIdentityRelationLoss()(visible, infrared, labels, labels).item() > 0.01


def test_manifest_has_fixed_c3_input_and_only_recipe_overrides():
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 2
    fixed = manifest["fixed_config"]
    assert fixed["sysu_sr_data_root"].endswith("SYSU-MM01-swinir-x2-pmt256-v1")
    assert fixed["sysu_sr_modalities"] == ["rgb", "ir"]
    assert fixed["img_size"] == [512, 256]
    assert fixed["sampler_type"] == "identity_camera_diverse"
    allowed = set(manifest["allowed_overrides"])
    assert len(manifest["variants"]) == 13
    assert len(manifest["diagnostics"]) == 2
    assert manifest["scheduling"]["default_gpus"] == [2, 3]
    assert set(manifest["scheduling"]["job_order"]) == (
        set(manifest["variants"]) | set(manifest["diagnostics"])
    )
    for variant in manifest["variants"].values():
        assert set(variant["overrides"]) <= allowed
        assert not set(variant["overrides"]) & {
            "sysu_sr_data_root",
            "sysu_sr_modalities",
            "img_size",
            "sampler_type",
            "normalized_classifier",
            "batch_size",
            "num_pos",
            "seed",
        }


def test_manifest_diagnostics_answer_gradient_and_modality_questions():
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    diagnostic_types = {
        item["type"] for item in manifest["diagnostics"].values()
    }
    assert diagnostic_types == {
        "loss_gradient_conflict",
        "modality_information",
    }
    assert all(
        Path(item["checkpoint"]).is_absolute()
        for item in manifest["diagnostics"].values()
    )


def test_nested_numeric_deltas_preserve_diagnostic_structure():
    before = {"metric": 0.4, "probe": {"test_accuracy": 0.55}, "label": "a"}
    after = {"metric": 0.7, "probe": {"test_accuracy": 0.65}, "label": "b"}
    assert _numeric_deltas(before, after) == {
        "metric": pytest.approx(0.3),
        "probe": {"test_accuracy": pytest.approx(0.1)},
    }


def test_zero_gradient_is_recorded_as_undefined_cosine_not_failure():
    assert _optional_summary([], expected_count=2) == {
        "mean": None,
        "std": None,
        "minimum": None,
        "maximum": None,
        "count": 0,
        "undefined_count": 2,
    }


def test_within_identity_dispersion_is_l2_normalized_and_identity_balanced():
    features = torch.tensor(
        [[2.0, 0.0], [0.0, 3.0], [5.0, 0.0], [6.0, 0.0]]
    )
    labels = torch.tensor([0, 0, 1, 1])
    summary = _within_identity_dispersion(
        features, labels, torch.tensor([0, 1])
    )

    expected = 2 ** -0.5
    assert summary["normalization"] == "l2_per_sample"
    assert summary["per_identity_mean_distance"]["count"] == 2
    assert summary["per_identity_mean_distance"]["mean"] == pytest.approx(
        expected / 2
    )
    assert summary["sample_distance"]["count"] == 4
    assert summary["sample_distance"]["mean"] == pytest.approx(expected / 2)
    assert summary["sample_distance"]["maximum"] == pytest.approx(expected)
