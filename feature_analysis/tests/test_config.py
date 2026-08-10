from pathlib import Path

import pytest
import yaml

from salt_feature_analysis.config import load_analysis_config


def test_config_rejects_missing_checkpoint(tmp_path: Path):
    model_config = tmp_path / "model.yaml"
    model_config.write_text("dataset: sysu\n", encoding="utf-8")
    payload = {
        "schema_version": 1,
        "run_id": "unit-test",
        "models": [{"id": "a", "config": str(model_config), "checkpoint": str(tmp_path / "missing.pth")}],
        "splits": {"query": True},
        "representations": [{"name": "ir", "encoder": "image", "modality": "ir", "splits": ["query"]}],
    }
    path = tmp_path / "analysis.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="checkpoint not found"):
        load_analysis_config(str(path))


def test_config_normalizes_valid_schema(tmp_path: Path):
    model_config = tmp_path / "model.yaml"
    checkpoint = tmp_path / "model.pth"
    model_config.write_text("dataset: sysu\n", encoding="utf-8")
    checkpoint.write_bytes(b"test")
    payload = {
        "schema_version": 1,
        "run_id": "unit-test",
        "models": [{"id": "a", "config": str(model_config), "checkpoint": str(checkpoint)}],
        "splits": {"query": True},
        "representations": [{"name": "ir", "encoder": "image", "modality": "ir", "splits": ["query"]}],
    }
    path = tmp_path / "analysis.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    config = load_analysis_config(str(path))
    assert config["run_id"] == "unit-test"
    assert config["runtime"]["batch_size"] == 16
    assert config["representations"][0]["stage"] == "post_bn"


def test_representation_can_target_selected_models(tmp_path: Path):
    model_config = tmp_path / "model.yaml"
    checkpoint = tmp_path / "model.pth"
    model_config.write_text("dataset: sysu\n", encoding="utf-8")
    checkpoint.write_bytes(b"test")
    payload = {
        "schema_version": 1,
        "run_id": "targeted-models",
        "models": [
            {"id": name, "config": str(model_config), "checkpoint": str(checkpoint)}
            for name in ("old", "new")
        ],
        "splits": {"query": True},
        "representations": [
            {
                "name": "protocol",
                "encoder": "protocol_query",
                "splits": ["query"],
                "models": ["old", "new"],
            }
        ],
    }
    path = tmp_path / "analysis.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    config = load_analysis_config(str(path))
    assert config["representations"][0]["models"] == ["old", "new"]
