from pathlib import Path

import pytest

from qwen_imagination import available_plugins, load_plugin
from qwen_imagination.api import ImaginationRequest


def test_registry_exposes_versioned_plugins():
    assert available_plugins() == ("qri-v1", "qri-v2", "qri-v6")


def test_unknown_plugin_is_rejected():
    with pytest.raises(ValueError, match="unknown imagination plugin"):
        load_plugin("qri-v9")


def test_request_is_version_neutral():
    request = ImaginationRequest(config_path=Path("config.yaml"))
    assert request.action == "run"
    assert request.limit is None


def test_qri_v6_registry_preflight_uses_injected_backend_contract(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("QRI_V6_OUTPUT_ROOT", str(tmp_path / "outputs"))
    plugin = load_plugin("qri-v6")
    result = plugin.execute(
        ImaginationRequest(config_path=plugin.config_path, action="preflight")
    )
    assert result.ok
    assert result.plugin_id == "qri-v6"
    assert result.payload["algorithm"]["world_sampling"] == "direct_joint_vlm_draws"
    assert result.payload["runtime_backends"] == "injected"
