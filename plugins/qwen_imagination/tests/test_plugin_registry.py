from pathlib import Path

import pytest

from qwen_imagination import available_plugins, load_plugin
from qwen_imagination.api import ImaginationRequest


def test_registry_exposes_versioned_plugins():
    assert available_plugins() == ("qri-v1", "qri-v2")


def test_unknown_plugin_is_rejected():
    with pytest.raises(ValueError, match="unknown imagination plugin"):
        load_plugin("qri-v9")


def test_request_is_version_neutral():
    request = ImaginationRequest(config_path=Path("config.yaml"))
    assert request.action == "run"
    assert request.limit is None
