"""QRI-v1 adapter."""

from .base import RegionalPlugin


class QRIv1Plugin(RegionalPlugin):
    plugin_id = "qwen-regional-imagination-v1"
    expected_plugin_version = "qri-v1"
