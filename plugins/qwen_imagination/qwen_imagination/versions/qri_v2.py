"""QRI-v2 adapter."""

from .base import RegionalPlugin


class QRIv2Plugin(RegionalPlugin):
    plugin_id = "qwen-regional-imagination-v2"
    expected_plugin_version = "qri-v2"
