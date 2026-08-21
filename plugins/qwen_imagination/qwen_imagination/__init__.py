"""Versioned Qwen imagination plugins for SALT-VI."""

from .api import ImaginationPlugin, ImaginationRequest, ImaginationResult
from .registry import available_plugins, load_plugin

__all__ = [
    "ImaginationPlugin",
    "ImaginationRequest",
    "ImaginationResult",
    "available_plugins",
    "load_plugin",
]
