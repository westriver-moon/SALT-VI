"""Versioned Qwen imagination plugins for SALT-VI."""

__version__ = "0.6.0"

from .api import ImaginationPlugin, ImaginationRequest, ImaginationResult
from .registry import available_plugins, load_plugin

__all__ = [
    "ImaginationPlugin",
    "ImaginationRequest",
    "ImaginationResult",
    "available_plugins",
    "load_plugin",
    "__version__",
]
