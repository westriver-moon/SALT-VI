"""Lazy registry for versioned Qwen imagination plugins."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Optional, Type, Union

from .api import ImaginationPlugin


_PLUGIN_SPECS = {
    "qri-v1": ("qwen_imagination.versions.qri_v1", "QRIv1Plugin"),
    "qri-v2": ("qwen_imagination.versions.qri_v2", "QRIv2Plugin"),
}
_DEFAULT_CONFIGS = {
    "qri-v1": "qri_v1_sysu.yaml",
    "qri-v2": "qri_v2_imaginative_sysu.yaml",
}


def available_plugins() -> tuple[str, ...]:
    return tuple(sorted(_PLUGIN_SPECS))


def _plugin_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_plugin(
    plugin_id: str,
    config_path: Optional[Union[str, Path]] = None,
) -> ImaginationPlugin:
    normalized = str(plugin_id).strip().lower()
    if normalized not in _PLUGIN_SPECS:
        raise ValueError(
            f"unknown imagination plugin {plugin_id!r}; "
            f"available: {', '.join(available_plugins())}"
        )
    module_name, class_name = _PLUGIN_SPECS[normalized]
    module = importlib.import_module(module_name)
    plugin_class: Type[ImaginationPlugin] = getattr(module, class_name)
    path = (
        Path(config_path).expanduser().resolve()
        if config_path is not None
        else _plugin_root() / "configs" / _DEFAULT_CONFIGS[normalized]
    )
    return plugin_class(path)
