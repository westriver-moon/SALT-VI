"""Thin SALT-VI bridge to the centralized Qwen imagination plugin area."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Optional, Union


def default_plugin_root() -> Path:
    configured = os.environ.get("SALT_QWEN_IMAGINATION_ROOT")
    candidates = []
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.extend(
        [
            Path.cwd() / "plugins" / "qwen_imagination",
            Path(__file__).resolve().parents[2]
            / "plugins"
            / "qwen_imagination",
        ]
    )
    for candidate in candidates:
        if (candidate / "qwen_imagination" / "registry.py").is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "centralized Qwen imagination plugin area was not found; "
        "set SALT_QWEN_IMAGINATION_ROOT"
    )


def load_imagination_plugin(
    plugin_id: str,
    config_path: Optional[Union[str, Path]] = None,
    *,
    plugin_root: Optional[Union[str, Path]] = None,
):
    root = (
        Path(plugin_root).expanduser().resolve()
        if plugin_root is not None
        else default_plugin_root()
    )
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    registry = importlib.import_module("qwen_imagination.registry")
    return registry.load_plugin(plugin_id, config_path)
