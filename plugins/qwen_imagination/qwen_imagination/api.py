"""Stable interface between SALT-VI and versioned Qwen imagination plugins."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol


@dataclass(frozen=True)
class ImaginationRequest:
    """Version-neutral execution request."""

    config_path: Path
    action: str = "run"
    limit: Optional[int] = None
    fail_fast: bool = False
    split: str = "train"
    check_server: bool = True
    execute: bool = False
    device: Optional[str] = None
    category_stats: Optional[Path] = None


@dataclass(frozen=True)
class ImaginationResult:
    plugin_id: str
    action: str
    payload: Mapping[str, Any]

    @property
    def ok(self) -> bool:
        return bool(self.payload.get("valid", self.payload.get("complete", True)))


class ImaginationPlugin(Protocol):
    plugin_id: str

    def execute(self, request: ImaginationRequest) -> ImaginationResult:
        """Execute one contract action and return a provenance-bearing result."""
