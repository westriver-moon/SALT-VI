"""Shared adapter for the versioned regional QRI implementations."""

from __future__ import annotations

from pathlib import Path
from typing import Union

from ..api import ImaginationRequest, ImaginationResult
from ..regional import cli
from ..regional.config import RegionalConfig, load_regional_config


class RegionalPlugin:
    """Expose the existing regional pipeline through the stable SALT contract."""

    plugin_id = "qwen-regional-imagination"
    expected_plugin_version = ""

    def __init__(self, config_path: Union[str, Path]):
        self.config_path = Path(config_path).expanduser().resolve()
        self.config: RegionalConfig = load_regional_config(self.config_path)
        if self.config.plugin_version != self.expected_plugin_version:
            raise ValueError(
                f"{self.plugin_id} requires plugin_version="
                f"{self.expected_plugin_version}, got {self.config.plugin_version}"
            )

    def execute(self, request: ImaginationRequest) -> ImaginationResult:
        if request.config_path.expanduser().resolve() != self.config_path:
            raise ValueError("request config_path does not match the loaded plugin config")

        if request.action == "preflight":
            payload = cli.preflight(self.config, check_server=request.check_server)
        elif request.action == "run":
            if request.limit is not None and request.limit < 1:
                raise ValueError("limit must be positive")
            if request.device is not None:
                self.config.roi["device"] = request.device
                self.config.pasd["device"] = request.device
                self.config.identity["device"] = request.device
            if request.category_stats is not None:
                stats_path = request.category_stats.expanduser().resolve()
                if not stats_path.is_file():
                    raise FileNotFoundError(stats_path)
                self.config.roi["category_stats_path"] = str(stats_path)
            payload = cli.run(
                self.config,
                limit=request.limit,
                fail_fast=request.fail_fast,
                split=request.split,
            )
        elif request.action == "serve":
            payload = cli.serve(self.config, execute=request.execute)
        else:
            raise ValueError(
                f"unsupported imagination action {request.action!r}; "
                "expected preflight, run, or serve"
            )
        return ImaginationResult(
            plugin_id=self.config.plugin_id,
            action=request.action,
            payload=payload,
        )
