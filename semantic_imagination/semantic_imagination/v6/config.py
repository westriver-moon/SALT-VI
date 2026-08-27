from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .contracts import QRI_V6


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


@dataclass(frozen=True)
class V6Config:
    schema_version: int
    plugin_version: str
    output_root: Path
    candidate_world_count: int = 8
    similarity_threshold: float = 0.85
    request_max_attempts: int = 3
    seed: int = 20260824

    def validate(self) -> "V6Config":
        if self.schema_version != 6 or self.plugin_version != QRI_V6:
            raise ValueError(
                "qri-v6 requires schema_version=6 and plugin_version=qri-v6"
            )
        if self.candidate_world_count < 1:
            raise ValueError("candidate_world_count must be positive")
        if not -1.0 <= self.similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be within [-1, 1]")
        if self.request_max_attempts < 1:
            raise ValueError("request_max_attempts must be positive")
        return self

    def algorithm_contract(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "plugin_version": self.plugin_version,
            "candidate_world_count": self.candidate_world_count,
            "similarity_threshold": self.similarity_threshold,
            "deduplication_linkage": "complete",
            "world_generation": "single_vlm_candidate_batch",
            "world_weighting": "uniform_over_unique_worlds",
            "request_max_attempts": self.request_max_attempts,
            "seed": self.seed,
        }


def load_v6_config(path: str | Path) -> V6Config:
    config_path = Path(path).expanduser().resolve()
    raw = config_path.read_text(encoding="utf-8")
    expanded = os.path.expandvars(raw)
    unresolved = sorted(set(_ENV_PATTERN.findall(expanded)))
    if unresolved:
        raise ValueError(f"unresolved qri-v6 environment variables: {unresolved}")
    document = yaml.safe_load(expanded) or {}
    output_root = Path(document["output_root"]).expanduser()
    if not output_root.is_absolute():
        output_root = config_path.parent / output_root
    document["output_root"] = output_root.resolve()
    return V6Config(**document).validate()
