from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if not isinstance(value, str):
        return value
    return os.path.expandvars(value)


def _unresolved(value: Any) -> list[str]:
    if isinstance(value, dict):
        return [name for item in value.values() for name in _unresolved(item)]
    if isinstance(value, list):
        return [name for item in value for name in _unresolved(item)]
    return _ENV_PATTERN.findall(value) if isinstance(value, str) else []


@dataclass(frozen=True)
class Asset:
    path: Path
    sha256: str


@dataclass
class RegionalConfig:
    schema_version: int
    dataset_root: Path
    output_root: Path
    plugin_version: str = "qri-v1"
    modalities: tuple[str, ...] = ("rgb", "ir")
    source_size_hw: tuple[int, int] = (256, 128)
    output_size_hw: tuple[int, int] = (512, 256)
    tta_count: int = 12
    qwen_sample_count: int = 16
    selected_region_count: int = 3
    max_worlds: int = 5
    proposal_rounds: int = 1
    coverage_sampling: bool = False
    ensure_editing_world_per_region: bool = False
    roi_board_size_px: int = 384
    mask_dilation_px: int = 4
    mask_feather_px: float = 3.0
    calibration_alpha: float = 1.0
    calibration_beta: float = 8.0
    calibration_gamma: float = 4.0
    calibration_delta: float = 16.0
    seed: int = 20260819
    assets: dict[str, Asset] = field(default_factory=dict)
    roi: dict[str, Any] = field(default_factory=dict)
    qwen: dict[str, Any] = field(default_factory=dict)
    pasd: dict[str, Any] = field(default_factory=dict)
    swinir: dict[str, Any] = field(default_factory=dict)
    identity: dict[str, Any] = field(default_factory=dict)

    @property
    def plugin_id(self) -> str:
        return {
            "qri-v1": "qwen-regional-imagination-v1",
            "qri-v2": "qwen-regional-imagination-v2",
        }[self.plugin_version]

    def validate(self) -> "RegionalConfig":
        expected_schema = {"qri-v1": 1, "qri-v2": 2}
        if self.plugin_version not in expected_schema:
            raise ValueError("plugin_version must be qri-v1 or qri-v2")
        if int(self.schema_version) != expected_schema[self.plugin_version]:
            raise ValueError(
                f"{self.plugin_version} config schema_version must be "
                f"{expected_schema[self.plugin_version]}"
            )
        if set(self.modalities) != {"rgb", "ir"}:
            raise ValueError("formal QRI SYSU runs require both rgb and ir")
        if tuple(self.source_size_hw) != (256, 128):
            raise ValueError("QRI source_size_hw must be [256, 128]")
        if tuple(self.output_size_hw) != (512, 256):
            raise ValueError("QRI output_size_hw must be [512, 256]")
        expected = {
            "tta_count": 12,
            "qwen_sample_count": 16,
            "selected_region_count": 3,
            "max_worlds": 5,
            "mask_dilation_px": 4,
        }
        for name, required in expected.items():
            if int(getattr(self, name)) != required:
                raise ValueError(f"QRI {name} must be {required}")
        if float(self.mask_feather_px) != 3.0:
            raise ValueError("QRI mask_feather_px must be 3.0")
        coefficients = (
            self.calibration_alpha,
            self.calibration_beta,
            self.calibration_gamma,
            self.calibration_delta,
        )
        if tuple(float(value) for value in coefficients) != (1.0, 8.0, 4.0, 16.0):
            raise ValueError("QRI calibration coefficients must be [1, 8, 4, 16]")
        if self.plugin_version == "qri-v1":
            if int(self.proposal_rounds) != 1:
                raise ValueError("QRI-v1 proposal_rounds must be 1")
            if self.coverage_sampling or self.ensure_editing_world_per_region:
                raise ValueError("QRI-v1 does not enable V2 coverage selection")
            if int(self.roi_board_size_px) != 384:
                raise ValueError("QRI-v1 roi_board_size_px must be 384")
        else:
            if int(self.proposal_rounds) != 3:
                raise ValueError("QRI-v2 proposal_rounds must be 3")
            if not self.coverage_sampling or not self.ensure_editing_world_per_region:
                raise ValueError(
                    "QRI-v2 requires coverage sampling and editing-world coverage"
                )
            if int(self.roi_board_size_px) != 512:
                raise ValueError("QRI-v2 roi_board_size_px must be 512")
            imaginative_pasd_defaults = {
                "realization": "roi-direct-rewrite-then-soft-mask-composite",
                "roi_context_scale": 1.75,
                "guidance_scale": 7.0,
                "conditioning_scale": 0.75,
                "localized_added_prompt": (
                    "high-detail localized semantic realization, make the requested detail "
                    "crisp and recognizable at surveillance scale, preserve the same person "
                    "and surrounding observed structure"
                ),
                "localized_negative_prompt": (
                    "different person, changed identity, changed pose, changed body "
                    "proportions, different clothing, changes outside the requested region, "
                    "unrequested accessories outside the requested region, duplicated object, "
                    "distorted anatomy, painting, cartoon, artificial texture, blurry, noise, "
                    "raster lines, over-smoothed"
                ),
            }
            for name, value in imaginative_pasd_defaults.items():
                self.pasd.setdefault(name, value)
            if self.pasd["realization"] != imaginative_pasd_defaults["realization"]:
                raise ValueError(
                    "QRI-v2 requires localized ROI PASD realization; whole-canvas PASD "
                    "dilutes small semantic hypotheses"
                )
            if not 1.0 <= float(self.pasd["roi_context_scale"]) <= 3.0:
                raise ValueError("QRI-v2 PASD roi_context_scale must be in [1, 3]")
            if not 1.0 <= float(self.pasd["guidance_scale"]) <= 12.0:
                raise ValueError("QRI-v2 PASD guidance_scale must be in [1, 12]")
            if not 0.1 <= float(self.pasd["conditioning_scale"]) <= 1.0:
                raise ValueError("QRI-v2 PASD conditioning_scale must be in [0.1, 1]")
            negative = str(self.pasd["localized_negative_prompt"]).lower()
            contradictory = {"new accessories", "altered face"}
            if any(term in negative for term in contradictory):
                raise ValueError(
                    "QRI-v2 localized negative prompt cannot suppress the requested "
                    "accessory or facial-detail hypothesis"
                )
        return self


def load_regional_config(
    path: str | Path, *, require_resolved: bool = True
) -> RegionalConfig:
    config_path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    expanded = _expand(raw)
    missing_env = sorted(set(_unresolved(expanded)))
    if require_resolved and missing_env:
        raise ValueError(f"unresolved QRI environment variables: {missing_env}")
    base = config_path.parent

    def resolve_path(value: str | Path) -> Path:
        candidate = Path(value).expanduser()
        return (
            candidate.resolve()
            if candidate.is_absolute()
            else (base / candidate).resolve()
        )

    assets = {
        str(name): Asset(resolve_path(spec["path"]), str(spec["sha256"]).lower())
        for name, spec in dict(expanded.pop("assets", {})).items()
    }
    expanded["dataset_root"] = resolve_path(expanded["dataset_root"])
    expanded["output_root"] = resolve_path(expanded["output_root"])
    expanded["modalities"] = tuple(
        str(item).lower() for item in expanded.get("modalities", ("rgb", "ir"))
    )
    expanded["source_size_hw"] = tuple(expanded.get("source_size_hw", (256, 128)))
    expanded["output_size_hw"] = tuple(expanded.get("output_size_hw", (512, 256)))
    expanded["assets"] = assets
    return RegionalConfig(**expanded).validate()
