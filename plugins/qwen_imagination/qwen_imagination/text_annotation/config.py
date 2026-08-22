from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item) for item in value]
    return os.path.expandvars(value) if isinstance(value, str) else value


def _unresolved(value: Any) -> list[str]:
    if isinstance(value, dict):
        return [name for item in value.values() for name in _unresolved(item)]
    if isinstance(value, list):
        return [name for item in value for name in _unresolved(item)]
    return _ENV_PATTERN.findall(value) if isinstance(value, str) else []


@dataclass
class TextAnnotationConfig:
    schema_version: int
    annotation_version: str
    dataset_root: Path
    output_root: Path
    modalities: tuple[str, ...] = ("rgb", "ir")
    source_size_hw: tuple[int, int] = (256, 128)
    output_size_hw: tuple[int, int] = (512, 256)
    selected_region_count: int = 3
    max_selected_region_count: int = 6
    roi_selection_threshold: float = 0.6
    roi_category_priority_boosts: dict[str, float] = field(default_factory=dict)
    roi_board_size_px: int = 512
    world_sample_count: int = 64
    max_worlds: int = 8
    probability_mode: str = "vlm_reported"
    probability_spec: str | None = None
    seed: int = 20260822
    strategy: str = "track_anchor"
    exact_selection_mode: str = "full_tta"
    precomputed_swinir_root: Path | None = None
    assets: dict[str, Path] = field(default_factory=dict)
    roi: dict[str, Any] = field(default_factory=dict)
    swinir: dict[str, Any] = field(default_factory=dict)
    qwen: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> "TextAnnotationConfig":
        if int(self.schema_version) != 1:
            raise ValueError("text annotation schema_version must be 1")
        if not str(self.annotation_version).strip():
            raise ValueError("annotation_version must be non-empty")
        if not self.modalities or not set(self.modalities).issubset({"rgb", "ir"}):
            raise ValueError("modalities must be a non-empty subset of rgb and ir")
        if tuple(self.source_size_hw) != (256, 128):
            raise ValueError("SYSU text annotation source_size_hw must be [256, 128]")
        if tuple(self.output_size_hw) != (512, 256):
            raise ValueError("SYSU text annotation output_size_hw must be [512, 256]")
        if int(self.selected_region_count) < 1:
            raise ValueError("selected_region_count must be positive")
        if not int(self.selected_region_count) <= int(self.max_selected_region_count) <= 13:
            raise ValueError(
                "max_selected_region_count must be within [selected_region_count, 13]"
            )
        if not 0.0 <= float(self.roi_selection_threshold) <= 1.0:
            raise ValueError("roi_selection_threshold must be within [0, 1]")
        for category, boost in self.roi_category_priority_boosts.items():
            if not str(category).strip():
                raise ValueError("roi_category_priority_boosts keys must be non-empty")
            if not 0.0 <= float(boost) <= 1.0:
                raise ValueError(
                    "roi_category_priority_boosts values must be within [0, 1]"
                )
        if int(self.roi_board_size_px) < 256 or int(self.roi_board_size_px) % 2:
            raise ValueError("roi_board_size_px must be an even integer >= 256")
        if int(self.world_sample_count) < 1:
            raise ValueError("world_sample_count must be positive")
        if not 1 <= int(self.max_worlds) <= int(self.world_sample_count):
            raise ValueError("max_worlds must be within [1, world_sample_count]")
        if self.strategy not in {"exact", "track_anchor"}:
            raise ValueError("strategy must be exact or track_anchor")
        if self.probability_mode not in {"vlm_reported", "deferred_empirical"}:
            raise ValueError(
                "probability_mode must be vlm_reported or deferred_empirical"
            )
        if self.probability_mode == "deferred_empirical" and not self.probability_spec:
            raise ValueError(
                "deferred_empirical probability mode requires probability_spec"
            )
        if self.exact_selection_mode not in {"full_tta", "fast_blur_eye_guard"}:
            raise ValueError(
                "exact_selection_mode must be full_tta or fast_blur_eye_guard"
            )
        if self.strategy == "track_anchor" and self.precomputed_swinir_root is None:
            raise ValueError("track_anchor strategy requires precomputed_swinir_root")
        required_assets = {
            "qwen_model",
            "qwen_mmproj",
            "swinir_model",
            "yolo_pose",
            "schp_lip",
            "sam_vit_b",
        }
        missing = sorted(required_assets.difference(self.assets))
        if missing:
            raise ValueError(f"text annotation config omits assets: {missing}")
        for name in ("schp_root", "sam_root", "device"):
            if name not in self.roi:
                raise ValueError(f"text annotation roi config omits {name}")
        if "root" not in self.swinir:
            raise ValueError("text annotation swinir config omits root")
        for name in ("endpoint", "model_id"):
            if name not in self.qwen:
                raise ValueError(f"text annotation qwen config omits {name}")
        return self

    def run_signature(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "annotation_version": self.annotation_version,
            "modalities": list(self.modalities),
            "source_size_hw": list(self.source_size_hw),
            "output_size_hw": list(self.output_size_hw),
            "selected_region_count": int(self.selected_region_count),
            "max_selected_region_count": int(self.max_selected_region_count),
            "roi_selection_threshold": float(self.roi_selection_threshold),
            "roi_category_priority_boosts": {
                str(category): float(boost)
                for category, boost in sorted(
                    self.roi_category_priority_boosts.items()
                )
            },
            "roi_board_size_px": int(self.roi_board_size_px),
            "world_sample_count": int(self.world_sample_count),
            "max_worlds": int(self.max_worlds),
            "probability_mode": self.probability_mode,
            "probability_spec": self.probability_spec,
            "seed": int(self.seed),
            "strategy": self.strategy,
            "exact_selection_mode": self.exact_selection_mode,
            "precomputed_swinir_root": (
                str(self.precomputed_swinir_root)
                if self.precomputed_swinir_root is not None
                else None
            ),
            "qwen_model_id": str(self.qwen["model_id"]),
            "qwen_thinking": bool(self.qwen.get("thinking_mode", False)),
            "qwen_reasoning_effort": str(self.qwen.get("reasoning_effort", "none")),
            "qwen_temperature": float(self.qwen.get("temperature", 0.35)),
            "qwen_max_tokens": int(self.qwen.get("max_tokens", 2048)),
            "qwen_response_profile": str(
                self.qwen.get("response_profile", "detailed_v1")
            ),
            "qwen_prompt_version": str(self.qwen.get("prompt_version", "v1")),
        }

    def provenance(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["dataset_root"] = str(self.dataset_root)
        payload["output_root"] = str(self.output_root)
        payload["precomputed_swinir_root"] = (
            str(self.precomputed_swinir_root)
            if self.precomputed_swinir_root is not None
            else None
        )
        payload["assets"] = {name: str(path) for name, path in self.assets.items()}
        return payload


def load_text_annotation_config(
    path: str | Path, *, require_resolved: bool = True
) -> TextAnnotationConfig:
    config_path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    expanded = _expand(raw)
    missing_env = sorted(set(_unresolved(expanded)))
    if require_resolved and missing_env:
        raise ValueError(
            f"unresolved text annotation environment variables: {missing_env}"
        )
    base = config_path.parent

    def resolve_path(value: str | Path) -> Path:
        candidate = Path(value).expanduser()
        return (
            candidate.resolve()
            if candidate.is_absolute()
            else (base / candidate).resolve()
        )

    expanded["dataset_root"] = resolve_path(expanded["dataset_root"])
    expanded["output_root"] = resolve_path(expanded["output_root"])
    if expanded.get("precomputed_swinir_root") is not None:
        expanded["precomputed_swinir_root"] = resolve_path(
            expanded["precomputed_swinir_root"]
        )
    expanded["modalities"] = tuple(
        str(item).lower() for item in expanded.get("modalities", ("rgb", "ir"))
    )
    expanded["source_size_hw"] = tuple(expanded.get("source_size_hw", (256, 128)))
    expanded["output_size_hw"] = tuple(expanded.get("output_size_hw", (512, 256)))
    expanded["assets"] = {
        str(name): resolve_path(value)
        for name, value in dict(expanded.get("assets", {})).items()
    }
    for group in ("roi", "swinir"):
        values = dict(expanded.get(group, {}))
        for name in tuple(values):
            if values[name] is not None and (
                name.endswith("_root") or name == "root" or name.endswith("_path")
            ):
                values[name] = str(resolve_path(values[name]))
        expanded[group] = values
    qwen = dict(expanded.get("qwen", {}))
    for name in ("server_binary", "llama_root"):
        if qwen.get(name):
            qwen[name] = str(resolve_path(qwen[name]))
    expanded["qwen"] = qwen
    return TextAnnotationConfig(**expanded).validate()
