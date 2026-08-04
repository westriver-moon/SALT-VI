from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


class Config(dict):
    """Small dict wrapper with attribute access for YAML configs."""

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        if isinstance(value, dict) and not isinstance(value, Config):
            value = Config(value)
            self[name] = value
        return value


def _to_config(value: Any) -> Any:
    if isinstance(value, dict):
        return Config({k: _to_config(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_to_config(v) for v in value]
    return value


def _deep_merge(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load_config_dict(path: Path, stack=()) -> dict:
    path = path.resolve()
    if path in stack:
        chain = " -> ".join(str(item) for item in (*stack, path))
        raise ValueError(f"Circular config inheritance: {chain}")
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    parent = raw.pop("extends", None)
    if parent is None:
        return raw
    parent_path = Path(parent)
    if not parent_path.is_absolute():
        parent_path = path.parent / parent_path
    return _deep_merge(_load_config_dict(parent_path, (*stack, path)), raw)


def load_config(path: str | Path) -> Config:
    return _to_config(_load_config_dict(Path(path)))


def _coerce_value(text: str) -> Any:
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    return text


def merge_overrides(config: Config, overrides: dict[str, Any]) -> Config:
    merged = copy.deepcopy(config)
    for dotted_key, value in overrides.items():
        cursor = merged
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            if part not in cursor or not isinstance(cursor[part], dict):
                cursor[part] = Config()
            cursor = cursor[part]
        cursor[parts[-1]] = _coerce_value(value) if isinstance(value, str) else value
    return _to_config(merged)


def to_plain_dict(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: to_plain_dict(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_plain_dict(v) for v in value]
    return value
