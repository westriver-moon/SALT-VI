from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml


SCHEMA_VERSION = 1
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
ENCODERS = {"image", "text", "fusion", "protocol_query", "protocol_gallery"}
SPLITS = {"query", "gallery", "train_rgb", "train_ir"}
STAGES = {"pre_bn", "post_bn"}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _resolve_path(value: str, config_dir: Path, project_root: Path) -> str:
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return str(candidate.resolve())
    choices = (project_root / candidate, config_dir / candidate)
    existing = next((item for item in choices if item.exists()), choices[0])
    return str(existing.resolve())


def _validate_name(value: Any, field: str) -> str:
    text = str(value or "")
    _require(bool(SAFE_NAME.fullmatch(text)), f"{field} must be filesystem-safe: {text!r}")
    return text


def _as_list(value: Any, field: str) -> List[Any]:
    _require(isinstance(value, list) and value, f"{field} must be a non-empty list")
    return value


def load_analysis_config(path: str) -> Dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Analysis config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    _require(isinstance(payload, dict), "Analysis config root must be a mapping")
    config = copy.deepcopy(payload)
    _require(int(config.get("schema_version", -1)) == SCHEMA_VERSION, "Unsupported schema_version")

    project_root = Path(__file__).resolve().parents[3]
    config_dir = config_path.parent
    config["run_id"] = _validate_name(config.get("run_id"), "run_id")
    config["config_path"] = str(config_path)
    config["project_root"] = str(project_root)
    config["output_root"] = _resolve_path(
        str(config.get("output_root", "feature_analysis/artifacts")), config_dir, project_root
    )
    config["overwrite"] = bool(config.get("overwrite", False))

    runtime = dict(config.get("runtime") or {})
    runtime.setdefault("cuda_visible_devices", "")
    runtime.setdefault("device", "cpu")
    runtime.setdefault("seed", 0)
    runtime.setdefault("batch_size", 16)
    runtime.setdefault("num_workers", 4)
    runtime.setdefault("amp", True)
    _require(int(runtime["batch_size"]) > 0, "runtime.batch_size must be positive")
    _require(int(runtime["num_workers"]) >= 0, "runtime.num_workers must be non-negative")
    config["runtime"] = runtime

    models = _as_list(config.get("models"), "models")
    model_ids = set()
    normalized_models = []
    for index, raw in enumerate(models):
        _require(isinstance(raw, dict), f"models[{index}] must be a mapping")
        item = dict(raw)
        item["id"] = _validate_name(item.get("id"), f"models[{index}].id")
        _require(item["id"] not in model_ids, f"Duplicate model id: {item['id']}")
        model_ids.add(item["id"])
        for field in ("config", "checkpoint"):
            _require(item.get(field), f"models[{index}].{field} is required")
            item[field] = _resolve_path(str(item[field]), config_dir, project_root)
            _require(Path(item[field]).is_file(), f"models[{index}].{field} not found: {item[field]}")
        item["overrides"] = dict(item.get("overrides") or {})
        normalized_models.append(item)
    config["models"] = normalized_models

    splits = dict(config.get("splits") or {})
    _require(any(bool(splits.get(name)) for name in SPLITS), "At least one split must be enabled")
    gallery = splits.get("gallery")
    if gallery:
        gallery = {} if gallery is True else dict(gallery)
        trials = gallery.get("trials", "all")
        if trials != "all":
            _require(isinstance(trials, list) and trials, "gallery.trials must be 'all' or a list")
            trials = [int(item) for item in trials]
            _require(all(0 <= item < 10 for item in trials), "gallery trial indices must be in [0, 9]")
        gallery["trials"] = trials
        splits["gallery"] = gallery
    for name in ("train_rgb", "train_ir"):
        if splits.get(name):
            entry = {} if splits[name] is True else dict(splits[name])
            views = entry.get("views", [0])
            if views != "all":
                _require(isinstance(views, list) and views, f"{name}.views must be 'all' or a list")
                views = [int(item) for item in views]
                _require(all(item >= 0 for item in views), f"{name}.views must be non-negative")
            entry["views"] = views
            splits[name] = entry
    config["splits"] = splits

    representations = _as_list(config.get("representations"), "representations")
    names = set()
    normalized_representations = []
    for index, raw in enumerate(representations):
        _require(isinstance(raw, dict), f"representations[{index}] must be a mapping")
        item = dict(raw)
        item["name"] = _validate_name(item.get("name"), f"representations[{index}].name")
        _require(item["name"] not in names, f"Duplicate representation name: {item['name']}")
        names.add(item["name"])
        item["encoder"] = str(item.get("encoder", ""))
        _require(item["encoder"] in ENCODERS, f"Unsupported encoder: {item['encoder']}")
        item["splits"] = [str(value) for value in _as_list(item.get("splits"), f"representations[{index}].splits")]
        _require(set(item["splits"]) <= SPLITS, f"Unsupported split in representation {item['name']}")
        selected_models = item.get("models")
        if selected_models is not None:
            selected_models = [str(value) for value in _as_list(selected_models, f"representations[{index}].models")]
            _require(set(selected_models) <= model_ids, f"Unknown model id in representation {item['name']}")
        item["models"] = selected_models
        item["stage"] = str(item.get("stage", "post_bn"))
        _require(item["stage"] in STAGES, f"Unsupported feature stage: {item['stage']}")
        if item["encoder"] in {"image", "fusion"}:
            item["modality"] = str(item.get("modality", "")).lower()
            _require(item["modality"] in {"rgb", "ir"}, f"{item['name']} requires modality rgb or ir")
        item["normalize"] = bool(item.get("normalize", False))
        item["use_backup"] = bool(item.get("use_backup", False))
        normalized_representations.append(item)
    config["representations"] = normalized_representations

    analysis = dict(config.get("analysis") or {})
    analysis.setdefault("max_pair_samples", 20000)
    analysis.setdefault("max_svd_samples", 5000)
    analysis.setdefault("max_plot_samples", 4000)
    analysis.setdefault("auto_checkpoint_comparisons", True)
    analysis.setdefault("make_figures", True)
    config["analysis"] = analysis

    comparisons = list(config.get("comparisons") or [])
    for index, comparison in enumerate(comparisons):
        _require(isinstance(comparison, dict), f"comparisons[{index}] must be a mapping")
        _validate_name(comparison.get("id"), f"comparisons[{index}].id")
        _require(comparison.get("left") and comparison.get("right"), f"comparisons[{index}] needs left/right")
    config["comparisons"] = comparisons
    return config


def representation_specs(
    config: Dict[str, Any], split: str, model_id: str = None
) -> List[Dict[str, Any]]:
    return [
        item
        for item in config["representations"]
        if split in item["splits"]
        and (model_id is None or item["models"] is None or model_id in item["models"])
    ]


def validate_enabled_representations(config: Dict[str, Any]) -> None:
    for model in config["models"]:
        for split in SPLITS:
            if config["splits"].get(split) and not representation_specs(config, split, model["id"]):
                raise ValueError(
                    f"Split {split!r} is enabled but model {model['id']!r} has no representation"
                )
