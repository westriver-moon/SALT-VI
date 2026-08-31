"""Diagnostics for the fixed-input C3 PMT recipe study."""

from __future__ import annotations

from contextlib import contextmanager
import inspect
import math
import os
from pathlib import Path
import random
from statistics import mean, pstdev

import numpy as np
import torch
import torch.nn.functional as F

from salt_vi.config.validation import validate_runtime_config
from salt_vi.data.loader import Loader
from salt_vi.engine import build_model
from salt_vi.utils.utils import load_train_configs


LOSS_KEYS = ("id_loss", "tri_loss", "msel_loss", "dcl_loss")


@contextmanager
def _output_environment(output_root: Path):
    name = "SALT_C3_RECIPE_OUTPUT_ROOT"
    previous = os.environ.get(name)
    os.environ[name] = str(output_root)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_context(
    project_root: Path,
    resolved: dict,
    output_root: Path,
    diagnostic_name: str,
):
    base_config = project_root / resolved["base_config"]
    with _output_environment(output_root):
        config = load_train_configs(str(base_config))
    for key, value in resolved["overrides"].items():
        setattr(config, key, value)
    config.output_path = str(output_root / "diagnostics" / "runtime" / diagnostic_name)
    config.output_root = config.output_path
    config.metric_events_path = str(
        output_root / "diagnostics" / "runtime" / f"{diagnostic_name}.jsonl"
    )
    config.experiment_name = diagnostic_name
    config.metric_experiment_id = diagnostic_name.upper().replace("_", "-")
    config.num_workers = 0
    config.gpu_id = 0
    config.pid_num = 395
    validate_runtime_config(config)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loaders = Loader(config)
    model = build_model(config).to(device)
    model.configure_fixed_visual_data_parallel()
    return config, loaders, model, device


def _checkpoint_state(path: Path):
    kwargs = {"map_location": torch.device("cpu")}
    try:
        if "weights_only" in inspect.signature(torch.load).parameters:
            kwargs["weights_only"] = True
    except (TypeError, ValueError):
        pass
    payload = torch.load(path, **kwargs)
    if isinstance(payload, dict) and "model" in payload:
        return payload["model"]
    if isinstance(payload, dict) and "model_state_dict" in payload:
        return payload["model_state_dict"]
    if isinstance(payload, dict) and "state_dict" in payload:
        return payload["state_dict"]
    return payload


def _load_checkpoint(model, path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"diagnostic checkpoint is missing: {path}")
    result = model.load_state_dict(_checkpoint_state(path), strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            "diagnostic checkpoint is incompatible: "
            f"missing={list(result.missing_keys)}, "
            f"unexpected={list(result.unexpected_keys)}"
        )
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "modified_time_ns": int(stat.st_mtime_ns),
    }


def _autocast(device):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda", enabled=device.type == "cuda")
    return torch.cuda.amp.autocast(enabled=device.type == "cuda")


def _batch_to_device(batch: dict, device) -> dict:
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
        if torch.is_tensor(value)
    }


def _summary(values: list[float]) -> dict:
    if not values:
        raise ValueError("cannot summarize an empty diagnostic series")
    return {
        "mean": float(mean(values)),
        "std": float(pstdev(values)) if len(values) > 1 else 0.0,
        "minimum": float(min(values)),
        "maximum": float(max(values)),
        "count": len(values),
    }


def _optional_summary(values: list[float], expected_count: int) -> dict:
    if not values:
        return {
            "mean": None,
            "std": None,
            "minimum": None,
            "maximum": None,
            "count": 0,
            "undefined_count": int(expected_count),
        }
    result = _summary(values)
    result["undefined_count"] = int(expected_count - len(values))
    return result


def _gradient_dot(left, right) -> float:
    total = 0.0
    for first, second in zip(left, right):
        if first is not None and second is not None:
            total += float(torch.sum(first * second).item())
    return total


def _gradient_norm(gradients) -> float:
    squared = _gradient_dot(gradients, gradients)
    return math.sqrt(max(0.0, squared))


def _loss_gradient_conflict(
    spec: dict,
    resolved: dict,
    project_root: Path,
    output_root: Path,
) -> dict:
    seed = int(resolved["fixed_config"]["seed"])
    _seed_everything(seed)
    config, loaders, model, device = _build_context(
        project_root, resolved, output_root, spec["name"]
    )
    checkpoint = _load_checkpoint(model, Path(spec["checkpoint"]))
    model.set_train()
    if hasattr(loaders, "set_training_epoch"):
        loaders.set_training_epoch(int(spec["current_epoch"]))
    patterns = tuple(str(item) for item in spec["parameter_patterns"])
    selected = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and any(pattern in name for pattern in patterns)
    ]
    if not selected:
        raise RuntimeError(f"no trainable parameters matched {patterns}")
    names, parameters = zip(*selected)
    pair_values = {
        f"{left}__{right}": []
        for index, left in enumerate(LOSS_KEYS)
        for right in LOSS_KEYS[index + 1 :]
    }
    loss_values = {name: [] for name in LOSS_KEYS}
    gradient_norms = {name: [] for name in LOSS_KEYS}

    batches = int(spec["batches"])
    completed = 0
    for batch in loaders.get_train_loader():
        if completed >= batches:
            break
        batch = _batch_to_device(batch, device)
        model.zero_grad(set_to_none=True)
        with _autocast(device):
            result = model(
                batch,
                mode=None,
                current_epoch=int(spec["current_epoch"]),
            )
        missing = [name for name in LOSS_KEYS if name not in result]
        if missing:
            raise RuntimeError(f"diagnostic losses are missing: {missing}")
        gradients = {}
        for index, loss_name in enumerate(LOSS_KEYS):
            loss = result[loss_name]
            loss_values[loss_name].append(float(loss.detach().float().item()))
            current = torch.autograd.grad(
                loss,
                parameters,
                retain_graph=index + 1 < len(LOSS_KEYS),
                allow_unused=True,
            )
            current = tuple(
                None if item is None else item.detach().float().cpu()
                for item in current
            )
            norm = _gradient_norm(current)
            gradients[loss_name] = current
            gradient_norms[loss_name].append(norm)
        for index, left in enumerate(LOSS_KEYS):
            for right in LOSS_KEYS[index + 1 :]:
                denominator = (
                    gradient_norms[left][-1] * gradient_norms[right][-1]
                )
                if denominator > 0.0:
                    cosine = (
                        _gradient_dot(gradients[left], gradients[right])
                        / denominator
                    )
                    pair_values[f"{left}__{right}"].append(float(cosine))
        completed += 1
        del result, gradients, batch
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if completed != batches:
        raise RuntimeError(f"requested {batches} batches, observed {completed}")
    pairwise = {
        name: _optional_summary(values, completed)
        for name, values in pair_values.items()
    }
    return {
        "schema_version": 1,
        "diagnostic": spec["name"],
        "type": spec["type"],
        "purpose": spec["purpose"],
        "checkpoint": checkpoint,
        "data_config": {
            "base_config": resolved["base_config"],
            "sysu_sr_data_root": resolved["fixed_config"]["sysu_sr_data_root"],
            "img_size": resolved["fixed_config"]["img_size"],
            "sampler_type": resolved["fixed_config"]["sampler_type"],
        },
        "current_epoch": int(spec["current_epoch"]),
        "batch_count": completed,
        "parameter_scope": {
            "patterns": list(patterns),
            "tensor_count": len(parameters),
            "element_count": int(sum(item.numel() for item in parameters)),
            "first_parameter": names[0],
            "last_parameter": names[-1],
        },
        "loss_values": {
            name: _summary(values) for name, values in loss_values.items()
        },
        "gradient_norms": {
            name: _summary(values) for name, values in gradient_norms.items()
        },
        "zero_gradient_losses": sorted(
            name
            for name, values in gradient_norms.items()
            if any(value <= 0.0 for value in values)
        ),
        "pairwise_gradient_cosine": pairwise,
        "negative_mean_pairs": sorted(
            name
            for name, values in pairwise.items()
            if values["mean"] is not None and values["mean"] < 0.0
        ),
    }


def _cache_geometry_batches(loaders, count: int) -> list[dict]:
    cached = []
    if hasattr(loaders, "set_training_epoch"):
        loaders.set_training_epoch(23)
    keys = ("img_rgb_ori", "img_ir", "target_rgb", "target_ir")
    for batch in loaders.get_train_loader():
        cached.append(
            {
                key: batch[key].detach().cpu().clone()
                for key in keys
            }
        )
        if len(cached) >= count:
            break
    if len(cached) != count:
        raise RuntimeError(f"requested {count} geometry batches, observed {len(cached)}")
    return cached


def _extract_geometry_features(model, batches: list[dict], device) -> dict:
    model.set_eval()
    visible_features = []
    ir_features = []
    visible_labels = []
    ir_labels = []
    with torch.no_grad():
        for batch in batches:
            visible = batch["img_rgb_ori"].to(device, non_blocking=True)
            infrared = batch["img_ir"].to(device, non_blocking=True)
            images = torch.cat((visible, infrared), dim=0)
            with _autocast(device):
                encoded = model.base_model.encode_image(images, None)
                size = infrared.size(0)
                visible_embedding = model._get_visual_embedding(
                    model._slice_visual_output(encoded, 0, size)
                )
                infrared_embedding = model._get_visual_embedding(
                    model._slice_visual_output(encoded, size, None)
                )
            visible_features.append(visible_embedding.detach().float().cpu())
            ir_features.append(infrared_embedding.detach().float().cpu())
            visible_labels.append(batch["target_rgb"].long())
            ir_labels.append(batch["target_ir"].long())
    return {
        "visible_features": torch.cat(visible_features),
        "ir_features": torch.cat(ir_features),
        "visible_labels": torch.cat(visible_labels),
        "ir_labels": torch.cat(ir_labels),
    }


def _mask_for_identities(labels: torch.Tensor, identities: torch.Tensor):
    mask = torch.zeros(labels.shape, dtype=torch.bool)
    for identity in identities:
        mask |= labels == identity
    return mask


def _linear_probe_accuracy(
    features: torch.Tensor,
    modalities: torch.Tensor,
    identities: torch.Tensor,
    *,
    steps: int,
    train_fraction: float,
    seed: int,
) -> dict:
    unique = torch.unique(identities, sorted=True)
    if unique.numel() < 4:
        raise RuntimeError("modality probe requires at least four identities")
    generator = torch.Generator().manual_seed(seed)
    shuffled = unique[torch.randperm(unique.numel(), generator=generator)]
    train_count = max(2, min(unique.numel() - 1, round(unique.numel() * train_fraction)))
    train_ids = shuffled[:train_count]
    test_ids = shuffled[train_count:]
    train_mask = _mask_for_identities(identities, train_ids)
    test_mask = _mask_for_identities(identities, test_ids)
    train_x = features[train_mask].float()
    test_x = features[test_mask].float()
    train_y = modalities[train_mask].long()
    test_y = modalities[test_mask].long()
    location = train_x.mean(0, keepdim=True)
    scale = train_x.std(0, keepdim=True).clamp_min(1e-6)
    train_x = (train_x - location) / scale
    test_x = (test_x - location) / scale
    torch.manual_seed(seed)
    probe = torch.nn.Linear(train_x.size(1), 2)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=0.03, weight_decay=1e-3)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(probe(train_x), train_y)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        train_accuracy = (probe(train_x).argmax(1) == train_y).float().mean()
        test_accuracy = (probe(test_x).argmax(1) == test_y).float().mean()
    return {
        "train_accuracy": float(train_accuracy),
        "test_accuracy": float(test_accuracy),
        "train_identity_count": int(train_ids.numel()),
        "test_identity_count": int(test_ids.numel()),
        "train_sample_count": int(train_mask.sum()),
        "test_sample_count": int(test_mask.sum()),
    }


def _pearson(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.float() - left.float().mean()
    right = right.float() - right.float().mean()
    denominator = left.norm() * right.norm()
    if float(denominator) <= 0.0:
        return 0.0
    return float(torch.dot(left, right) / denominator)


def _dispersion_summary(values: list[float]) -> dict:
    result = _summary(values)
    ordered = torch.tensor(values, dtype=torch.float32)
    result.update(
        {
            "p10": float(torch.quantile(ordered, 0.10)),
            "p50": float(torch.quantile(ordered, 0.50)),
            "p90": float(torch.quantile(ordered, 0.90)),
        }
    )
    return result


def _within_identity_dispersion(
    features: torch.Tensor,
    labels: torch.Tensor,
    identities: torch.Tensor,
) -> dict:
    """Summarize within-ID scatter after per-sample L2 normalization.

    The per-identity series weights identities equally, while the sample series
    exposes long tails from individual images.  Neither is an objective to
    maximize or minimize; they are post-hoc diagnostics for retrieval tradeoffs.
    """
    normalized = F.normalize(features.float(), dim=1)
    per_identity = []
    per_sample = []
    for identity in identities:
        group = normalized[labels == identity]
        if group.numel() == 0:
            raise RuntimeError(f"identity {int(identity)} has no feature samples")
        center = group.mean(0, keepdim=True)
        distances = (group - center).norm(dim=1)
        per_identity.append(float(distances.mean()))
        per_sample.extend(float(value) for value in distances)
    return {
        "normalization": "l2_per_sample",
        "per_identity_mean_distance": _dispersion_summary(per_identity),
        "sample_distance": _dispersion_summary(per_sample),
    }


def _geometry_metrics(
    payload: dict,
    *,
    probe_steps: int,
    train_fraction: float,
    seed: int,
) -> dict:
    visible = payload["visible_features"].float()
    infrared = payload["ir_features"].float()
    visible_labels = payload["visible_labels"].long()
    ir_labels = payload["ir_labels"].long()
    identities = torch.unique(visible_labels, sorted=True)
    if not torch.equal(identities, torch.unique(ir_labels, sorted=True)):
        raise RuntimeError("geometry diagnostic requires matched RGB/IR identities")
    visible_centers = torch.stack(
        [visible[visible_labels == identity].mean(0) for identity in identities]
    )
    ir_centers = torch.stack(
        [infrared[ir_labels == identity].mean(0) for identity in identities]
    )
    visible_normalized = F.normalize(visible_centers, dim=1)
    ir_normalized = F.normalize(ir_centers, dim=1)
    visible_relation = visible_normalized @ visible_normalized.t()
    ir_relation = ir_normalized @ ir_normalized.t()
    off_diagonal = ~torch.eye(identities.numel(), dtype=torch.bool)
    joint_centers = {
        int(identity): torch.cat(
            (
                visible[visible_labels == identity],
                infrared[ir_labels == identity],
            )
        ).mean(0)
        for identity in identities
    }
    visible_residual = torch.stack(
        [
            feature - joint_centers[int(identity)]
            for feature, identity in zip(visible, visible_labels)
        ]
    )
    ir_residual = torch.stack(
        [
            feature - joint_centers[int(identity)]
            for feature, identity in zip(infrared, ir_labels)
        ]
    )
    all_features = torch.cat((visible, infrared))
    all_residual = torch.cat((visible_residual, ir_residual))
    all_identities = torch.cat((visible_labels, ir_labels))
    modalities = torch.cat(
        (
            torch.zeros(visible.size(0), dtype=torch.long),
            torch.ones(infrared.size(0), dtype=torch.long),
        )
    )
    residual_rms = all_residual.square().sum(1).mean().sqrt()
    modality_residual_gap = (
        visible_residual.mean(0) - ir_residual.mean(0)
    ).norm()
    return {
        "sample_count_per_modality": int(visible.size(0)),
        "identity_count": int(identities.numel()),
        "paired_identity_center_cosine": float(
            (visible_normalized * ir_normalized).sum(1).mean()
        ),
        "paired_identity_center_distance": float(
            (visible_normalized - ir_normalized).norm(dim=1).mean()
        ),
        "within_modality_relation_pearson": _pearson(
            visible_relation[off_diagonal],
            ir_relation[off_diagonal],
        ),
        "within_modality_relation_mae": float(
            (visible_relation[off_diagonal] - ir_relation[off_diagonal]).abs().mean()
        ),
        "within_id_dispersion": {
            "visible": _within_identity_dispersion(
                visible, visible_labels, identities
            ),
            "infrared": _within_identity_dispersion(
                infrared, ir_labels, identities
            ),
        },
        "identity_residual_modality_gap_over_rms": float(
            modality_residual_gap / residual_rms.clamp_min(1e-12)
        ),
        "raw_modality_probe": _linear_probe_accuracy(
            all_features,
            modalities,
            all_identities,
            steps=probe_steps,
            train_fraction=train_fraction,
            seed=seed,
        ),
        "identity_residual_modality_probe": _linear_probe_accuracy(
            all_residual,
            modalities,
            all_identities,
            steps=probe_steps,
            train_fraction=train_fraction,
            seed=seed,
        ),
    }


def _numeric_deltas(before: dict, after: dict) -> dict:
    deltas = {}
    for key, value in before.items():
        if isinstance(value, (int, float)) and isinstance(after.get(key), (int, float)):
            deltas[key] = float(after[key] - value)
        elif isinstance(value, dict) and isinstance(after.get(key), dict):
            nested = _numeric_deltas(value, after[key])
            if nested:
                deltas[key] = nested
    return deltas


def _modality_information(
    spec: dict,
    resolved: dict,
    project_root: Path,
    output_root: Path,
) -> dict:
    seed = int(resolved["fixed_config"]["seed"])
    _seed_everything(seed)
    config, loaders, model, device = _build_context(
        project_root, resolved, output_root, spec["name"]
    )
    batches = _cache_geometry_batches(loaders, int(spec["batches"]))
    pretrained_features = _extract_geometry_features(model, batches, device)
    checkpoint = _load_checkpoint(model, Path(spec["checkpoint"]))
    trained_features = _extract_geometry_features(model, batches, device)
    arguments = {
        "probe_steps": int(spec["probe_steps"]),
        "train_fraction": float(spec["probe_train_fraction"]),
        "seed": seed,
    }
    pretrained = _geometry_metrics(pretrained_features, **arguments)
    trained = _geometry_metrics(trained_features, **arguments)
    return {
        "schema_version": 1,
        "diagnostic": spec["name"],
        "type": spec["type"],
        "purpose": spec["purpose"],
        "checkpoint": checkpoint,
        "data_config": {
            "base_config": resolved["base_config"],
            "sysu_sr_data_root": resolved["fixed_config"]["sysu_sr_data_root"],
            "img_size": resolved["fixed_config"]["img_size"],
            "sampler_type": resolved["fixed_config"]["sampler_type"],
        },
        "batch_count": len(batches),
        "pretrained_initialization": pretrained,
        "c3_trained": trained,
        "c3_minus_pretrained": _numeric_deltas(pretrained, trained),
    }


def run_c3_recipe_diagnostic(
    spec: dict,
    resolved: dict,
    project_root: Path,
    output_root: Path,
) -> dict:
    diagnostic_type = str(spec["type"])
    if diagnostic_type == "loss_gradient_conflict":
        return _loss_gradient_conflict(
            spec, resolved, project_root, output_root
        )
    if diagnostic_type == "modality_information":
        return _modality_information(
            spec, resolved, project_root, output_root
        )
    raise ValueError(f"unsupported C3 diagnostic type: {diagnostic_type}")
