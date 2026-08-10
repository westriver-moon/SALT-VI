from __future__ import annotations

import datetime as dt
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .config import representation_specs, validate_enabled_representations
from .salt_adapter import SaltModelAdapter
from .storage import ArtifactLayout, FeatureArtifact, artifact_key, save_feature_artifact, sha256_file, write_json


def extract_all(config: Dict[str, Any]) -> Dict[str, Any]:
    import torch

    validate_enabled_representations(config)
    runtime = config["runtime"]
    seed = int(runtime["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    layout = ArtifactLayout(config["output_root"], config["run_id"])
    layout.create()
    catalog_path = layout.manifest_root / "catalog.json"
    if catalog_path.exists() and not config["overwrite"]:
        raise FileExistsError(
            f"Run {config['run_id']!r} already has a catalog; choose a new run_id or set overwrite: true"
        )
    catalog = {
        "schema_version": 1,
        "run_id": config["run_id"],
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config_path": config["config_path"],
        "artifacts": [],
    }
    work_root = layout.manifest_root / "model_runtime"

    for model_spec in config["models"]:
        needs_train_data = bool(config["splits"].get("train_rgb") or config["splits"].get("train_ir"))
        adapter = SaltModelAdapter(model_spec, runtime, work_root, needs_train_data=needs_train_data)
        checkpoint_hash = sha256_file(model_spec["checkpoint"])
        try:
            for split, split_config in config["splits"].items():
                if not split_config:
                    continue
                specs = representation_specs(config, split, adapter.model_id)
                for source in adapter.split_sources(split, split_config):
                    entries = _extract_source(adapter, source, specs, config, checkpoint_hash, layout)
                    catalog["artifacts"].extend(entries)
        finally:
            del adapter.model
            del adapter
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    write_json(catalog_path, catalog)
    write_json(layout.manifest_root / "resolved_analysis_config.json", config)
    return catalog


def _extract_source(
    adapter: SaltModelAdapter,
    source: Any,
    specs: List[Dict[str, Any]],
    config: Dict[str, Any],
    checkpoint_hash: str,
    layout: ArtifactLayout,
) -> List[Dict[str, Any]]:
    torch = adapter.torch
    accumulators = {spec["name"]: [] for spec in specs}
    seen = 0
    amp_enabled = bool(config["runtime"]["amp"]) and adapter.device.type == "cuda"
    with torch.no_grad():
        for batch in source.loader:
            batch_count = int(batch["img"].shape[0])
            with _autocast_context(torch, adapter.device.type, amp_enabled):
                for spec in specs:
                    values = adapter.encode(batch, spec)
                    if values.ndim != 2 or values.shape[0] != batch_count:
                        raise ValueError(
                            f"Encoder {spec['name']} returned {tuple(values.shape)} for batch {batch_count}"
                        )
                    accumulators[spec["name"]].append(values.detach().cpu().numpy().astype(np.float32))
            seen += batch_count
    if seen != len(source.labels):
        raise ValueError(f"Split {source.split_tag} yielded {seen} samples, expected {len(source.labels)}")

    entries = []
    for spec in specs:
        features = np.concatenate(accumulators[spec["name"]], axis=0)
        key = artifact_key(adapter.model_id, source.split_tag, spec["name"])
        relative = Path(adapter.model_id) / source.split_tag / f"{spec['name']}.npz"
        destination = layout.feature_root / relative
        metadata = {
            "schema_version": 1,
            "artifact_key": key,
            "model_id": adapter.model_id,
            "checkpoint": adapter.checkpoint,
            "checkpoint_sha256": checkpoint_hash,
            "salt_config": str(Path(adapter.model_spec["config"]).resolve()),
            "config_overrides": adapter.model_spec.get("overrides", {}),
            "dataset": str(adapter.config.dataset),
            "split": source.split,
            "split_tag": source.split_tag,
            "representation": spec,
            "sample_count": int(features.shape[0]),
            "feature_dim": int(features.shape[1]),
            "seed": int(config["runtime"]["seed"]),
        }
        save_feature_artifact(
            destination,
            FeatureArtifact(features, source.labels, source.cameras, source.sample_ids, metadata),
            overwrite=config["overwrite"],
        )
        entries.append(
            {
                **metadata,
                "feature_path": str(destination.resolve()),
                "metadata_path": str(destination.with_suffix(".meta.json").resolve()),
                "feature_sha256": sha256_file(str(destination)),
            }
        )
    return entries


def _autocast_context(torch: Any, device_type: str, enabled: bool):
    """Return an AMP context compatible with both old and new PyTorch releases."""
    if not enabled:
        return nullcontext()
    if hasattr(torch, "autocast"):
        return torch.autocast(device_type=device_type, enabled=True)
    if device_type == "cuda" and hasattr(torch.cuda, "amp"):
        return torch.cuda.amp.autocast(enabled=True)
    return nullcontext()
