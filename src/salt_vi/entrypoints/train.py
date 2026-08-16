import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import gc
import hashlib
import inspect
import json
import re
import time
import uuid
import torch
import random
import numpy as np
import yaml
from pathlib import Path
from salt_vi.data.loader import Loader
from salt_vi.engine import train, test, build_model
from salt_vi.utils import make_dirs, Logger
from salt_vi.optim import build_optimizer, build_lr_scheduler
from salt_vi.config.config_rn import get_args
from salt_vi.config.validation import validate_runtime_config
from salt_vi.entrypoints.output_paths import ensure_fresh_run_directory, resolve_run_directory
from salt_vi.retrieval import build_protocol_spec, get_retrieval_protocol
from salt_vi.utils.utils import save_train_configs, load_train_configs, time_now
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy
import math
from datetime import datetime, timezone


best_mAP_text = 0
best_rank1_text = 0
best_mINP_text = 0
best_mAP_ir = 0
best_rank1_ir = 0
best_mINP_ir = 0
best_mAP_fusion = 0
best_rank1_fusion = 0
best_mINP_fusion = 0

_BEST_METRIC_NAMES = (
    "best_mAP_text",
    "best_rank1_text",
    "best_mINP_text",
    "best_mAP_ir",
    "best_rank1_ir",
    "best_mINP_ir",
    "best_mAP_fusion",
    "best_rank1_fusion",
    "best_mINP_fusion",
)
_TRAINING_CHECKPOINT_SCHEMA_VERSION = 2
_RUN_MANIFEST_SCHEMA_VERSION = 1
_RUN_MANIFEST_FILENAME = "run_manifest.json"
_GOLDEN_EVALUATION_SCHEMA_VERSION = 1

_RUN_MANIFEST_TRANSIENT_KEYS = {
    "_explicit_cli_destinations",
    "run_uuid",
    "run_manifest_sha256",
    "training_weight_init_verified_path",
    "training_weight_init_verified_sha256",
}


def _merge_runtime_config(cli_config):
    """Merge default YAML, selected YAML, then explicitly supplied CLI values."""
    selected_path = cli_config.config_select
    if selected_path == 'default':
        selected_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config', 'default.yaml'
        )
    merged = load_train_configs(selected_path)
    if cli_config.config_select == 'default':
        merged.config_select = 'default'

    cli_values = vars(cli_config)
    explicit = set(cli_values.get('_explicit_cli_destinations', ()))
    explicit.discard('config_select')
    explicit.discard('config_overrides')
    for key in explicit:
        setattr(merged, key, cli_values[key])

    for item in cli_values.get('config_overrides', ()):
        key, separator, raw_value = item.partition('=')
        key = key.strip()
        if not separator or not key:
            raise ValueError(f"Invalid --set override {item!r}; expected KEY=VALUE")
        if key not in merged:
            raise KeyError(f"Unknown config key in --set override: {key}")
        setattr(merged, key, yaml.safe_load(raw_value))
    return merged


def _reset_best_metrics():
    namespace = globals()
    for name in _BEST_METRIC_NAMES:
        namespace[name] = 0


def _metric_bucket(result_key):
    return {
        "IR": "ir",
        "Fusion": "fusion",
        "Text": "text",
        "IR-RGBText": "fusion",
    }.get(str(result_key), "fusion")


def _best_value(bucket, metric):
    return float(globals()["best_{}_{}".format(metric, bucket)])


def _set_best_value(bucket, metric, value):
    globals()["best_{}_{}".format(metric, bucket)] = float(value)


def _log_retrieval_result(config, result_dict, protocol, logger):
    for result_key in protocol.RESULT_KEYS:
        if result_key not in result_dict:
            continue
        m_inp, m_ap, cmc = result_dict[result_key]
        message = (
            "Time: {}; Dataset: {}, Test Mode: {}, \nmINP: {} \nmAP: {} \n"
            "Rank: {}\n"
        ).format(
            time_now(),
            config.dataset,
            result_key,
            m_inp,
            m_ap,
            cmc,
        )
        if config.LOG4TEST:
            logger(message)
        else:
            print(message)


def _record_eval_epoch(
    config,
    model,
    result_dict,
    protocol,
    protocol_spec,
    current_epoch,
    performance_writer,
    logger,
):
    if not any(result_key in result_dict for result_key in protocol.RESULT_KEYS):
        raise RuntimeError(
            "Evaluation returned no protocol result keys; expected {}".format(
                list(protocol.RESULT_KEYS)
            )
        )
    save_best_per_metric = bool(getattr(config, "save_best_per_metric", False))
    for result_key in protocol.RESULT_KEYS:
        if result_key not in result_dict:
            continue
        m_inp, m_ap, cmc = result_dict[result_key]
        bucket = _metric_bucket(result_key)
        is_best_rank = cmc[0] >= _best_value(bucket, "rank1")
        performance_writer.add_scalar(f"R1_{result_key}", cmc[0], current_epoch)
        performance_writer.add_scalar(f"mAP_{result_key}", m_ap, current_epoch)
        performance_writer.add_scalar(f"mINP_{result_key}", m_inp, current_epoch)
        checkpoint_paths = {}
        new_best_metrics = []
        if bucket == "ir":
            if is_best_rank:
                logger(f"New Best {result_key}!!!")
                _set_best_value("ir", "rank1", cmc[0])
                _set_best_value("ir", "mAP", m_ap)
                _set_best_value("ir", "mINP", m_inp)
            logger(
                "Best {} mINP: {}, Best mAP: {}, Best Rank1: {}".format(
                    result_key,
                    _best_value("ir", "mINP"),
                    _best_value("ir", "mAP"),
                    _best_value("ir", "rank1"),
                )
            )
            model.save_model(current_epoch, is_best_rank, mode="IR")
        elif bucket == "text":
            if is_best_rank:
                logger(f"New Best {result_key}!!!")
                _set_best_value("text", "rank1", cmc[0])
                _set_best_value("text", "mAP", m_ap)
                _set_best_value("text", "mINP", m_inp)
            logger(
                "Best {} mINP: {}, Best mAP: {}, Best Rank1: {}".format(
                    result_key,
                    _best_value("text", "mINP"),
                    _best_value("text", "mAP"),
                    _best_value("text", "rank1"),
                )
            )
            model.save_model(current_epoch, is_best_rank, mode="Text")
        else:
            is_best_rank = (
                cmc[0] > _best_value("fusion", "rank1")
                if save_best_per_metric
                else cmc[0] >= _best_value("fusion", "rank1")
            )
            is_best_map = save_best_per_metric and m_ap > _best_value("fusion", "mAP")
            is_best_minp = save_best_per_metric and m_inp > _best_value("fusion", "mINP")
            if is_best_rank:
                logger(f"New Best {result_key}!!!")
                _set_best_value("fusion", "rank1", cmc[0])
                if not save_best_per_metric:
                    _set_best_value("fusion", "mAP", m_ap)
                    _set_best_value("fusion", "mINP", m_inp)
            if is_best_map:
                _set_best_value("fusion", "mAP", m_ap)
            if is_best_minp:
                _set_best_value("fusion", "mINP", m_inp)
            logger(
                "Best {} mINP: {}, Best mAP: {}, Best Rank1: {}".format(
                    result_key,
                    _best_value("fusion", "mINP"),
                    _best_value("fusion", "mAP"),
                    _best_value("fusion", "rank1"),
                )
            )
            if save_best_per_metric:
                new_best_metrics = [
                    metric
                    for metric, improved in (
                        ("Rank-1", is_best_rank),
                        ("mAP", is_best_map),
                        ("mINP", is_best_minp),
                    )
                    if improved
                ]
                checkpoint_paths = model.save_metric_checkpoints(
                    current_epoch, new_best_metrics, mode="Fusion"
                )
            else:
                model.save_model(current_epoch, is_best_rank, mode="Fusion")
        _append_metric_event(
            config,
            "eval_epoch",
            epoch=current_epoch,
            dataset=config.dataset,
            protocol=protocol_spec.identifier,
            protocol_spec=protocol_spec.as_dict(),
            query=protocol.QUERY_NAME,
            gallery=protocol.GALLERY_NAME,
            metrics={
                "Rank-1": float(cmc[0]),
                "mAP": float(m_ap),
                "mINP": float(m_inp),
            },
            best_so_far={
                "Rank-1": _best_value(bucket, "rank1"),
                "mAP": _best_value(bucket, "mAP"),
                "mINP": _best_value(bucket, "mINP"),
            },
            is_new_best=bool(is_best_rank),
            new_best_metrics=new_best_metrics,
            checkpoint_paths=checkpoint_paths,
        )
        logger(
            "Time: {}; Dataset: {}, Test Mode: {}, \nmINP: {} \nmAP: {} \n"
            "Rank: {}\n".format(
                time_now(),
                config.dataset,
                result_key,
                m_inp,
                m_ap,
                cmc,
            )
        )


def _best_metric_state():
    namespace = globals()
    return {name: float(namespace[name]) for name in _BEST_METRIC_NAMES}


def _restore_best_metric_state(state):
    missing = sorted(set(_BEST_METRIC_NAMES) - set(state))
    unknown = sorted(set(state) - set(_BEST_METRIC_NAMES))
    if missing or unknown:
        raise ValueError(f"Invalid best-metric state; missing={missing}, unknown={unknown}")
    namespace = globals()
    for name in _BEST_METRIC_NAMES:
        namespace[name] = float(state[name])


def _capture_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _training_checkpoint_path(output_path):
    return os.path.join(str(output_path), "checkpoint", "checkpoint_latest.pth")


def _load_trusted_training_checkpoint(path):
    """Load a locally produced full-state training checkpoint.

    Training checkpoints deliberately contain optimizer and Python/NumPy RNG
    state, so they require pickle deserialization.  They are trusted local
    assets, never untrusted downloads.  PyTorch 2.6 changed its default to
    ``weights_only=True``; requesting the historical full-state behaviour
    explicitly keeps automatic resume portable across supported PyTorch
    versions.
    """
    kwargs = {"map_location": torch.device("cpu")}
    try:
        supports_weights_only = "weights_only" in inspect.signature(torch.load).parameters
    except (TypeError, ValueError):  # pragma: no cover - defensive for custom loaders.
        supports_weights_only = False
    if supports_weights_only:
        kwargs["weights_only"] = False
    return torch.load(path, **kwargs)


def _save_training_checkpoint(
    path,
    epoch,
    model,
    optimizer,
    scheduler,
    scaler,
    *,
    run_uuid=None,
    run_manifest_sha256=None,
):
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "schema_version": _TRAINING_CHECKPOINT_SCHEMA_VERSION,
        "epoch": int(epoch),
        "run_uuid": run_uuid,
        "run_manifest_sha256": run_manifest_sha256,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "experiment_state": {
            "best_metrics": _best_metric_state(),
            "metric_checkpoint_paths": dict(
                getattr(model, "_metric_checkpoint_paths", {}) or {}
            ),
        },
        "rng_state": _capture_rng_state(),
    }
    temporary = path + ".tmp"
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    return path


def _validate_checkpoint_run_identity(
    checkpoint,
    expected_run_uuid=None,
    expected_run_manifest_sha256=None,
):
    if expected_run_uuid is None and expected_run_manifest_sha256 is None:
        return
    checkpoint_run_uuid = checkpoint.get("run_uuid")
    checkpoint_manifest_sha256 = checkpoint.get("run_manifest_sha256")
    if not checkpoint_run_uuid or not checkpoint_manifest_sha256:
        raise ValueError(
            "Training checkpoint has no run identity; convert it to the "
            "run-manifest checkpoint schema before complete resume"
        )
    if checkpoint_run_uuid != expected_run_uuid:
        raise ValueError(
            "Training checkpoint run UUID mismatch: checkpoint={}, manifest={}".format(
                checkpoint_run_uuid, expected_run_uuid
            )
        )
    if checkpoint_manifest_sha256 != expected_run_manifest_sha256:
        raise ValueError(
            "Training checkpoint run manifest hash mismatch: checkpoint={}, manifest={}".format(
                checkpoint_manifest_sha256, expected_run_manifest_sha256
            )
        )


def _load_training_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    scaler,
    device,
    *,
    expected_run_uuid=None,
    expected_run_manifest_sha256=None,
):
    # Keep RNG byte tensors on CPU; load_state_dict moves model and optimizer
    # tensors to their owning parameter devices.
    checkpoint = _load_trusted_training_checkpoint(path)
    required = {
        "schema_version",
        "epoch",
        "run_uuid",
        "run_manifest_sha256",
        "model",
        "optimizer",
        "scheduler",
        "scaler",
        "experiment_state",
        "rng_state",
    }
    missing = sorted(required - set(checkpoint)) if isinstance(checkpoint, dict) else sorted(required)
    if missing:
        raise ValueError(f"Training checkpoint is incomplete; missing={missing}")
    if int(checkpoint["schema_version"]) != _TRAINING_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported training checkpoint schema: {checkpoint['schema_version']}"
        )
    _validate_checkpoint_run_identity(
        checkpoint,
        expected_run_uuid=expected_run_uuid,
        expected_run_manifest_sha256=expected_run_manifest_sha256,
    )
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    scaler.load_state_dict(checkpoint["scaler"])
    experiment_state = checkpoint["experiment_state"]
    _restore_best_metric_state(experiment_state["best_metrics"])
    model._metric_checkpoint_paths = dict(
        experiment_state.get("metric_checkpoint_paths", {})
    )
    _restore_rng_state(checkpoint["rng_state"])
    return int(checkpoint["epoch"]) + 1


def _expand_shared_bn_for_uni_bn(model, state):
    """Clone an E4 shared BN into the four modality-specific QBN branches."""
    if not getattr(getattr(model, "classifier", None), "uni_BN", False):
        return state
    state = dict(state)
    suffixes = ("weight", "bias", "running_mean", "running_var", "num_batches_tracked")
    shared = {suffix: state.get(f"classifier.BN.{suffix}") for suffix in suffixes}
    if all(value is None for value in shared.values()):
        return state
    missing = [suffix for suffix, value in shared.items() if value is None]
    if missing:
        raise KeyError(f"Incomplete shared classifier BN in warm-start checkpoint: {missing}")
    for branch in ("RGB", "IR", "Fusion", "Text"):
        for suffix, value in shared.items():
            state[f"classifier.BN_{branch}.{suffix}"] = value.clone()
    for suffix in suffixes:
        state.pop(f"classifier.BN.{suffix}", None)
    return state


def _append_metric_event(config, event_type, *, epoch=None, **payload):
    path = getattr(config, "metric_events_path", None)
    experiment_id = getattr(config, "metric_experiment_id", None)
    if not path or not experiment_id:
        return
    event = {
        "schema_version": 1,
        "event_id": f"{experiment_id}:{event_type}:{epoch if epoch is not None else 'run'}:{time.time_ns()}",
        "experiment_id": experiment_id,
        "event_type": event_type,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "attempt": int(getattr(config, "metric_attempt", 1)),
    }
    if epoch is not None:
        event["epoch"] = int(epoch)
    event.update(payload)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _jsonable(value):
    if isinstance(value, dict):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted(_jsonable(item) for item in value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _resolved_config_digest(config):
    values = dict(vars(config)) if hasattr(config, "__dict__") else dict(config)
    values = {
        key: item
        for key, item in values.items()
        if key not in _RUN_MANIFEST_TRANSIENT_KEYS
    }
    payload = _jsonable(values)
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _data_manifest_digest(config):
    entries = []
    view_manifest = getattr(config, "sysu_sr_view_manifest", None)
    if view_manifest:
        path = os.path.abspath(os.path.expanduser(str(view_manifest)))
        if not os.path.isfile(path):
            raise FileNotFoundError(f"PASD view manifest is missing: {path}")
        entries.append((path, _sha256_file(path)))
    data_root = getattr(config, "sysu_sr_data_root", None)
    if data_root and not view_manifest:
        root = os.path.abspath(os.path.expanduser(str(data_root)))
        if os.path.isdir(root):
            for name in ("manifest.json", "manifest.jsonl", "validation-report.json"):
                path = os.path.join(root, name)
                if os.path.isfile(path):
                    entries.append((path, _sha256_file(path)))
    entries = sorted(set(entries))
    if not entries:
        return None
    payload = [
        {"path": path, "sha256": digest}
        for path, digest in entries
    ]
    return {
        "paths": [path for path, _ in entries],
        "sha256": hashlib.sha256(
            _canonical_json(payload).encode("utf-8")
        ).hexdigest(),
    }


def _run_manifest_path(config):
    return os.path.join(str(config.output_path), _RUN_MANIFEST_FILENAME)


def _load_run_manifest(config):
    path = _run_manifest_path(config)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Complete resume requested but run manifest is missing: {path}"
        )
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if int(payload.get("schema_version", 0)) != _RUN_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported run manifest schema: {}".format(
                payload.get("schema_version")
            )
        )
    return payload


def _validate_run_manifest(config, protocol_spec, manifest):
    expected = {
        "resolved_config_sha256": _resolved_config_digest(config),
        "data_manifest": _data_manifest_digest(config),
        "init_checkpoint_sha256": _init_checkpoint_digest(config),
        "protocol_identifier": protocol_spec.identifier,
    }
    mismatches = []
    for key, value in expected.items():
        recorded = manifest.get(key)
        if recorded != value:
            mismatches.append(
                "{}: recorded={!r}, recomputed={!r}".format(
                    key, recorded, value
                )
            )
    if mismatches:
        raise ValueError(
            "Run manifest identity mismatch in {}: {}".format(
                _run_manifest_path(config), "; ".join(mismatches)
            )
        )


def _init_checkpoint_digest(config):
    if str(getattr(config, "mode", "")) == "test":
        path = getattr(config, "test_model_path", None)
    else:
        path = getattr(config, "training_weight_init", None)
    if not path:
        return None
    if str(getattr(config, "mode", "")) != "test":
        return _verify_training_weight_init(config)
    path = os.path.abspath(os.path.expanduser(str(path)))
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Test checkpoint is missing: {path}")
    return _sha256_file(path)


def _write_run_manifest(config, run_uuid, protocol_spec):
    path = _run_manifest_path(config)
    payload = {
        "schema_version": _RUN_MANIFEST_SCHEMA_VERSION,
        "run_uuid": run_uuid,
        "resolved_config_sha256": _resolved_config_digest(config),
        "data_manifest": _data_manifest_digest(config),
        "init_checkpoint_sha256": _init_checkpoint_digest(config),
        "protocol_identifier": protocol_spec.identifier,
        "protocol_spec": protocol_spec.as_dict(),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = path + ".tmp"
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    return path, _sha256_file(path)


def _write_golden_evaluation(config, protocol_spec, result_dict):
    path = os.path.abspath(os.path.expanduser(str(config.golden_evaluation_path)))
    if os.path.exists(path):
        raise FileExistsError(f"Refusing to overwrite golden evaluation: {path}")
    payload = {
        "schema_version": _GOLDEN_EVALUATION_SCHEMA_VERSION,
        "experiment_id": getattr(config, "metric_experiment_id", None),
        "checkpoint_path": os.path.abspath(
            os.path.expanduser(str(config.test_model_path))
        ),
        "checkpoint_sha256": _sha256_file(config.test_model_path),
        "resolved_config_sha256": _resolved_config_digest(config),
        "data_manifest": _data_manifest_digest(config),
        "protocol_spec": protocol_spec.as_dict(),
        "eval_caption_seed": protocol_spec.eval_caption_seed,
        "metrics": {},
        "run_manifest_path": _run_manifest_path(config),
        "run_manifest_sha256": config.run_manifest_sha256,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    for mode, (minp, m_ap, cmc) in result_dict.items():
        payload["metrics"][mode] = {
            "Rank-1": float(cmc[0]),
            "mAP": float(m_ap),
            "mINP": float(minp),
        }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    return path


def _verify_training_weight_init(config):
    source = getattr(config, "training_weight_init", None)
    expected = getattr(config, "training_weight_init_sha256", None)
    if not source:
        return None
    if not expected:
        raise ValueError(
            "training_weight_init_sha256 is required whenever "
            "training_weight_init is set"
        )
    expected = str(expected).lower()
    if re.fullmatch(r"[0-9a-f]{64}", expected) is None:
        raise ValueError(
            "training_weight_init_sha256 must be exactly 64 hexadecimal characters"
        )
    source = os.path.abspath(str(source))
    if not os.path.isfile(source):
        raise FileNotFoundError(f"Warm-start checkpoint is missing: {source}")
    cached = getattr(config, "training_weight_init_verified_sha256", None)
    cached_source = getattr(config, "training_weight_init_verified_path", None)
    if cached == expected and cached_source == source:
        return expected
    actual = _sha256_file(source)
    if actual != expected:
        raise ValueError(
            "Warm-start checkpoint SHA-256 mismatch for {}: expected {}, got {}".format(
                source, expected, actual
            )
        )
    config.training_weight_init_verified_sha256 = actual
    config.training_weight_init_verified_path = source
    return actual


def _materialize_warm_start_checkpoint(model, config):
    """Persist the exact converted in-memory state evaluated at epoch -1."""
    source = os.path.abspath(str(config.training_weight_init))
    if not os.path.isfile(source):
        raise FileNotFoundError(f"Warm-start checkpoint is missing: {source}")
    os.makedirs(model.save_model_path, exist_ok=True)
    destination = os.path.join(model.save_model_path, "model_Fusion_epoch_-1.pth")
    source_hash = _sha256_file(source)
    if os.path.exists(destination):
        raise FileExistsError(f"Refusing to replace existing epoch -1 checkpoint: {destination}")
    temporary = destination + ".tmp"
    try:
        torch.save(model.state_dict(), temporary)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    converted_hash = _sha256_file(destination)
    metadata = {
        "source_path": source,
        "source_sha256": source_hash,
        "converted_checkpoint_sha256": converted_hash,
        "conversions": dict(getattr(model, "_warm_start_conversion_info", {}) or {}),
    }
    checkpoint_paths = {metric: destination for metric in ("Rank-1", "mAP", "mINP")}
    model._metric_checkpoint_paths = dict(checkpoint_paths)
    return checkpoint_paths, {metric: converted_hash for metric in checkpoint_paths}, metadata
def seed_torch(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _configure_cuda_visibility(config, environ=None):
    """Apply the configured CUDA visibility for canonical training."""
    environ = os.environ if environ is None else environ
    configured = str(config.CUDA_VISIBLE_DEVICES)
    environ["CUDA_VISIBLE_DEVICES"] = configured
    return configured


def _infer_pos_grid(token_count, target_ratio=2.0):
    candidates = []
    for height in range(1, int(math.sqrt(token_count)) + 2):
        if token_count % height:
            continue
        width = token_count // height
        for h, w in ((height, width), (width, height)):
            candidates.append((abs((h / float(w)) - target_ratio), h, w))
    if not candidates:
        raise ValueError(f"Cannot factor positional token count {token_count}")
    _, height, width = min(candidates)
    return height, width


def _load_compatible_state_dict(
    model, path, device, *, preserve_derived_backups=False
):
    payload = torch.load(path, map_location=device)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        payload = payload["model_state_dict"]
    elif isinstance(payload, dict) and "state_dict" in payload:
        payload = payload["state_dict"]
    shared_bn_expanded = bool(
        getattr(getattr(model, "classifier", None), "uni_BN", False)
        and any(key.startswith("classifier.BN.") for key in payload)
    )
    state = _expand_shared_bn_for_uni_bn(model, payload)
    if (
        preserve_derived_backups
        and model._uses_spatial_map_visual()
        and any(
            key.startswith(("backup_pool.", "backup_classifier."))
            for key in state
        )
    ):
        _initialize_spatial_backups(model, model.args)
    target_state = model.state_dict()
    discarded_derived_backup_keys = []
    for prefix in ("backup_pool.", "backup_classifier."):
        if any(key.startswith(prefix) for key in target_state):
            continue
        for key in [key for key in state if key.startswith(prefix)]:
            discarded_derived_backup_keys.append(key)
            state.pop(key)
    key = "base_model.visual.vit.pos_embed"
    target = target_state.get(key)
    source = state.get(key)
    position_embedding = None
    if source is not None and target is not None and tuple(source.shape) != tuple(target.shape):
        visual = model.base_model.visual.vit
        old_h, old_w = _infer_pos_grid(source.shape[1] - 1, target_ratio=visual.base_grid_size[0] / float(visual.base_grid_size[1]))
        new_h, new_w = visual.base_grid_size
        token, grid = source[:, :1], source[:, 1:]
        grid = grid.reshape(1, old_h, old_w, -1).permute(0, 3, 1, 2).float()
        grid = torch.nn.functional.interpolate(grid, size=(new_h, new_w), mode="bilinear", align_corners=False)
        grid = grid.permute(0, 2, 3, 1).reshape(1, new_h * new_w, -1).to(source.dtype)
        state[key] = torch.cat([token, grid], dim=1)
        position_embedding = {
            "key": key,
            "source_grid_hw": [old_h, old_w],
            "target_grid_hw": [new_h, new_w],
            "mode": "bilinear",
            "align_corners": False,
        }
        print(f"Resized warm-start positional embedding {old_h}x{old_w} -> {new_h}x{new_w}")
    result = model.load_state_dict(state, strict=False)
    missing_keys = list(result.missing_keys)
    unexpected_keys = list(result.unexpected_keys)
    incompatible_missing = sorted(set(missing_keys))
    incompatible_unexpected = sorted(unexpected_keys)
    if incompatible_missing or incompatible_unexpected:
        raise RuntimeError(
            f"Incompatible checkpoint after supported conversions: {path}; "
            f"missing_keys={incompatible_missing}; "
            f"unexpected_keys={incompatible_unexpected}"
        )
    conversion_info = {
        "shared_bn_expanded_to_qbn": shared_bn_expanded,
        "position_embedding_interpolation": position_embedding,
        "discarded_derived_backup_keys": sorted(discarded_derived_backup_keys),
    }
    model._warm_start_conversion_info = conversion_info
    return result


def _load_training_weight_init(model, config, device):
    if not config.training_weight_init:
        return
    _verify_training_weight_init(config)
    _load_compatible_state_dict(model, config.training_weight_init, device)
    if config.Fix_Visual:
        _initialize_spatial_backups(model, config)
    print(f"Successfully load model from {config.training_weight_init}")
    if getattr(config, "training_weight_init_source_config", None):
        print(f"Resolved PMT_VIT image-only source config: {config.training_weight_init_source_config}")
    if getattr(config, "training_weight_init_metrics", None):
        print(f"Resolved PMT_VIT image-only metrics (mINP, mAP, Rank1): {config.training_weight_init_metrics}")
    if getattr(config, "training_weight_init_epoch", None) is not None:
        print(f"Resolved PMT_VIT image-only checkpoint epoch: {config.training_weight_init_epoch}")


def _initialize_spatial_backups(model, config):
    """Create registered backup modules before full-state restore or after warm-start."""
    if not model._uses_spatial_map_visual():
        return
    model.backup_pool = deepcopy(model.base_model.visual.__getattr__(config.pooling))
    model.backup_classifier = deepcopy(model.classifier)


def main(config):
    if bool(getattr(config, "DataParallel", False)):
        raise RuntimeError(
            "Legacy DataParallel is unsupported by SALT-VI. Use one process per GPU "
            "or fixed_visual_data_parallel for frozen visual replicas."
        )
    validate_runtime_config(config)
    retrieval_protocol = get_retrieval_protocol(
        getattr(config, "retrieval_backend", "identity_text")
    )
    fusion_result_key = retrieval_protocol.RESULT_KEY
    os.environ["CUDA_VISIBLE_DEVICES"] = config.CUDA_VISIBLE_DEVICES
    protocol_spec = build_protocol_spec(config, retrieval_protocol)
    device = torch.device(f'cuda:{config.gpu_id}' if torch.cuda.is_available() else "cpu")

    global best_mAP_text
    global best_rank1_text
    global best_mINP_text
    global best_mAP_ir
    global best_rank1_ir
    global best_mINP_ir
    global best_mAP_fusion
    global best_rank1_fusion
    global best_mINP_fusion
    _reset_best_metrics()

    print("=================Constructing output dir=================")
    config.output_path = resolve_run_directory(config)
    complete_resume = bool(
        getattr(config, "auto_resume_training_from_lastest_step", False)
    )
    if int(getattr(config, "resume_train_epoch", -1)) >= 0:
        raise RuntimeError(
            "model-only resume via resume_train_epoch is retired; convert the "
            "checkpoint to the run-manifest full-state schema before resuming"
        )
    if int(getattr(config, "metric_boost_resume_epoch", 0)) > 0:
        raise RuntimeError(
            "metric_boost_resume_epoch is retired; start a fresh run or use "
            "complete run-manifest resume"
        )
    is_resume = complete_resume
    if config.mode == "train" and not complete_resume:
        _verify_training_weight_init(config)
    if config.mode == "train" and not complete_resume:
        ensure_fresh_run_directory(config)
    if complete_resume or (
        config.mode == "test" and os.path.isfile(_run_manifest_path(config))
    ):
        manifest = _load_run_manifest(config)
        _validate_run_manifest(config, protocol_spec, manifest)
        config.run_uuid = manifest["run_uuid"]
        config.run_manifest_sha256 = _sha256_file(_run_manifest_path(config))
    else:
        config.run_uuid = uuid.uuid4().hex
        _, config.run_manifest_sha256 = _write_run_manifest(
            config, config.run_uuid, protocol_spec
        )
    if config.DEBUG:
        print(f"Debug [{config.mode}] mode, dir: {config.output_path}")
    elif complete_resume and config.mode == 'train':
        print(f"Resume training from the latest step, dir: {config.output_path}")
    elif config.mode == 'test':
        print(f"Start testing with trained model, dir: {config.output_path}")
    else:
        print(f"start training from zero, dir {config.output_path}, training mode: {config.training_mode}")


    print("=================Preparing data=================")
    if config.dataset == 'sysu':
        print(f"Dataset: {config.dataset}, dir: {config.sysu_data_path}")
        config.pid_num = 395
    elif config.dataset == 'regdb':
        print(f"Dataset: {config.dataset}, dir: {config.regdb_data_path}")
        config.pid_num = 206
    elif config.dataset == 'llcm':
        print(f"Dataset: {config.dataset}, dir: {config.llcm_data_path}")
        config.pid_num = 713
    loaders = Loader(config)


    print("=================Preparing model=================")
    model = build_model(config)
    model = model.to(device)

    if config.mode == 'train':
        make_dirs(model.output_path)
        make_dirs(model.save_model_path)
        make_dirs(model.save_logs_path)
        check_point_path = _training_checkpoint_path(model.output_path)

        logger = Logger(os.path.join(os.path.join(config.output_path, 'logs/'), 'log.log'))
        logger('\n' * 3)
        logger(config)


        performance_writer = SummaryWriter(os.path.join(model.output_path,'vis_logs/performance'))
        loss_writer = SummaryWriter(os.path.join(model.output_path,'vis_logs/loss'))
        save_train_configs(config.output_path, config)

        if complete_resume:
            _initialize_spatial_backups(model, config)
        else:
            _load_training_weight_init(model, config, device)

        print("=================preparing optimizer=================")

        optimizer = build_optimizer(config, model)
        scheduler = build_lr_scheduler(config, optimizer)
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        else:
            scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

        start_train_epoch = 0
        if complete_resume:
            if not os.path.isfile(check_point_path):
                raise FileNotFoundError(
                    f"Automatic resume requested but checkpoint is missing: {check_point_path}"
                )
            start_train_epoch = _load_training_checkpoint(
                check_point_path,
                model,
                optimizer,
                scheduler,
                scaler,
                device,
                expected_run_uuid=config.run_uuid,
                expected_run_manifest_sha256=config.run_manifest_sha256,
            )
            print(f"Resuming complete training state from epoch {start_train_epoch}")

        # Replicas are snapshots, so they must be created only after the final
        # model state (fresh warm-start or full resume) is loaded.
        model.configure_fixed_visual_data_parallel()

        if bool(getattr(config, "eval_before_train", False)) and start_train_epoch == 0:
            result_dict = test(model, loaders, config, device)
            if fusion_result_key not in result_dict:
                raise RuntimeError(
                    f"eval_before_train requires {fusion_result_key} retrieval metrics"
                )
            mINP_fusion, mAP_fusion, cmc_fusion = result_dict[fusion_result_key]
            initial_values = (float(cmc_fusion[0]), float(mAP_fusion), float(mINP_fusion))
            if not all(np.isfinite(value) for value in initial_values):
                raise FloatingPointError(f"Non-finite warm-start metrics: {initial_values}")
            best_rank1_fusion, best_mAP_fusion, best_mINP_fusion = initial_values
            warm_start_paths, warm_start_hashes, warm_start_metadata = (
                _materialize_warm_start_checkpoint(model, config)
            )
            _append_metric_event(
                config,
                "eval_epoch",
                epoch=-1,
                phase="warm_start_before_training",
                dataset=config.dataset,
                protocol=protocol_spec.identifier,
                protocol_spec=protocol_spec.as_dict(),
                query=retrieval_protocol.QUERY_NAME,
                gallery=retrieval_protocol.GALLERY_NAME,
                metrics={
                    "Rank-1": initial_values[0],
                    "mAP": initial_values[1],
                    "mINP": initial_values[2],
                },
                best_so_far={
                    "Rank-1": initial_values[0],
                    "mAP": initial_values[1],
                    "mINP": initial_values[2],
                },
                is_new_best=True,
                new_best_metrics=["Rank-1", "mAP", "mINP"],
                checkpoint_paths=warm_start_paths,
                checkpoint_sha256=warm_start_hashes,
                source_warm_start=str(config.training_weight_init),
                source_warm_start_sha256=warm_start_metadata["source_sha256"],
                warm_start_conversion=warm_start_metadata["conversions"],
            )
            logger(
                "Warm-start evaluation before training: "
                f"mINP={initial_values[2]}, mAP={initial_values[1]}, Rank-1={initial_values[0]}"
            )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Match preflight: training randomness starts from the configured seed,
            # independent of RNG consumed by warm-start evaluation/DataLoader workers.
            seed_torch(config.seed)


        for current_epoch in range(start_train_epoch, config.total_train_epoch):
            epoch_started = time.monotonic()
            unfreeze_summary = model.configure_epoch_trainability(current_epoch)
            if unfreeze_summary and unfreeze_summary.get("epoch") == current_epoch:
                logger(f"Visual unfreeze summary: {unfreeze_summary}")
            scheduler.step(current_epoch)

            result_vals, result = train(model, loaders, scaler, config, optimizer, current_epoch=current_epoch)
            # visual log
            for key, value in zip(*result_vals):
                loss_writer.add_scalar(key, value, current_epoch)
            logger('Time: {}; Epoch: {}; {}'.format(time_now(), current_epoch, result))
            epoch_values = {key: float(value) for key, value in zip(*result_vals)}
            _append_metric_event(
                config,
                "train_epoch",
                epoch=current_epoch,
                losses={key: value for key, value in epoch_values.items() if "loss" in key},
                scalars={
                    **({"accuracy": epoch_values["acc"]} if "acc" in epoch_values else {}),
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                },
                duration_seconds=float(time.monotonic() - epoch_started),
                amp_skipped_steps=int(epoch_values.get("amp_skipped_steps", 0)),
            )
            if getattr(model, "raw_pa", None) is not None:
                pa_value = float(model.current_pa().detach().cpu())
                logger(f"Learnable pa: {pa_value}")
                performance_writer.add_scalar("learnable_pa", pa_value, current_epoch)

            # testing while training
            if current_epoch + 1 >= config.eval_start_epoch and (current_epoch + 1) % config.eval_epoch == 0:
                result_dict = test(model, loaders, config, device)
                _record_eval_epoch(
                    config,
                    model,
                    result_dict,
                    retrieval_protocol,
                    protocol_spec,
                    current_epoch,
                    performance_writer,
                    logger,
                )

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            checkpoint_interval = max(1, int(getattr(config, "checkpoint_epoch", 1)))
            if (
                (current_epoch + 1) % checkpoint_interval == 0
                or current_epoch + 1 == int(config.total_train_epoch)
            ):
                saved_checkpoint = _save_training_checkpoint(
                    check_point_path,
                    current_epoch,
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    run_uuid=config.run_uuid,
                    run_manifest_sha256=config.run_manifest_sha256,
                )
                logger(f"Saved complete training checkpoint: {saved_checkpoint}")

        performance_writer.close()
        loss_writer.close()

    elif config.mode == 'test':
        make_dirs(model.output_path)
        make_dirs(model.save_model_path)
        make_dirs(model.save_logs_path)
        logger = Logger(os.path.join(os.path.join(config.output_path, 'logs/'), 'test.log'))
        logger('\n' * 3)
        logger(config)
        print('Testing Modality Mode:{}'.format(config.test_modality))
        print('Testing Model Type:{}'.format(config.test_model_type))
        _load_compatible_state_dict(
            model,
            config.test_model_path,
            device,
            preserve_derived_backups=True,
        )
        if model._uses_spatial_map_visual() and not hasattr(model, "backup_pool"):
            _initialize_spatial_backups(model, config)
        model.configure_fixed_visual_data_parallel()
        print('Successfully resume model from {}'.format(config.test_model_path))
        result_dict = test(model, loaders, config, device)
        _log_retrieval_result(config, result_dict, retrieval_protocol, logger)

        if getattr(config, "golden_evaluation_path", None):
            golden_path = _write_golden_evaluation(
                config, protocol_spec, result_dict
            )
            print(f"Wrote golden evaluation: {golden_path}")


if __name__ == '__main__':
    config = _merge_runtime_config(get_args())
    # CUDA visibility must be fixed before seed_torch calls torch.cuda.* and
    # initializes the CUDA runtime.
    _configure_cuda_visibility(config)
    seed_torch(config.seed)
    main(config)
