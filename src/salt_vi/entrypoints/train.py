import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import gc
import hashlib
import inspect
import json
import time
import torch
import random
import numpy as np
import yaml
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
_TRAINING_CHECKPOINT_SCHEMA_VERSION = 1


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


def _save_training_checkpoint(path, epoch, model, optimizer, scheduler, scaler):
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "schema_version": _TRAINING_CHECKPOINT_SCHEMA_VERSION,
        "epoch": int(epoch),
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


def _load_training_checkpoint(path, model, optimizer, scheduler, scaler, device):
    # Keep RNG byte tensors on CPU; load_state_dict moves model and optimizer
    # tensors to their owning parameter devices.
    checkpoint = _load_trusted_training_checkpoint(path)
    required = {
        "schema_version",
        "epoch",
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
    validate_runtime_config(config)
    retrieval_protocol = get_retrieval_protocol(
        getattr(config, "retrieval_backend", "legacy")
    )
    fusion_result_key = retrieval_protocol.RESULT_KEY
    if bool(getattr(config, "DataParallel", False)):
        raise RuntimeError(
            "Legacy DataParallel is unsupported by SALT-VI. Use one process per GPU "
            "or fixed_visual_data_parallel for frozen visual replicas."
        )
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
    ensure_fresh_run_directory(config)
    if config.DEBUG:
        print(f"Debug [{config.mode}] mode, dir: {config.output_path}")
    elif (config.auto_resume_training_from_lastest_step or config.resume_train_epoch>=0) and config.mode == 'train':
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

        complete_resume = bool(
            getattr(config, "auto_resume_training_from_lastest_step", False)
        )
        legacy_resume_epoch = int(getattr(config, "resume_train_epoch", -1))
        if complete_resume or legacy_resume_epoch >= 0:
            # Full/model-only checkpoints may contain the registered RN backup modules.
            _initialize_spatial_backups(model, config)
        else:
            _load_training_weight_init(model, config, device)
        if legacy_resume_epoch >= 0 and not complete_resume:
            model.resume_model(legacy_resume_epoch, mode="Fusion")

        print("=================preparing optimizer=================")

        optimizer = build_optimizer(config, model)
        scheduler = build_lr_scheduler(config, optimizer)
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        else:
            scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

        start_train_epoch = max(0, int(getattr(config, "metric_boost_resume_epoch", 0)))
        if start_train_epoch:
            print(
                "Metric-boost recovery: resuming model weights at "
                f"epoch {start_train_epoch} with a freshly initialized optimizer state"
            )
        if complete_resume:
            if not os.path.isfile(check_point_path):
                raise FileNotFoundError(
                    f"Automatic resume requested but checkpoint is missing: {check_point_path}"
                )
            start_train_epoch = _load_training_checkpoint(
                check_point_path, model, optimizer, scheduler, scaler, device
            )
            print(f"Resuming complete training state from epoch {start_train_epoch}")
        elif legacy_resume_epoch >= 0:
            start_train_epoch = legacy_resume_epoch + 1
            print(
                "Resuming legacy model-only checkpoint at epoch "
                f"{start_train_epoch}; optimizer state starts fresh"
            )

        # Replicas are snapshots, so they must be created only after the final
        # model state (fresh warm-start, legacy resume, or full resume) is loaded.
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
                if retrieval_protocol.IS_LEGACY and 'IR' in config.test_modality:
                    mINP_ir, mAP_ir, cmc_ir = result_dict['IR']
                    is_best_rank_ir = (cmc_ir[0] >= best_rank1_ir)
                    # visual log
                    performance_writer.add_scalar(f'R1_IR', cmc_ir[0], current_epoch)
                    performance_writer.add_scalar(f'mAP_IR', mAP_ir, current_epoch)
                    performance_writer.add_scalar(f'mINP_IR', mINP_ir, current_epoch)
                    # new add
                    if is_best_rank_ir:
                        logger(f"New Best IR_RGB!!!")
                        best_rank1_ir = max(cmc_ir[0], best_rank1_ir)
                        best_mAP_ir = mAP_ir
                        best_mINP_ir = mINP_ir
                    logger(f"Best IR_RGB mINP: {best_mINP_ir}, Best mAP: {best_mAP_ir}, Best Rank1: {best_rank1_ir}")
                    # new add
                    model.save_model(current_epoch, is_best_rank_ir, mode="IR")
                    logger('Time: {}; Dataset: {}, Test Mode: {}, \nmINP: {} \nmAP: {} \n Rank: {}\n'.format(time_now(),
                                                                                    config.dataset,"IR_RGB",
                                                                                    mINP_ir, mAP_ir, cmc_ir))
                if not retrieval_protocol.IS_LEGACY or 'Fusion' in config.test_modality:
                    mINP_fusion, mAP_fusion, cmc_fusion = result_dict[fusion_result_key]
                    save_best_per_metric = bool(getattr(config, "save_best_per_metric", False))
                    is_best_rank_fusion = (
                        cmc_fusion[0] > best_rank1_fusion
                        if save_best_per_metric else cmc_fusion[0] >= best_rank1_fusion
                    )
                    is_best_map_fusion = save_best_per_metric and (mAP_fusion > best_mAP_fusion)
                    is_best_minp_fusion = save_best_per_metric and (mINP_fusion > best_mINP_fusion)
                    # visual log
                    performance_writer.add_scalar(f'R1_{fusion_result_key}', cmc_fusion[0], current_epoch)
                    performance_writer.add_scalar(f'mAP_{fusion_result_key}', mAP_fusion, current_epoch)
                    performance_writer.add_scalar(f'mINP_{fusion_result_key}', mINP_fusion, current_epoch)
                    # new add
                    if is_best_rank_fusion:
                        logger(f"New Best {fusion_result_key}!!!")
                        best_rank1_fusion = cmc_fusion[0]
                        if not save_best_per_metric:
                            best_mAP_fusion = mAP_fusion
                            best_mINP_fusion = mINP_fusion
                    if is_best_map_fusion:
                        best_mAP_fusion = mAP_fusion
                    if is_best_minp_fusion:
                        best_mINP_fusion = mINP_fusion
                    logger(f"Best {fusion_result_key} mINP: {best_mINP_fusion}, Best mAP: {best_mAP_fusion}, Best Rank1: {best_rank1_fusion}")
                    checkpoint_paths = dict(getattr(model, "_metric_checkpoint_paths", {}))
                    new_best_metrics = []
                    if save_best_per_metric:
                        new_best_metrics = [
                            metric for metric, improved in (
                                ("Rank-1", is_best_rank_fusion),
                                ("mAP", is_best_map_fusion),
                                ("mINP", is_best_minp_fusion),
                            ) if improved
                        ]
                        checkpoint_paths = model.save_metric_checkpoints(
                            current_epoch, new_best_metrics, mode='Fusion'
                        )
                    else:
                        model.save_model(current_epoch, is_best_rank_fusion, mode='Fusion')
                    _append_metric_event(
                        config,
                        "eval_epoch",
                        epoch=current_epoch,
                        dataset=config.dataset,
                        protocol=protocol_spec.identifier,
                        protocol_spec=protocol_spec.as_dict(),
                        query=retrieval_protocol.QUERY_NAME,
                        gallery=retrieval_protocol.GALLERY_NAME,
                        metrics={
                            "Rank-1": float(cmc_fusion[0]),
                            "mAP": float(mAP_fusion),
                            "mINP": float(mINP_fusion),
                        },
                        best_so_far={
                            "Rank-1": float(best_rank1_fusion),
                            "mAP": float(best_mAP_fusion),
                            "mINP": float(best_mINP_fusion),
                        },
                        is_new_best=bool(is_best_rank_fusion),
                        new_best_metrics=new_best_metrics,
                        checkpoint_paths=checkpoint_paths,
                    )
                    logger('Time: {}; Dataset: {}, Test Mode: {}, \nmINP: {} \nmAP: {} \n Rank: {}\n'.format(time_now(),
                                                                                    config.dataset,fusion_result_key,
                                                                                    mINP_fusion, mAP_fusion, cmc_fusion))
                if retrieval_protocol.IS_LEGACY and 'Text' in config.test_modality:
                    mINP_text, mAP_text, cmc_text = result_dict['Text']
                    is_best_rank_text = (cmc_text[0] >= best_rank1_text)
                    # visual log
                    performance_writer.add_scalar(f'R1_Text', cmc_text[0], current_epoch)
                    performance_writer.add_scalar(f'mAP_Text', mAP_text, current_epoch)
                    performance_writer.add_scalar(f'mINP_Text', mINP_text, current_epoch)
                    # new add
                    if is_best_rank_text:
                        logger(f"New Best Text_RGB!!!")
                        best_rank1_text = max(cmc_text[0], best_rank1_text)
                        best_mAP_text = mAP_text
                        best_mINP_text = mINP_text
                    logger(f"Best Text_RGB mINP: {best_mINP_text}, Best mAP: {best_mAP_text}, Best Rank1: {best_rank1_text}")
                    # new add
                    model.save_model(current_epoch, is_best_rank_text, mode='Text')
                    logger('Time: {}; Dataset: {}, Test Mode: {}, \nmINP: {} \nmAP: {} \n Rank: {}\n'.format(time_now(),
                                                                                    config.dataset,"Text_RGB",
                                                                                    mINP_text, mAP_text, cmc_text))

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
        if retrieval_protocol.IS_LEGACY and "IR" in config.test_modality:
            mINP_ir, mAP_ir, cmc_ir = result_dict['IR']
            if config.LOG4TEST:
                logger('Time: {}; Dataset: {}, Test Mode: {}, \nmINP: {} \nmAP: {} \n Rank: {}\n'.format(time_now(),
                                                                                    config.dataset,"IR_RGB",
                                                                                    mINP_ir, mAP_ir, cmc_ir))
            else:
                print('Time: {}; Dataset: {}, Test Mode: {}, \nmINP: {} \nmAP: {} \n Rank: {}\n'.format(time_now(),
                                                                                    config.dataset,"IR_RGB",
                                                                                    mINP_ir, mAP_ir, cmc_ir))

        if not retrieval_protocol.IS_LEGACY or "Fusion" in config.test_modality:
            mINP_fusion, mAP_fusion, cmc_fusion = result_dict[fusion_result_key]
            if config.LOG4TEST:
                if config.CAT_EVAL:
                    logger('===================Test with CAT FEAT===================')
                else:
                    logger('===================Test without CAT FEAT===================')
                logger('Time: {}; Dataset: {}, Test Mode: {}, \nmINP: {} \nmAP: {} \n Rank: {}\n'.format(time_now(),
                                                                                    config.dataset,fusion_result_key,
                                                                                    mINP_fusion, mAP_fusion, cmc_fusion))
            else:
                print('Time: {}; Dataset: {}, Test Mode: {}, \nmINP: {} \nmAP: {} \n Rank: {}\n'.format(time_now(),
                                                                                    config.dataset,fusion_result_key,
                                                                                    mINP_fusion, mAP_fusion, cmc_fusion))

        if retrieval_protocol.IS_LEGACY and "Text" in config.test_modality:
            mINP_text, mAP_text, cmc_text = result_dict['Text']
            if config.LOG4TEST:
                logger('Time: {}; Dataset: {}, Test Mode: {}, \nmINP: {} \nmAP: {} \n Rank: {}\n'.format(time_now(),
                                                                                    config.dataset,"Text_RGB",
                                                                                    mINP_text, mAP_text, cmc_text))
            else:
                print('Time: {}; Dataset: {}, Test Mode: {}, \nmINP: {} \nmAP: {} \n Rank: {}\n'.format(time_now(),
                                                                                    config.dataset,"Text_RGB",
                                                                                    mINP_text, mAP_text, cmc_text))


if __name__ == '__main__':
    config = _merge_runtime_config(get_args())
    # CUDA visibility must be fixed before seed_torch calls torch.cuda.* and
    # initializes the CUDA runtime.
    _configure_cuda_visibility(config)
    seed_torch(config.seed)
    main(config)
