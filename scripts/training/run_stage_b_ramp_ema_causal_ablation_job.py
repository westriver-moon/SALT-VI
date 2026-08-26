#!/usr/bin/env python3
"""Run one preregistered Ramp/EMA ablation job.

The canonical SALT-VI train loop is reused in-process. The wrapper adds only
experiment-local controls: delayed EMA activation, online evaluation beside
EMA evaluation, epoch-level diagnostics, checkpoint averaging for G7, and a
safe stop after epoch 11 while the optimizer/scheduler still see 30 epochs.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

DEFAULT_BASE_CONFIG = (
    PROJECT_ROOT
    / "configs/experiments/stage_b_ramp_ema_causal_ablation_20260825/base.yaml"
)
MAX_EPOCH_INDEX = 11
IMPLEMENTATION_VERSION = "ramp-ema-v4-20260826"


class IntentionalEarlyStop(RuntimeError):
    """Raised after the final requested checkpoint has been written."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def atomic_torch_save(path: Path, payload: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def effective_hard_weight(job: dict, epoch: int) -> float:
    """Match the project's B5 convention: five epochs include both endpoints."""
    target = float(job["hard_weight"])
    start = int(job["hard_start_epoch"])
    ramp = int(job["hard_ramp_epochs"])
    if ramp <= 1:
        return target if epoch >= start else 0.0
    return target * min(1.0, max(0.0, (epoch - start) / float(ramp - 1)))


def gradient_windows(values: list[float], width: int = 100) -> list[dict]:
    windows = []
    for start in range(0, len(values), width):
        chunk = values[start:start + width]
        mean = sum(chunk) / len(chunk)
        windows.append({
            "first_successful_step": start,
            "count": len(chunk),
            "mean": mean,
            "variance": sum((value - mean) ** 2 for value in chunk) / len(chunk),
        })
    return windows


def evaluate_preserving_rng(train_module, evaluate, *args):
    """An additional evaluation must not change the next training batch/dropout."""
    state = train_module._capture_rng_state()
    try:
        return evaluate(*args)
    finally:
        train_module._restore_rng_state(state)


def install_ema_checkpoint_writers(model, holder: dict) -> None:
    """The canonical loop saves after evaluation has restored online weights."""
    def wrap(writer):
        def save(*args, **kwargs):
            ema = holder.get("ema")
            if ema is not None and ema.active:
                with ema.average_parameters(model):
                    return writer(*args, **kwargs)
            return writer(*args, **kwargs)
        return save
    model.save_metric_checkpoints = wrap(model.save_metric_checkpoints)
    model.save_model = wrap(model.save_model)


def save_best_rank1(model, metrics: dict, model_type: str, holder: dict, job_dir: Path) -> Path:
    path = job_dir / "checkpoints" / f"{model_type}_best.pth"
    key = f"best_{model_type}_rank1"
    if metrics["Rank-1"] > holder.get(key, -math.inf):
        atomic_torch_save(path, {name: value.detach().cpu() for name, value in model.state_dict().items()})
        holder[key] = metrics["Rank-1"]
    return path


def finite_float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def metric_from_result(result_dict: dict, protocol) -> dict[str, float]:
    key = protocol.RESULT_KEY
    if key not in result_dict:
        raise RuntimeError(f"evaluation returned no {key} result")
    m_inp, m_ap, cmc = result_dict[key]
    return {
        "Rank-1": float(cmc[0]),
        "mAP": float(m_ap),
        "mINP": float(m_inp),
    }


def global_grad_norm(optimizer) -> float | None:
    squared = 0.0
    found = False
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            gradient = parameter.grad
            if gradient is None:
                continue
            value = gradient.detach().float()
            if not bool(torch.isfinite(value).all()):
                return None
            squared += float(value.pow(2).sum().item())
            found = True
    return math.sqrt(squared) if found and math.isfinite(squared) else None


class TrackingScaler:
    """Delegate GradScaler while sampling the unscaled gradient norm."""

    def __init__(self, scaler, holder: dict):
        self._scaler = scaler
        self._holder = holder

    def step(self, optimizer):
        value = global_grad_norm(optimizer)
        if value is not None:
            self._holder.setdefault("gradient_norms", []).append(value)
        result = self._scaler.step(optimizer)
        ema = self._holder.get("ema")
        model = self._holder.get("model")
        criterion = getattr(model, "_ramp_ema_stats_criterion", None)
        if criterion is not None:
            criterion.mark_step(value is not None)
        if value is not None and ema is not None and model is not None and ema.active:
            ema.update(model)
        return result

    def __getattr__(self, name):
        return getattr(self._scaler, name)


class StatsCrossModalCriterion(torch.nn.Module):
    """Keep the original scalar loss and expose its native d+/d- statistics."""

    def __init__(self, inner, pair_weights):
        super().__init__()
        from salt_vi.training.common import CROSS_MODAL_PAIR_NAMES, resolve_cross_modal_pair_weights

        self.inner = inner
        self.pair_names = CROSS_MODAL_PAIR_NAMES
        self.pair_weights = resolve_cross_modal_pair_weights(pair_weights)
        self.records: list[dict[str, float]] = []

    def forward(self, visible_feats, ir_feats, labels):
        loss, stats = self.inner(
            visible_feats, ir_feats, labels, return_stats=True
        )
        self.records.append(
            {
                "pair": self.pair_names[len(self.records) % len(self.pair_names)],
                "successful_step": True,
                "raw_hard_loss": float(loss.detach().item()),
                "d_plus": float(stats["pos_dist"].detach().item()),
                "d_minus": float(stats["neg_dist"].detach().item()),
                "valid_anchors": 2 * len(labels) * float(stats["valid_anchor_ratio"].item()),
            }
        )
        return loss

    def mark_step(self, successful: bool) -> None:
        for item in self.records[-len(self.pair_names):]:
            item["successful_step"] = successful

    def consume(self) -> dict[str, float | None]:
        records = self.records
        self.records = []
        total_calls = len(records)
        records = [item for item in records if item["successful_step"]]
        if not records:
            return {
                "raw_hard_loss": None,
                "d_plus": None,
                "d_minus": None,
                "hard_gap": None,
                "sample_count": 0.0,
            }
        if len(records) % len(self.pair_names):
            raise RuntimeError("incomplete six-pair hard-loss diagnostic batch")
        denominator = sum(self.pair_weights[item["pair"]] for item in records)
        values = {
            key: sum(item[key] * self.pair_weights[item["pair"]] for item in records) / denominator
            for key in ("raw_hard_loss", "d_plus", "d_minus")
        }
        values["hard_gap"] = values["d_minus"] - values["d_plus"]
        values["sample_count"] = sum(item["valid_anchors"] for item in records if self.pair_weights[item["pair"]] > 0)
        values["criterion_calls"] = len(records)
        values["batch_count"] = len(records) // len(self.pair_names)
        values["forward_batch_count"] = total_calls // len(self.pair_names)
        values["skipped_batch_count"] = (total_calls - len(records)) // len(self.pair_names)
        values["pair_weights"] = self.pair_weights
        return values


@torch.no_grad()
def fixed_probe_stats(model, batches, pair_weights):
    """Same cached training images/identities each epoch, eval-mode features.

    This diagnostic is separate from training-batch statistics. It never calls
    the classifier, never updates BN, and uses the recipe's pre-classifier
    RGB/IR/Text embeddings and the same hard-mining criterion.
    """
    from salt_vi.training.recipes import _encode_batch
    from salt_vi.training.common import extract_text_token_feat, resolve_cross_modal_pair_weights

    if not batches:
        raise RuntimeError("fixed diagnostic probe captured no training batches")
    weights = resolve_cross_modal_pair_weights(pair_weights)
    active = {pair: weight for pair, weight in weights.items() if weight > 0}
    if any("Fusion" in pair for pair in active):
        raise ValueError("this registered probe requires the three RGB/IR/Text pairs")
    flags = [(module, module.training) for module in model.modules()]
    total = {"raw_hard_loss": 0., "d_plus": 0., "d_minus": 0.}
    criterion = model._ramp_ema_stats_criterion.inner
    try:
        model.eval()
        with torch.cuda.amp.autocast(enabled=model.device.type == "cuda"):
            for cached in batches:
                batch = {name: value.to(model.device) for name, value in cached.items()}
                context = _encode_batch(model, batch, "1/3")
                count = context.batch_size
                features = {
                    "RGB": (context.rgb_feats[:count] + context.rgb_feats[count:]) * .5,
                    "IR": context.ir_feats,
                    "Text": extract_text_token_feat(model.base_model.encode_text(batch["text_rgb"]), batch["text_rgb"]),
                }
                for pair, weight in active.items():
                    left, right = pair.split("-")
                    loss, stats = criterion(features[left], features[right], context.label_rgb, return_stats=True)
                    total["raw_hard_loss"] += weight * float(loss)
                    total["d_plus"] += weight * float(stats["pos_dist"])
                    total["d_minus"] += weight * float(stats["neg_dist"])
    finally:
        for module, training in flags:
            module.training = training
    denominator = len(batches) * sum(active.values())
    values = {name: value / denominator for name, value in total.items()}
    values["hard_gap"] = values["d_minus"] - values["d_plus"]
    values["batch_count"] = len(batches)
    return values


class DelayedModelEMA:
    """ModelEMA-compatible object with an explicit epoch activation point."""

    def __init__(self, model, decay):
        self.decay = float(decay)
        self.start_epoch = int(getattr(model.args, "ema_start_epoch", 0))
        self.updates = 0
        self.active = False
        self.current_epoch = -1
        self.shadow = {}
        self.frozen_names = {name for name, value in model.named_parameters() if not value.requires_grad}

    @torch.no_grad()
    def set_epoch(self, epoch, model):
        self.current_epoch = int(epoch)
        if self.active or self.current_epoch < self.start_epoch:
            return
        self.shadow = {
            name: value.detach().float().clone() if value.is_floating_point() else value.detach().clone()
            for name, value in model.state_dict().items()
        }
        self.updates = 0
        self.active = True

    @torch.no_grad()
    def update(self, model):
        if not self.active:
            return
        self.updates += 1
        for name, value in model.state_dict().items():
            if value.is_floating_point() and name not in self.frozen_names:
                self.shadow[name].mul_(self.decay).add_(
                    value.detach(), alpha=1.0 - self.decay
                )
            else:
                self.shadow[name].copy_(value)

    def state_dict(self):
        return {
            "decay": self.decay,
            "updates": self.updates,
            "active": self.active,
            "start_epoch": self.start_epoch,
            "shadow": {name: value.detach().cpu() for name, value in self.shadow.items()},
        }

    def load_state_dict(self, state, model):
        self.decay = float(state["decay"])
        self.updates = int(state["updates"])
        self.active = bool(state.get("active", True))
        self.start_epoch = int(state.get("start_epoch", 0))
        device_by_name = {name: value.device for name, value in model.state_dict().items()}
        self.shadow = {
            name: value.to(device_by_name[name]) for name, value in state["shadow"].items()
        }

    @contextmanager
    def average_parameters(self, model):
        if not self.active:
            yield
            return
        # Keep the temporary online backup off GPU during evaluation/save.
        live = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        try:
            model.load_state_dict(self.shadow)
            yield
        finally:
            model.load_state_dict(live)


def ema_distance(model, ema) -> float | None:
    if ema is None or not ema.active:
        return None
    squared = 0.0
    count = 0
    for name, value in model.named_parameters():
        if not value.requires_grad:
            continue
        shadow = ema.shadow[name]
        delta = value.detach().float() - shadow.detach().float()
        squared += float(delta.pow(2).sum().item())
        count += int(delta.numel())
    if count == 0 or not math.isfinite(squared):
        return None
    return math.sqrt(squared / count)


def average_states(states: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not states:
        raise RuntimeError("checkpoint averaging has no saved states")
    keys = states[-1].keys()
    result = {}
    for key in keys:
        values = [state[key] for state in states]
        if values[-1].is_floating_point():
            value = values[-1].clone().float()
            for item in values[:-1]:
                value.add_(item.float())
            value.div_(float(len(values)))
            result[key] = value.to(dtype=values[-1].dtype)
        else:
            result[key] = values[-1].clone()
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args(argv)


def load_job(manifest_path: Path, job_id: str) -> dict:
    import yaml

    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    jobs = {str(item["id"]): item for item in payload.get("jobs", [])}
    if job_id not in jobs:
        raise KeyError(f"unknown job id: {job_id}")
    job = dict(jobs[job_id])
    job["training"] = dict(payload["training"])
    job["protocol"] = payload["protocol"]
    job["stage_a_checkpoint"] = payload["stage_a"]["checkpoint"]
    job["implementation_version"] = payload["implementation_version"]
    if job["implementation_version"] != IMPLEMENTATION_VERSION:
        raise ValueError("manifest and runner implementation versions differ")
    if job["training"]["completed_epochs"] != MAX_EPOCH_INDEX + 1 or job["training"]["last_completed_epoch_index"] != MAX_EPOCH_INDEX:
        raise ValueError("runner requires exactly 12 completed epochs (indices 0..11)")
    return job


def classify_event_model(event: dict, job: dict) -> str:
    if event.get("event_type") == "eval_epoch_model":
        return str(event.get("model_type", "online"))
    if not bool(job.get("ema", False)):
        return "online"
    epoch = int(event.get("epoch", -1))
    start = int(job.get("ema_start_epoch", 0))
    return "ema" if epoch >= start and epoch >= 0 else "online"


def collect_best_metrics(events_path: Path, job: dict, max_epoch: int) -> dict:
    by_model: dict[str, list[dict]] = {}
    for line in events_path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("event_type") not in {"eval_epoch", "eval_epoch_model"}:
            continue
        epoch = int(event.get("epoch", -1))
        if epoch < 0 or epoch > max_epoch:
            continue
        metrics = event.get("metrics") or {}
        values = {key: finite_float(metrics.get(key)) for key in ("Rank-1", "mAP", "mINP")}
        if any(value is None for value in values.values()):
            continue
        model_name = classify_event_model(event, job)
        by_model.setdefault(model_name, []).append(
            {"epoch": epoch, **{key: float(value) for key, value in values.items()},
             "checkpoint_path": (event.get("checkpoint_paths") or {}).get("Rank-1")}
        )

    best = {}
    for model_name, rows in by_model.items():
        best[model_name] = dict(max(rows, key=lambda row: row["Rank-1"]))
        if model_name in {"online", "ema"}:
            # Keep trained-model selection independent of the warm-start score
            # and late EMA selection independent of its online warmup.
            best[model_name]["checkpoint_path"] = str(events_path.parent / "checkpoints" / f"{model_name}_best.pth")

    primary_model = (
        "checkpoint_average" if bool(job.get("checkpoint_average", False)) else
        ("ema" if bool(job.get("ema", False)) else "online")
    )
    required_models = {"online", primary_model}
    if not required_models.issubset(best):
        raise RuntimeError(f"missing required evaluation models: {sorted(required_models - set(best))}")
    return {"best_by_model": best, "primary_model": primary_model}


def build_config(train_module, config_path: Path, job: dict, job_dir: Path, gpu: int):
    from salt_vi.config.config_rn import get_args

    cli = get_args(["--config_select", str(config_path)])
    config = train_module._merge_runtime_config(cli)
    config.output_root = str(job_dir / "model_output")
    config.metric_events_path = str(job_dir / "events.jsonl")
    config.metric_experiment_id = f"SALTVI-STAGEB-RAMP-EMA-{job['id']}"
    config.training_weight_init = str(job["stage_a_checkpoint"])
    config.CUDA_VISIBLE_DEVICES = str(gpu)
    config.gpu_id = "0"
    config.seed = int(job["seed"])
    config.mode = "train"
    config.auto_resume_training_from_lastest_step = False
    config.resume_train_epoch = -1
    config.total_train_epoch = int(job["training"]["planned_epochs"])
    config.checkpoint_epoch = 1
    config.eval_before_train = True
    config.eval_start_epoch = 1
    config.eval_epoch = 1
    config.gallery_trials = 10
    config.save_best_per_metric = True
    config.max_save_model_num = 1
    config.cross_modal_hard_weight = float(job["hard_weight"])
    config.cross_modal_hard_start_epoch = int(job["hard_start_epoch"])
    config.cross_modal_hard_ramp_epochs = int(job["hard_ramp_epochs"])
    config.ema_enabled = bool(job.get("ema", False))
    config.ema_decay = float(job["training"]["ema_decay"])
    config.ema_start_epoch = int(job.get("ema_start_epoch", 0))
    return config


def run_job(args) -> int:
    # Set physical visibility before CUDA initialization or entrypoint imports.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    import importlib
    import torch

    train_module = importlib.import_module("salt_vi.entrypoints.train")
    manifest_path = args.manifest.resolve()
    job = load_job(manifest_path, args.job_id)
    output_root = args.output_root.resolve()
    job_dir = output_root / "jobs" / args.job_id
    job_dir.mkdir(parents=True, exist_ok=False)
    results_dir = job_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    events_path = job_dir / "events.jsonl"
    holder: dict = {"recent_states": [], "gradient_norms": [], "probe_batches": []}

    if not Path(job["stage_a_checkpoint"]).is_file():
        raise FileNotFoundError(f"Stage-A checkpoint is missing: {job['stage_a_checkpoint']}")

    config = build_config(train_module, DEFAULT_BASE_CONFIG, job, job_dir, args.gpu)
    atomic_json(job_dir / "resolved_config.json", dict(config) if isinstance(config, dict) else vars(config))
    atomic_json(job_dir / "resolved_job.json", job)
    protocol = train_module.get_retrieval_protocol(config.retrieval_backend)
    original_train = train_module.train
    original_test = train_module.test
    original_save = train_module._save_training_checkpoint

    def train_with_diagnostics(base, loaders, scaler, config, optimizer, current_epoch=None):
        holder["current_epoch"] = int(current_epoch)
        hard_weight = effective_hard_weight(job, int(current_epoch))
        # The pinned base revision has no ramp support. Its recipe reads args
        # directly, so setting configuration fields alone previously did nothing.
        base.args.cross_modal_hard_weight = hard_weight
        ema = holder.get("ema")
        if bool(job.get("ema", False)):
            if ema is None:
                ema = DelayedModelEMA(base, config.ema_decay)
                holder["ema"] = ema
            ema.set_epoch(int(current_epoch), base)
        holder["model"] = base
        holder["loaders"] = loaders
        holder["config"] = config
        holder["gradient_norms"] = []
        if not hasattr(base, "_ramp_ema_stats_criterion"):
            base.cross_modal_tri_criterion = StatsCrossModalCriterion(
                base.cross_modal_tri_criterion, config.cross_modal_pair_weights
            )
            base._ramp_ema_stats_criterion = base.cross_modal_tri_criterion
            install_ema_checkpoint_writers(base, holder)
            def capture_probe(model, inputs):
                if len(holder["probe_batches"]) < int(job["training"]["diagnostic_probe_batches"]):
                    batch = inputs[0]
                    holder["probe_batches"].append({key: value.detach().cpu().clone() for key, value in batch.items()})
            holder["probe_hook"] = base.register_forward_pre_hook(capture_probe)
        train_started = time.monotonic()
        result = original_train(
            base,
            loaders,
            TrackingScaler(scaler, holder),
            config,
            optimizer,
            current_epoch=current_epoch,
        )
        pure_train_seconds = time.monotonic() - train_started
        if holder.get("probe_hook") is not None:
            holder.pop("probe_hook").remove()
        keys, values = result[0]
        scalar_values = {}
        for key, value in zip(keys, values):
            converted = finite_float(value)
            if converted is not None:
                scalar_values[key] = converted
        hard_stats = base._ramp_ema_stats_criterion.consume()
        gradients = holder.get("gradient_norms", [])
        grad_mean = sum(gradients) / len(gradients) if gradients else None
        grad_var = (
            sum((value - grad_mean) ** 2 for value in gradients) / len(gradients)
            if gradients and grad_mean is not None
            else None
        )
        diagnostic = {
            **hard_stats,
            "weighted_hard_loss": scalar_values.get("cross_modal_hard_loss"),
            "training_duration_seconds": pure_train_seconds,
            "gradient_norm_mean": grad_mean,
            "gradient_norm_variance": grad_var,
            "gradient_norm_samples": float(len(gradients)),
            "gradient_norm_windows": gradient_windows(gradients),
            "effective_hard_weight": hard_weight,
            "ema_updates": ema.updates if ema is not None else 0,
            "online_ema_distance": ema_distance(base, ema),
            "online_ema_distance_definition": "RMS over trainable parameters; excludes frozen parameters and buffers",
        }
        probe_started = time.monotonic()
        diagnostic["fixed_probe_online"] = evaluate_preserving_rng(
            train_module, fixed_probe_stats, base, holder["probe_batches"], config.cross_modal_pair_weights)
        if ema is not None and ema.active:
            with ema.average_parameters(base):
                diagnostic["fixed_probe_ema"] = evaluate_preserving_rng(
                    train_module, fixed_probe_stats, base, holder["probe_batches"], config.cross_modal_pair_weights)
        diagnostic["probe_duration_seconds"] = time.monotonic() - probe_started
        diagnostic["fixed_probe_definition"] = "first four epoch-0 training batches, cached augmentations, eval-mode pre-classifier embeddings"
        train_module._append_metric_event(
            config,
            "diagnostic_epoch",
            epoch=int(current_epoch),
            diagnostics=diagnostic,
        )

        # test_with_ema supplies the later canonical evaluation. This additional
        # online evaluation must leave all training RNG streams unchanged.
        if ema is not None and ema.active:
            online_result = evaluate_preserving_rng(
                train_module, original_test, base, loaders, config, base.device
            )
            online_metrics = metric_from_result(online_result, protocol)
            online_path = save_best_rank1(base, online_metrics, "online", holder, job_dir)
            train_module._append_metric_event(
                config,
                "eval_epoch_model",
                epoch=int(current_epoch),
                model_type="online",
                dataset=config.dataset,
                protocol=job["protocol"],
                retrieval_backend=protocol.NAME,
                metrics=online_metrics,
                checkpoint_paths={"Rank-1": str(online_path)},
            )
        return result

    train_module.train = train_with_diagnostics

    def test_with_ema(base, loaders, config, device, *test_args, **test_kwargs):
        ema = holder.get("ema")
        if ema is not None and ema.active:
            with ema.average_parameters(base):
                result = original_test(
                    base, loaders, config, device, *test_args, **test_kwargs
                )
                save_best_rank1(base, metric_from_result(result, protocol), "ema", holder, job_dir)
                return result
        result = original_test(base, loaders, config, device, *test_args, **test_kwargs)
        if holder.get("current_epoch", -1) >= 0:
            save_best_rank1(base, metric_from_result(result, protocol), "online", holder, job_dir)
        return result

    train_module.test = test_with_ema

    def save_and_stop(path, epoch, model, optimizer, scheduler, scaler, **kwargs):
        saved = original_save(path, epoch, model, optimizer, scheduler, scaler, **kwargs)
        ema = holder.get("ema")
        if ema is not None:
            checkpoint = train_module._load_trusted_training_checkpoint(path)
            checkpoint["ema"] = ema.state_dict()
            checkpoint["ablation_implementation"] = IMPLEMENTATION_VERSION
            atomic_torch_save(Path(path), checkpoint)
        if bool(job.get("checkpoint_average", False)):
            holder["recent_states"].append(
                {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            )
            keep_k = int(job["training"]["checkpoint_average_k"])
            if len(holder["recent_states"]) > keep_k:
                del holder["recent_states"][:-keep_k]
        holder["last_completed_epoch"] = int(epoch)
        if int(epoch) == MAX_EPOCH_INDEX:
            raise IntentionalEarlyStop(
                f"completed {MAX_EPOCH_INDEX + 1} epochs with 30-epoch schedule"
            )
        return saved

    train_module._save_training_checkpoint = save_and_stop

    try:
        train_module._configure_cuda_visibility(config)
        train_module.seed_torch(config.seed)
        try:
            train_module.main(config)
        except IntentionalEarlyStop:
            pass
        if holder.get("last_completed_epoch") != MAX_EPOCH_INDEX:
            raise RuntimeError("training ended before saving epoch 11")

        if bool(job.get("checkpoint_average", False)):
            model = holder.get("model")
            loaders = holder.get("loaders")
            if model is None or loaders is None:
                raise RuntimeError("checkpoint averaging lost the final model context")
            averaged = average_states(holder["recent_states"])
            average_path = job_dir / "checkpoints" / "checkpoint_average.pth"
            atomic_torch_save(average_path, averaged)
            model.load_state_dict(averaged)
            # Do not call test_with_ema: it also maintains online_best and an
            # improved averaged model must never overwrite the online weights.
            average_result = original_test(model, loaders, config, model.device)
            average_metrics = metric_from_result(average_result, protocol)
            train_module._append_metric_event(
                config,
                "eval_epoch_model",
                epoch=MAX_EPOCH_INDEX,
                model_type="checkpoint_average",
                dataset=config.dataset,
                protocol=job["protocol"],
                retrieval_backend=protocol.NAME,
                metrics=average_metrics,
                checkpoint_paths={"Rank-1": str(average_path)},
                averaged_epochs=list(range(MAX_EPOCH_INDEX + 1 - len(holder["recent_states"]), MAX_EPOCH_INDEX + 1)),
            )

        collected = collect_best_metrics(events_path, job, MAX_EPOCH_INDEX)
        best_by_model = collected["best_by_model"]
        for values in best_by_model.values():
            if not values.get("checkpoint_path") or not Path(values["checkpoint_path"]).is_file():
                raise RuntimeError(f"selected metrics have no saved weights: {values}")
        primary_model = collected["primary_model"]
        primary = best_by_model[primary_model]
        metrics_payload = {
            "primary_metric": float(primary["Rank-1"]),
            "metrics": {
                "Rank-1": float(primary["Rank-1"]),
                "mAP": float(primary["mAP"]),
                "mINP": float(primary["mINP"]),
                "best_epoch": float(primary["epoch"]),
                "selected_gpu": float(args.gpu),
                "seed": float(job["seed"]),
                "planned_epochs": float(job["training"]["planned_epochs"]),
                "completed_epochs": float(MAX_EPOCH_INDEX + 1),
                "ema_enabled": 1.0 if bool(job.get("ema", False)) else 0.0,
            },
        }
        for model_name, values in best_by_model.items():
            prefix = model_name.replace("-", "_")
            for key in ("Rank-1", "mAP", "mINP"):
                metrics_payload["metrics"][f"{prefix}_{key}"] = float(values[key])
            metrics_payload["metrics"][f"{prefix}_best_epoch"] = float(values["epoch"])
        atomic_json(results_dir / "metrics.json", metrics_payload)
        job_result = {
            "schema_version": 2,
            "implementation_version": IMPLEMENTATION_VERSION,
            "job_id": args.job_id,
            "phase": job["phase"],
            "group": job["group"],
            "seed": int(job["seed"]),
            "gpu": int(args.gpu),
            "status": "completed",
            "primary_model": primary_model,
            "best_by_model": best_by_model,
            "metrics_path": str(results_dir / "metrics.json"),
            "events_path": str(events_path),
            "completed_epochs": MAX_EPOCH_INDEX + 1,
            "planned_epochs": int(job["training"]["planned_epochs"]),
            "hard_mode": job["hard_mode"],
            "ema_start_epoch": int(job.get("ema_start_epoch", 0)),
        }
        atomic_json(job_dir / "job_result.json", job_result)
        return 0
    except Exception as exc:
        atomic_json(
            job_dir / "job_result.json",
            {
                "schema_version": 2,
                "job_id": args.job_id,
                "phase": job.get("phase"),
                "group": job.get("group"),
                "seed": job.get("seed"),
                "gpu": args.gpu,
                "status": "failed",
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise


def main(argv=None) -> int:
    args = parse_args(argv)
    # Imported lazily so the process-level visibility is applied first.
    return run_job(args)


if __name__ == "__main__":
    raise SystemExit(main())
