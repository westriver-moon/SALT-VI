#!/usr/bin/env python3
"""Run provenance-bound warm-start and 20-step RTX 3090 gates for one SYSU SR config."""

import argparse
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from torch.cuda import amp

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from salt_vi.engine import build_model, test, train
from salt_vi.data.loader import Loader
from salt_vi.entrypoints.train import _load_fixed_visual_init, seed_torch
from salt_vi.optim import build_lr_scheduler, build_optimizer
from salt_vi.utils.super_resolution.provenance import (
    assert_clean_algorithm_source,
    build_preflight_provenance,
    provenance_matches,
)
from salt_vi.utils.utils import load_train_configs


METRIC_KEYS = ("Rank-1", "mAP", "mINP")


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


class LimitedLoaders:
    def __init__(self, loaders, steps):
        self.loaders = loaders
        self.steps = steps

    def get_train_loader(self):
        return itertools.islice(self.loaders.get_train_loader(), self.steps)


def serialize_fusion_metrics(fusion):
    if not isinstance(fusion, (tuple, list)) or len(fusion) != 3:
        raise ValueError(f"Expected Fusion=(mINP,mAP,CMC), got {type(fusion)}")
    minp, mean_ap, cmc = fusion
    cmc = np.asarray(cmc, dtype=np.float64).reshape(-1)
    values = np.concatenate(([float(minp), float(mean_ap)], cmc))
    if cmc.size < 1 or not np.isfinite(values).all():
        raise FloatingPointError("Fusion retrieval metrics are missing or non-finite")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("Fusion retrieval metrics are outside [0, 1]")
    return {
        "Rank-1": float(cmc[0]),
        "mAP": float(mean_ap),
        "mINP": float(minp),
        "cmc_length": int(cmc.size),
        "cmc_last": float(cmc[-1]),
    }


def extract_fusion_metrics(result):
    return serialize_fusion_metrics(result.get("Fusion_RGB") or result.get("Fusion"))


def metric_floor_errors(metrics, floors, label):
    errors = []
    for key in METRIC_KEYS:
        if key in floors and float(metrics[key]) < float(floors[key]):
            errors.append(
                f"{label} {key} {metrics[key]:.6f} is below required {float(floors[key]):.6f}"
            )
    return errors


def metric_drop_errors(metrics, reference, maximum_drops, label):
    errors = []
    for key in METRIC_KEYS:
        if key not in maximum_drops:
            continue
        drop = float(reference[key]) - float(metrics[key])
        if drop > float(maximum_drops[key]):
            errors.append(
                f"{label} {key} drop {drop:.6f} exceeds {float(maximum_drops[key]):.6f} "
                f"(reference={float(reference[key]):.6f}, observed={float(metrics[key]):.6f})"
            )
    return errors


def load_reference_metrics(path):
    if path is None:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not payload.get("valid"):
        raise ValueError(f"A0 reference preflight is not valid: {path}")
    return payload["selected_attempt"]["warm_start_metrics"]


def timed_evaluation(model, loaders, config, device):
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    metrics = extract_fusion_metrics(test(model, loaders, config, device))
    torch.cuda.synchronize(device)
    return {
        "metrics": metrics,
        "peak_bytes": int(torch.cuda.max_memory_allocated(device)),
        "seconds": float(time.monotonic() - started),
    }


def prepare_epoch_zero_training(model, scheduler):
    """Apply the same epoch-0 trainability and LR controls as main.py."""
    summary = model.configure_epoch_trainability(0)
    scheduler.step(0)
    return summary


def worker(config_path, steps, chunk_size, test_batch_size, output_path, reference_preflight):
    provenance = build_preflight_provenance(
        config_path, REPO_ROOT, reference_preflight=reference_preflight
    )
    config = load_train_configs(str(config_path))
    config.visual_forward_chunk_size = chunk_size
    config.test_batch_size = test_batch_size
    config.metric_events_path = None
    seed_torch(config.seed)
    device = torch.device("cuda:0")
    loaders = Loader(config)
    model = build_model(config)
    _load_fixed_visual_init(model, config, device)
    model = model.to(device)
    optimizer = build_optimizer(config, model)
    scheduler = build_lr_scheduler(config, optimizer)
    scaler = amp.GradScaler()

    torch.cuda.empty_cache()
    warm_start = timed_evaluation(model, loaders, config, device)
    # Keep the 20-step smoke independent of any RNG consumed by evaluation.
    seed_torch(config.seed)
    prepare_epoch_zero_training(model, scheduler)

    torch.cuda.reset_peak_memory_stats(device)
    train_started = time.monotonic()
    train(model, LimitedLoaders(loaders, steps), scaler, config, optimizer, current_epoch=0)
    torch.cuda.synchronize(device)
    train_peak = int(torch.cuda.max_memory_allocated(device))
    train_seconds = float(time.monotonic() - train_started)

    post_train = timed_evaluation(model, loaders, config, device)
    gate_errors = []
    floors = dict(getattr(config, "preflight_min_warmstart_metrics", {}) or {})
    gate_errors.extend(metric_floor_errors(warm_start["metrics"], floors, "warm-start"))
    reference_metrics = load_reference_metrics(reference_preflight)
    reference_drops = dict(getattr(config, "preflight_max_reference_drop", {}) or {})
    if reference_drops and reference_metrics is None:
        gate_errors.append("A0 reference preflight is required for this group")
    elif reference_metrics is not None:
        gate_errors.extend(metric_drop_errors(
            warm_start["metrics"], reference_metrics, reference_drops, "warm-start versus A0"
        ))
    post_drops = dict(getattr(config, "preflight_max_post_train_drop", {}) or {})
    gate_errors.extend(metric_drop_errors(
        post_train["metrics"], warm_start["metrics"], post_drops, "post-20-step versus warm-start"
    ))

    peak = max(warm_start["peak_bytes"], train_peak, post_train["peak_bytes"])
    result = {
        "valid": not gate_errors,
        "provenance": provenance,
        "config": str(config_path.resolve()),
        "steps": int(steps),
        "visual_forward_chunk_size": int(chunk_size),
        "test_batch_size": int(test_batch_size),
        "warm_start_metrics": warm_start["metrics"],
        "post_train_metrics": post_train["metrics"],
        "reference_metrics": reference_metrics,
        "warm_start_eval_peak_bytes": warm_start["peak_bytes"],
        "train_peak_bytes": train_peak,
        "post_train_eval_peak_bytes": post_train["peak_bytes"],
        "peak_bytes": int(peak),
        "warm_start_eval_seconds": warm_start["seconds"],
        "train_seconds": train_seconds,
        "post_train_eval_seconds": post_train["seconds"],
        "errors": gate_errors,
    }
    atomic_json(output_path, result)
    if gate_errors:
        raise SystemExit("; ".join(gate_errors))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reference-preflight", type=Path)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--max-gib", type=float, default=22.0)
    parser.add_argument("--gpu-index", type=int, default=0,
                        help="Physical GPU used by the isolated worker (worker still sees cuda:0).")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--test-batch-size", type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    assert_clean_algorithm_source(REPO_ROOT)
    config_path = args.config.resolve()
    reference_preflight = args.reference_preflight.resolve() if args.reference_preflight else None
    config = load_train_configs(str(config_path))
    output = args.output or (
        REPO_ROOT / "reports" / "super_resolution" / "preflight" / f"{config.metric_experiment_id}.json"
    )
    output = output.resolve()
    if args.worker:
        worker(
            config_path, args.steps, args.chunk_size, args.test_batch_size,
            output, reference_preflight,
        )
        return

    expected_provenance = build_preflight_provenance(
        config_path, REPO_ROOT, reference_preflight=reference_preflight
    )
    attempts = []
    limit = int(args.max_gib * 1024 ** 3)
    for chunk_size, test_batch_size in ((16, 8), (8, 4), (4, 2)):
        attempt_output = output.with_name(f"{output.stem}.chunk{chunk_size}.test{test_batch_size}.json")
        log_path = attempt_output.with_suffix(".log")
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--config", str(config_path),
            "--output", str(attempt_output),
            "--steps", str(args.steps),
            "--chunk-size", str(chunk_size),
            "--test-batch-size", str(test_batch_size),
        ]
        if reference_preflight is not None:
            command.extend(["--reference-preflight", str(reference_preflight)])
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
        with open(log_path, "w", encoding="utf-8") as log:
            completed = subprocess.run(
                command, cwd=str(REPO_ROOT), env=env, stdout=log, stderr=subprocess.STDOUT
            )
        attempt = {
            "chunk_size": chunk_size,
            "test_batch_size": test_batch_size,
            "returncode": completed.returncode,
            "log": str(log_path),
        }
        if attempt_output.exists():
            attempt.update(json.loads(attempt_output.read_text(encoding="utf-8")))
            provenance_ok = provenance_matches(attempt, expected_provenance)
            if not provenance_ok:
                attempt.setdefault("errors", []).append("preflight provenance mismatch")
            if (
                completed.returncode == 0
                and attempt.get("valid")
                and provenance_ok
                and attempt["peak_bytes"] <= limit
            ):
                final = {
                    "valid": True,
                    "provenance": expected_provenance,
                    "physical_gpu_index": args.gpu_index,
                    "selected_chunk_size": chunk_size,
                    "selected_test_batch_size": test_batch_size,
                    "max_peak_bytes": limit,
                    "selected_attempt": attempt,
                    "attempts": attempts + [attempt],
                }
                atomic_json(output, final)
                print(output)
                return
        attempts.append(attempt)
    atomic_json(output, {
        "valid": False,
        "provenance": expected_provenance,
        "max_peak_bytes": limit,
        "attempts": attempts,
    })
    raise SystemExit("No 3090-safe and metric-valid preflight combination passed")


if __name__ == "__main__":
    main()
