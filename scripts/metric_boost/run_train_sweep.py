#!/usr/bin/env python
"""Prepare or foreground-run the ordered TRAIN-1..TRAIN-8 sweep."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import (
    E4_CONFIG_PATH,
    GpuLease,
    REPORT_ROOT,
    atomic_write_json,
    base_status,
    experiment_dir,
    experiment_status_path,
    idle_gpu_ids,
    load_yaml,
    prepare_runtime_config,
    query_gpu_states,
    read_json,
    resolve_e4_checkpoint,
    utc_now,
    validate_experiment_manifest,
)


MAIN = REPO_ROOT / "scripts" / "train.py"
EVALUATOR = SCRIPT_DIR / "eval_engine.py"


def _experiment(experiment_id, stage, overrides, validity="training experiment", dependency=None):
    return {"id": experiment_id, "stage": stage, "overrides": dict(overrides), "validity": validity, "dependency": dependency}


def build_train_plan() -> List[Dict[str, Any]]:
    base = {
        "mode": "train",
        "dataset": "sysu",
        "test_mode": "all",
        "gall_mode": "single",
        "test_modality": "Fusion",
        "gallery_trials": 10,
        "Fix_Visual": True,
        "CAT_EVAL": False,
        "test_flip_tta": False,
        "test_multi_scale": [[288, 144]],
        "rerank": False,
        "ensemble_mode": "none",
        "learnable_pa": False,
        "visual_unfreeze_last_n_blocks": 0,
        "label_smoothing": 0.0,
        "cross_modal_hard_weight": 1.0,
        "visual_pooling": "cls",
    }
    plan = []
    for blocks, visual_lr in ((2, 1e-6), (2, 3e-6), (4, 1e-6), (4, 3e-6)):
        name = f"TRAIN-1-U{len(plan)+1}"
        plan.append(_experiment(name, "TRAIN-1", dict(base, visual_unfreeze_last_n_blocks=blocks, visual_unfreeze_start_epoch=3, visual_lr=visual_lr), dependency="EVAL-phase-complete"))
    for pa in (0.3, 0.4, 0.5, 0.6, 0.7):
        plan.append(_experiment(f"TRAIN-2-pa-{str(pa).replace('.', 'p')}", "TRAIN-2", dict(base, pa=pa), dependency="TRAIN-1"))
    plan.append(_experiment("TRAIN-2-learnable-pa", "TRAIN-2", dict(base, learnable_pa=True, pa_init=0.5, pa=0.5), dependency="TRAIN-1"))
    hard_variants = [
        ("H0", "id,wrt", 0.0),
        ("H1", "id,cross_modal_hard", 1.0),
        ("H2", "id,wrt,cross_modal_hard", 0.25),
        ("H3", "id,wrt,cross_modal_hard", 0.5),
    ]
    for name, losses, weight in hard_variants:
        plan.append(_experiment(f"TRAIN-3-{name}", "TRAIN-3", dict(base, loss_names=losses, cross_modal_hard_weight=weight), dependency="TRAIN-2"))
    for seed in (0, 1, 42):
        plan.append(_experiment(f"TRAIN-4-seed-{seed}", "TRAIN-4", dict(base, seed=seed, selection_pending=True), "exploratory replicate if selected without independent validation", "best_train_config"))
    plan.extend(
        [
            _experiment(
                "TRAIN-4-feature-ensemble", "TRAIN-4",
                dict(base, mode="test", ensemble_mode="feature", selection_pending=True),
                "exploratory seed ensemble if selected without independent validation", "seed_checkpoints",
            ),
            _experiment(
                "TRAIN-4-score-ensemble", "TRAIN-4",
                dict(base, mode="test", ensemble_mode="score", selection_pending=True),
                "exploratory seed ensemble if selected without independent validation", "seed_checkpoints",
            ),
        ]
    )
    for probability in (0.25, 0.5):
        plan.append(_experiment(f"TRAIN-5-llm-{str(probability).replace('.', 'p')}", "TRAIN-5", dict(base, llm_aug=True, llm_aug_prob=probability), dependency="TRAIN-4"))
    train6 = [
        ("eps0", {"label_smoothing": 0.0}),
        ("eps005", {"label_smoothing": 0.05}),
        ("eps01", {"label_smoothing": 0.1}),
        ("id05", {"id_loss_weight": 0.5}),
        ("id15", {"id_loss_weight": 1.5}),
        ("wrt05", {"wrt_loss_weight": 0.5}),
    ]
    for name, override in train6:
        plan.append(_experiment(f"TRAIN-6-{name}", "TRAIN-6", dict(base, **override), dependency="TRAIN-5"))
    for height, width in ((320, 160), (384, 192)):
        plan.append(_experiment(f"TRAIN-7-{height}x{width}", "TRAIN-7", dict(base, img_h=height, img_w=width, img_size=[height, width]), dependency="TRAIN-6"))
    pooling = [
        ("cls", {"visual_pooling": "cls"}),
        ("mean", {"visual_pooling": "mean_patch"}),
        ("gem", {"visual_pooling": "gem_patch"}),
        ("cls-gem", {"visual_pooling": "cls_gem", "patch_pool_gamma_init": 0.0}),
    ]
    for name, override in pooling:
        plan.append(_experiment(f"TRAIN-8-{name}", "TRAIN-8", dict(base, **override), dependency="TRAIN-7"))
    return plan


def _evaluation_phase_terminal() -> bool:
    from run_eval_sweep import build_eval_plan

    statuses = [read_json(experiment_status_path(item["id"]), {}) for item in build_eval_plan()]
    return bool(statuses) and all(item.get("status") in {"succeeded", "blocked", "skipped"} for item in statuses)


def _stage_succeeded(stage: str) -> bool:
    runs = REPORT_ROOT / "runs"
    if not runs.is_dir():
        return False
    return any(
        (payload := read_json(path, {})).get("phase") == stage and payload.get("status") == "succeeded"
        for path in runs.glob("*/status.json")
    )


def _dependency_ready(experiment: Mapping[str, Any]) -> bool:
    dependency = experiment.get("dependency")
    if dependency == "EVAL-phase-complete":
        return _evaluation_phase_terminal()
    if dependency == "best_train_config":
        return _stage_succeeded("TRAIN-1") or _stage_succeeded("TRAIN-2") or _stage_succeeded("TRAIN-3")
    if dependency == "seed_checkpoints":
        return len(_seed_checkpoints()) == 3
    if isinstance(dependency, str) and dependency.startswith("TRAIN-"):
        return _stage_succeeded(dependency)
    return True


def _successful_statuses(phases):
    runs = REPORT_ROOT / "runs"
    if not runs.is_dir():
        return []
    result = []
    for path in runs.glob("*/status.json"):
        payload = read_json(path, {})
        if payload.get("phase") in phases and payload.get("status") == "succeeded" and payload.get("Rank-1") is not None:
            result.append(payload)
    return result


def _best_training_overrides():
    rows = _successful_statuses({"TRAIN-1", "TRAIN-2", "TRAIN-3"})
    if not rows:
        return None
    best = max(rows, key=lambda row: (row["Rank-1"], row["mAP"], row["mINP"]))
    config = load_yaml(Path(best["runtime_config"]))
    excluded = {
        "output_path", "CUDA_VISIBLE_DEVICES", "gpu_id", "config_select", "test_model_path",
        "ensemble_checkpoints", "ensemble_mode", "selection_pending", "seed",
    }
    return {key: value for key, value in config.items() if key not in excluded}


def _seed_checkpoints():
    rows = _successful_statuses({"TRAIN-4"})
    selected = [
        row for row in rows
        if str(row.get("experiment", "")).startswith("TRAIN-4-seed-") and row.get("checkpoint")
    ]
    selected.sort(key=lambda row: int(str(row["experiment"]).rsplit("-", 1)[1]))
    return [row["checkpoint"] for row in selected]


def _materialize_overrides(experiment: Mapping[str, Any], checkpoint: Mapping[str, Any]):
    overrides = dict(experiment["overrides"])
    dependency = experiment.get("dependency")
    if dependency == "best_train_config":
        selected = _best_training_overrides()
        if selected:
            selected.update(overrides)
            overrides = selected
            overrides["selection_pending"] = False
    elif dependency == "seed_checkpoints":
        checkpoints = _seed_checkpoints()
        overrides["ensemble_checkpoints"] = checkpoints
        overrides["test_model_path"] = checkpoints[0] if checkpoints else checkpoint["checkpoint"]
        overrides["selection_pending"] = not bool(checkpoints)
    return overrides


def prepare(experiment: Mapping[str, Any], checkpoint: Mapping[str, Any]) -> Dict[str, Any]:
    run_dir = experiment_dir(experiment["id"])
    run_dir.mkdir(parents=True, exist_ok=True)
    existing = read_json(experiment_status_path(experiment["id"]), None)
    if existing and existing.get("status") == "succeeded":
        return existing
    overrides = _materialize_overrides(experiment, checkpoint)
    overrides["training_weight_init"] = checkpoint["checkpoint"]
    # Periodic SYSU validation runs alongside the training model.  Keep its
    # batch deliberately small so an unrelated user reclaiming GPU memory
    # cannot turn an otherwise healthy long run into an OOM.
    overrides.setdefault("test_batch_size", 16)
    recovery = (existing or {}).get("oom_recovery")
    if recovery:
        overrides["test_batch_size"] = int(recovery["test_batch_size"])
        overrides["training_weight_init"] = recovery["resume_checkpoint"]
        overrides["metric_boost_resume_epoch"] = int(recovery["resume_epoch"])
    overrides["output_path"] = str(run_dir / "model_output") + "/"
    runtime = prepare_runtime_config(experiment["id"], E4_CONFIG_PATH, overrides)
    if overrides.get("mode") == "test" and overrides.get("ensemble_mode") in {"feature", "score"}:
        output_json = run_dir / "result.json"
        command = [sys.executable, str(EVALUATOR), "--config", str(runtime), "--output-json", str(output_json)]
    else:
        command = [sys.executable, str(MAIN), "--config_select", str(runtime)]
    status = base_status(experiment["id"], experiment["stage"], experiment["validity"])
    status.update(
        {
            "checkpoint": checkpoint["checkpoint"],
            "best_epoch": checkpoint["best_epoch"],
            "runtime_config": str(runtime),
            "command": command,
            "command_shell": shlex.join(command),
            "log_path": str(run_dir / "launcher.log"),
            "dependency": experiment.get("dependency"),
            "dependency_ready": _dependency_ready(experiment),
            "warm_start": True,
        }
    )
    if recovery:
        status.update(
            {
                "oom_retry_count": int((existing or {}).get("oom_retry_count", 0)),
                "oom_recovery": recovery,
                "previous_error": (existing or {}).get("error"),
            }
        )
    atomic_write_json(experiment_status_path(experiment["id"]), status)
    (run_dir / "command.txt").write_text(shlex.join(command) + "\n", encoding="utf-8")
    return status


def _concurrent_train1_ready(experiment: Mapping[str, Any], enabled: bool) -> bool:
    if not enabled or experiment.get("stage") != "TRAIN-1":
        return False
    from run_eval_sweep import _baseline_reproduced

    return _baseline_reproduced()


def _acquire_training_gpu_lease(
    gpu_id_override: Optional[int],
    timeout_seconds: float = 600.0,
    poll_seconds: float = 2.0,
):
    """Atomically wait for a hardware-idle and lease-free training GPU."""
    deadline = time.monotonic() + timeout_seconds
    while True:
        idle = idle_gpu_ids(query_gpu_states())
        if gpu_id_override is None:
            candidates = idle
        else:
            candidates = [gpu_id_override] if gpu_id_override in idle else []
        for gpu_id in candidates:
            lease = GpuLease(gpu_id)
            try:
                lease.__enter__()
            except RuntimeError:
                continue
            return gpu_id, lease
        if time.monotonic() >= deadline:
            requested = "any GPU" if gpu_id_override is None else f"GPU {gpu_id_override}"
            raise RuntimeError(
                f"{requested} did not become hardware-idle and lease-free within "
                f"{timeout_seconds:g} seconds"
            )
        time.sleep(poll_seconds)


def run_one(
    experiment: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    *,
    gpu_id_override: Optional[int] = None,
    allow_concurrent_train1: bool = False,
) -> Dict[str, Any]:
    status = prepare(experiment, checkpoint)
    if status.get("status") == "succeeded":
        print(f"[skip] {experiment['id']} already succeeded", flush=True)
        return status
    concurrent_train1 = _concurrent_train1_ready(experiment, allow_concurrent_train1)
    if not _evaluation_phase_terminal() and not concurrent_train1:
        raise RuntimeError("Training is blocked until EVAL-0..EVAL-6 are terminal")
    if not status.get("dependency_ready") and not concurrent_train1:
        raise RuntimeError(f"Dependency is not ready for {experiment['id']}: {status.get('dependency')}")
    gpu_id, lease = _acquire_training_gpu_lease(gpu_id_override)
    import yaml
    from common import atomic_write_yaml

    runtime_path = Path(status["runtime_config"])
    with runtime_path.open("r", encoding="utf-8") as handle:
        runtime = yaml.load(handle, Loader=yaml.FullLoader)
    runtime["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    runtime["gpu_id"] = "0"
    atomic_write_yaml(runtime_path, runtime)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    try:
        status.update(
            {
                "status": "running",
                "gpu": gpu_id,
                "start_time": utc_now(),
                "runner_pid": os.getpid(),
            }
        )
        atomic_write_json(experiment_status_path(experiment["id"]), status)
        with Path(status["log_path"]).open("a", encoding="utf-8") as log:
            completed = subprocess.run(status["command"], stdout=log, stderr=subprocess.STDOUT, env=env)
    finally:
        lease.__exit__(None, None, None)
    status.update({"end_time": utc_now(), "return_code": completed.returncode})
    if completed.returncode != 0:
        status.update({"status": "failed", "error": f"Training returned {completed.returncode}"})
        atomic_write_json(experiment_status_path(experiment["id"]), status)
        raise RuntimeError(status["error"])
    runtime = load_yaml(Path(status["runtime_config"]))
    if runtime.get("mode") == "test":
        result = read_json(experiment_dir(experiment["id"]) / "result.json", {})
        metrics = result["metrics"]
        status.update(
            {
                "status": "succeeded",
                "Rank-1": metrics["Rank-1"],
                "mAP": metrics["mAP"],
                "mINP": metrics["mINP"],
                "checkpoint": runtime.get("ensemble_checkpoints"),
            }
        )
        atomic_write_json(experiment_status_path(experiment["id"]), status)
        return status
    status["status"] = "completed_pending_summary"
    atomic_write_json(experiment_status_path(experiment["id"]), status)
    from summarize_results import write_outputs

    write_outputs(REPORT_ROOT)
    finalized = read_json(experiment_status_path(experiment["id"]), status)
    if finalized.get("status") != "succeeded":
        raise RuntimeError(f"Training finished but metrics/checkpoint could not be summarized for {experiment['id']}")
    return finalized


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--gpu-id", type=int, help="Bind this foreground runner to one verified-idle physical GPU")
    parser.add_argument(
        "--allow-concurrent-train1-after-eval0",
        action="store_true",
        help="Allow only TRAIN-1 to run while later evaluations continue, after EVAL-0 reproduces E4",
    )
    args = parser.parse_args()
    if args.prepare_only and args.run:
        parser.error("Choose either --prepare-only or --run")
    if not args.prepare_only and not args.run:
        args.prepare_only = True
    from run_eval_sweep import build_eval_plan

    validate_experiment_manifest(build_eval_plan(), build_train_plan())
    checkpoint = resolve_e4_checkpoint()
    selected = [item for item in build_train_plan() if not args.only or item["id"] in args.only]
    if args.allow_concurrent_train1_after_eval0 and any(item["stage"] != "TRAIN-1" for item in selected):
        parser.error("--allow-concurrent-train1-after-eval0 is restricted to TRAIN-1 experiments")
    failure_seen = False
    for experiment in selected:
        if args.prepare_only:
            prepare(experiment, checkpoint)
        elif not failure_seen:
            try:
                run_one(
                    experiment,
                    checkpoint,
                    gpu_id_override=args.gpu_id,
                    allow_concurrent_train1=args.allow_concurrent_train1_after_eval0,
                )
            except Exception:
                failure_seen = True
                raise
    print(json.dumps({"mode": "prepare-only" if args.prepare_only else "run", "experiment_count": len(selected), "checkpoint": checkpoint["checkpoint"]}, sort_keys=True))


if __name__ == "__main__":
    main()
