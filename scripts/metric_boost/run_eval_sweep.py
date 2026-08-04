#!/usr/bin/env python
"""Prepare or foreground-run the ordered EVAL-0..EVAL-6 sweep."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import (
    E4_CONFIG_PATH,
    EXPECTED_E4,
    GpuLease,
    REPORT_ROOT,
    atomic_write_json,
    base_status,
    experiment_dir,
    experiment_status_path,
    idle_gpu_ids,
    nearby_checkpoints,
    prepare_runtime_config,
    query_gpu_states,
    read_json,
    resolve_e4_checkpoint,
    succeeded,
    utc_now,
    validate_experiment_manifest,
)


EVALUATOR = SCRIPT_DIR / "eval_engine.py"


def _experiment(experiment_id: str, stage: str, overrides: Mapping[str, Any], validity="standard", dependency=None):
    return {
        "id": experiment_id,
        "stage": stage,
        "overrides": dict(overrides),
        "validity": validity,
        "dependency": dependency,
    }


def build_eval_plan() -> List[Dict[str, Any]]:
    base = {
        "mode": "test",
        "dataset": "sysu",
        "test_mode": "all",
        "gall_mode": "single",
        "test_modality": "Fusion",
        "test_model_type": "Fusion",
        "gallery_trials": 10,
        "test_flip_tta": False,
        "test_multi_scale": [[288, 144]],
        "rerank": False,
        "ensemble_mode": "none",
    }
    plan = [
        _experiment("EVAL-0", "EVAL-0", dict(base, CAT_EVAL=False), "standard baseline reproduction"),
        _experiment(
            "EVAL-1",
            "EVAL-1",
            dict(base, CAT_EVAL=True, mer_l2_normalize=False, mer_fusion_weight=1.0, mer_ir_weight=1.0, mer_text_weight=1.0),
            "standard legacy equal-weight MER",
            "EVAL-0",
        ),
    ]
    for ir_weight, text_weight in itertools.product((0.0, 0.25, 0.5, 0.75, 1.0), repeat=2):
        suffix = f"ir{ir_weight:.2f}_text{text_weight:.2f}".replace(".", "p")
        plan.append(
            _experiment(
                f"EVAL-2-{suffix}",
                "EVAL-2",
                dict(
                    base,
                    CAT_EVAL=True,
                    mer_l2_normalize=True,
                    mer_fusion_weight=1.0,
                    mer_ir_weight=ir_weight,
                    mer_text_weight=text_weight,
                ),
                "exploratory test-set-tuned",
                "EVAL-1",
            )
        )
    plan.extend(
        [
            _experiment("EVAL-3-fusion-flip", "EVAL-3", dict(base, CAT_EVAL=False, test_flip_tta=True), dependency="EVAL-2"),
            _experiment(
                "EVAL-3-mer-flip", "EVAL-3", dict(base, CAT_EVAL=True, mer_l2_normalize=True, test_flip_tta=True),
                "exploratory test-set-tuned", "best_mer",
            ),
        ]
    )
    scale_sets = {
        "288": [[288, 144]],
        "256_288": [[256, 128], [288, 144]],
        "288_320": [[288, 144], [320, 160]],
        "256_288_320": [[256, 128], [288, 144], [320, 160]],
    }
    for name, scales in scale_sets.items():
        plan.append(_experiment(f"EVAL-4-fusion-{name}", "EVAL-4", dict(base, CAT_EVAL=False, test_multi_scale=scales), dependency="EVAL-3"))
        plan.append(
            _experiment(
                f"EVAL-4-mer-{name}", "EVAL-4",
                dict(base, CAT_EVAL=True, mer_l2_normalize=True, test_multi_scale=scales),
                "exploratory test-set-tuned", "best_mer",
            )
        )
    rerank_bases = {
        "fusion": dict(base, CAT_EVAL=False),
        "mer": dict(base, CAT_EVAL=True, mer_l2_normalize=True),
        "mer_tta": dict(base, CAT_EVAL=True, mer_l2_normalize=True, test_flip_tta=True),
    }
    for base_name, base_overrides in rerank_bases.items():
        for k1, k2, lambda_value in itertools.product((10, 20, 30), (3, 6), (0.2, 0.3, 0.5)):
            plan.append(
                _experiment(
                    f"EVAL-5-{base_name}-k{k1}-k{k2}-l{str(lambda_value).replace('.', 'p')}",
                    "EVAL-5",
                    dict(base_overrides, rerank=True, rerank_k1=k1, rerank_k2=k2, rerank_lambda=lambda_value),
                    "exploratory test-set-tuned post-processing",
                    "best_mer_tta" if base_name == "mer_tta" else ("best_mer" if base_name == "mer" else "EVAL-0"),
                )
            )
    plan.extend(
        [
            _experiment("EVAL-6-feature-ensemble", "EVAL-6", dict(base, CAT_EVAL=False, ensemble_mode="feature"), dependency="nearby_checkpoints"),
            _experiment("EVAL-6-score-ensemble", "EVAL-6", dict(base, CAT_EVAL=False, ensemble_mode="score"), dependency="nearby_checkpoints"),
        ]
    )
    return plan


def _successful_stage(stage: str) -> List[Dict[str, Any]]:
    rows = []
    for path in (REPORT_ROOT / "runs").glob("*/status.json") if (REPORT_ROOT / "runs").is_dir() else []:
        payload = read_json(path, {})
        if payload.get("phase") == stage and payload.get("status") == "succeeded" and payload.get("Rank-1") is not None:
            rows.append(payload)
    return rows


def _best_overrides(stage: str) -> Optional[Dict[str, Any]]:
    rows = _successful_stage(stage)
    if not rows:
        return None
    best = max(rows, key=lambda row: (row["Rank-1"], row["mAP"], row["mINP"]))
    runtime = Path(best["runtime_config"])
    import yaml

    with runtime.open("r", encoding="utf-8") as handle:
        config = yaml.load(handle, Loader=yaml.FullLoader)
    keys = [
        "mer_fusion_weight", "mer_ir_weight", "mer_text_weight", "mer_l2_normalize",
        "test_flip_tta", "test_multi_scale",
    ]
    return {key: config[key] for key in keys if key in config}


def materialize(experiment: Mapping[str, Any], checkpoint: Mapping[str, Any]) -> Dict[str, Any]:
    overrides = dict(experiment["overrides"])
    overrides["test_model_path"] = checkpoint["checkpoint"]
    # load_train_configs resolves the legacy AUTO_FIND sentinel before the
    # evaluator applies test_model_path.  Isolated worktrees intentionally do
    # not contain the old image-only training outputs, so bind both fields to
    # the already audited E4 checkpoint for evaluation.
    overrides["training_weight_init"] = checkpoint["checkpoint"]
    overrides["output_path"] = str(experiment_dir(experiment["id"]) / "model_output")
    dependency = experiment.get("dependency")
    dependency_ready = True
    if dependency == "best_mer":
        selected = _best_overrides("EVAL-2")
        dependency_ready = selected is not None
        if selected:
            overrides.update(selected)
    elif dependency == "best_mer_tta":
        selected = _best_overrides("EVAL-4") or _best_overrides("EVAL-3")
        dependency_ready = selected is not None
        if selected:
            overrides.update(selected)
    elif dependency == "nearby_checkpoints":
        available = [item["path"] for item in nearby_checkpoints(Path(checkpoint["checkpoint"])) if item["exists"]]
        dependency_ready = len(available) >= 2
        overrides["ensemble_checkpoints"] = available[:5]
    elif isinstance(dependency, str) and dependency.startswith("EVAL-"):
        if dependency in {"EVAL-0", "EVAL-1"}:
            dependency_ready = succeeded(dependency)
        else:
            dependency_ready = bool(_successful_stage(dependency))
    return {"overrides": overrides, "dependency_ready": dependency_ready}


def prepare(experiment: Mapping[str, Any], checkpoint: Mapping[str, Any]) -> Dict[str, Any]:
    run_dir = experiment_dir(experiment["id"])
    run_dir.mkdir(parents=True, exist_ok=True)
    materialized = materialize(experiment, checkpoint)
    runtime = prepare_runtime_config(experiment["id"], E4_CONFIG_PATH, materialized["overrides"])
    output_json = run_dir / "result.json"
    command = [sys.executable, str(EVALUATOR), "--config", str(runtime), "--output-json", str(output_json)]
    status = read_json(experiment_status_path(experiment["id"]), None)
    if status and status.get("status") == "succeeded":
        return status
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
            "dependency_ready": materialized["dependency_ready"],
        }
    )
    atomic_write_json(experiment_status_path(experiment["id"]), status)
    (run_dir / "command.txt").write_text(shlex.join(command) + "\n", encoding="utf-8")
    return status


def _baseline_reproduced() -> bool:
    payload = read_json(experiment_status_path("EVAL-0"), {})
    return payload.get("status") == "succeeded" and all(
        abs(float(payload[key]) - EXPECTED_E4[key]) <= 5e-4 for key in EXPECTED_E4
    )


def _acquire_idle_gpu_lease(timeout_seconds: float = 600.0, poll_seconds: float = 2.0):
    """Wait for a hardware-idle GPU whose metric-boost lease is also free."""
    deadline = time.monotonic() + timeout_seconds
    while True:
        for gpu_id in idle_gpu_ids(query_gpu_states()):
            lease = GpuLease(gpu_id)
            try:
                lease.__enter__()
            except RuntimeError:
                continue
            return gpu_id, lease
        if time.monotonic() >= deadline:
            raise RuntimeError(
                "No GPU became hardware-idle and lease-free within "
                f"{timeout_seconds:g} seconds"
            )
        time.sleep(poll_seconds)


def run_one(experiment: Mapping[str, Any], checkpoint: Mapping[str, Any]) -> Dict[str, Any]:
    status = prepare(experiment, checkpoint)
    if status.get("status") == "succeeded":
        print(f"[skip] {experiment['id']} already succeeded", flush=True)
        return status
    if experiment["id"] != "EVAL-0" and not _baseline_reproduced():
        raise RuntimeError("EVAL-0 has not reproduced E4; all dependent evaluation experiments are blocked")
    status = prepare(experiment, checkpoint)
    if not status.get("dependency_ready", True):
        status.update(
            {
                "status": "blocked",
                "end_time": utc_now(),
                "error": (
                    f"Dependency is not available for {experiment['id']}: "
                    f"{status.get('dependency')}"
                ),
            }
        )
        atomic_write_json(experiment_status_path(experiment["id"]), status)
        print(f"[blocked] {experiment['id']}: {status['error']}", flush=True)
        return status
    gpu_id, lease = _acquire_idle_gpu_lease()
    runtime_path = Path(status["runtime_config"])
    import yaml

    with runtime_path.open("r", encoding="utf-8") as handle:
        runtime_payload = yaml.load(handle, Loader=yaml.FullLoader)
    runtime_payload["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    runtime_payload["gpu_id"] = "0"
    from common import atomic_write_yaml

    atomic_write_yaml(runtime_path, runtime_payload)
    try:
        status.update({"status": "running", "gpu": gpu_id, "start_time": utc_now()})
        atomic_write_json(experiment_status_path(experiment["id"]), status)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        log_path = Path(status["log_path"])
        with log_path.open("a", encoding="utf-8") as log:
            completed = subprocess.run(status["command"], stdout=log, stderr=subprocess.STDOUT, text=True, env=env)
    finally:
        lease.__exit__(None, None, None)
    status["end_time"] = utc_now()
    status["return_code"] = completed.returncode
    if completed.returncode != 0:
        status.update({"status": "failed", "error": f"Evaluator returned {completed.returncode}"})
        atomic_write_json(experiment_status_path(experiment["id"]), status)
        raise RuntimeError(status["error"])
    result = read_json(experiment_dir(experiment["id"]) / "result.json", {})
    metrics = result["metrics"]
    status.update(
        {
            "status": "succeeded",
            "Rank-1": metrics["Rank-1"],
            "mAP": metrics["mAP"],
            "mINP": metrics["mINP"],
            "delta_rank1_pp": (metrics["Rank-1"] - EXPECTED_E4["Rank-1"]) * 100.0,
            "delta_map_pp": (metrics["mAP"] - EXPECTED_E4["mAP"]) * 100.0,
            "delta_minp_pp": (metrics["mINP"] - EXPECTED_E4["mINP"]) * 100.0,
        }
    )
    atomic_write_json(experiment_status_path(experiment["id"]), status)
    if experiment["id"] == "EVAL-0" and not _baseline_reproduced():
        raise RuntimeError("EVAL-0 completed but failed the E4 reproduction tolerance")
    return status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true", help="Generate configs/status/commands without launching CUDA")
    parser.add_argument("--run", action="store_true", help="Foreground-run in strict order when an idle GPU exists")
    parser.add_argument("--only", action="append", default=[])
    args = parser.parse_args()
    if args.prepare_only and args.run:
        parser.error("Choose either --prepare-only or --run")
    if not args.prepare_only and not args.run:
        args.prepare_only = True
    from run_train_sweep import build_train_plan

    validate_experiment_manifest(build_eval_plan(), build_train_plan())
    checkpoint = resolve_e4_checkpoint()
    selected = [item for item in build_eval_plan() if not args.only or item["id"] in args.only]
    for experiment in selected:
        if args.prepare_only:
            prepare(experiment, checkpoint)
        else:
            run_one(experiment, checkpoint)
    print(json.dumps({"mode": "prepare-only" if args.prepare_only else "run", "experiment_count": len(selected), "checkpoint": checkpoint["checkpoint"]}, sort_keys=True))


if __name__ == "__main__":
    main()
