#!/usr/bin/env python3
"""Run one route from the four-GPU RN50 Stage-B two-stage grid."""

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import traceback

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ROUTES = (
    REPO_ROOT
    / "configs"
    / "experiments"
    / "reproduction"
    / "archived_configs"
    / "stage_b_rn50_two_stage_grid_40"
    / "routes.yaml"
)


def atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.{}".format(os.getpid()))
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def best_eval_event(events_path):
    best = None
    with Path(events_path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    "invalid JSONL at {}:{}: {}".format(events_path, line_number, exc)
                )
            if event.get("event_type") != "eval_epoch":
                continue
            metrics = event.get("metrics") or {}
            try:
                rank1 = float(metrics["Rank-1"])
                map_value = float(metrics["mAP"])
                minp = float(metrics["mINP"])
                epoch = int(event.get("epoch", -1))
            except (KeyError, TypeError, ValueError):
                continue
            if not all(math.isfinite(value) for value in (rank1, map_value, minp)):
                continue
            checkpoint = (event.get("checkpoint_paths") or {}).get("Rank-1")
            candidate = {
                "primary_metric": rank1,
                "metrics": {
                    "Rank-1": rank1,
                    "mAP": map_value,
                    "mINP": minp,
                    "best_epoch": float(epoch),
                },
                "checkpoint_path": checkpoint,
            }
            if best is None or rank1 > best[0]:
                best = (rank1, candidate)
    if best is None:
        raise RuntimeError("no finite eval_epoch metrics found in {}".format(events_path))
    result = best[1]
    checkpoint = result["checkpoint_path"]
    if not checkpoint or not Path(checkpoint).is_file():
        raise RuntimeError("best checkpoint is missing: {}".format(checkpoint))
    result["checkpoint_sha256"] = sha256_file(checkpoint)
    return result


def load_design(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        design = yaml.safe_load(handle) or {}
    if int(design.get("schema_version", 0)) != 1:
        raise ValueError("unsupported route design schema")
    routes = design.get("routes")
    if not isinstance(routes, dict) or len(routes) != 4:
        raise ValueError("route design must contain exactly four routes")
    total = int(design.get("total_epochs", 0))
    if total != 40:
        raise ValueError("route design must use exactly 40 total epochs")
    for name, route in routes.items():
        phase1 = int(route["phase1_epochs"])
        phase2 = int(route["phase2_epochs"])
        if phase1 + phase2 != total:
            raise ValueError("{} does not sum to 40 epochs".format(name))
    assigned = sorted(int(route["gpu"]) for route in routes.values())
    if assigned != [0, 1, 2, 3]:
        raise ValueError("routes must map one-to-one to GPUs 0,1,2,3")
    return design


def resolve_repo_path(value):
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def validate_design(design):
    src_root = REPO_ROOT / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    from salt_vi.config.validation import validate_runtime_config
    from salt_vi.utils.utils import load_train_configs

    source = Path(design["source_stage_a_checkpoint"])
    if not source.is_file():
        raise FileNotFoundError("missing Stage-A checkpoint: {}".format(source))
    config_keys = {"phase1_config", "triangle_config", "fullpairs_config"}
    resolved = {}
    for key in sorted(config_keys):
        path = resolve_repo_path(design[key])
        if not path.is_file():
            raise FileNotFoundError("missing {}: {}".format(key, path))
        config = validate_runtime_config(load_train_configs(str(path)))
        if config.pretrain_choice != "RN50_ORI" or int(config.prj_output_dim) != 2048:
            raise ValueError("{} is not an RN50_ORI/2048 config".format(path))
        if not bool(config.Fix_Visual):
            raise ValueError("{} must freeze the visual backbone".format(path))
        resolved[key] = str(path)
    return {
        "source_stage_a_checkpoint": str(source),
        "source_stage_a_sha256": sha256_file(source),
        "configs": resolved,
    }


def override(name, value):
    if isinstance(value, str):
        return "{}={}".format(name, json.dumps(value))
    return "{}={}".format(name, json.dumps(value))


def run_phase(route_root, phase_name, config_path, epochs, experiment_id, init_checkpoint, gpu):
    phase_root = route_root / phase_name
    model_root = phase_root / "model_output"
    events_path = phase_root / "events.jsonl"
    phase_root.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "train.py"),
        "--config_select",
        str(config_path),
        "--set",
        override("CUDA_VISIBLE_DEVICES", str(gpu)),
        "--set",
        override("gpu_id", "0"),
        "--set",
        override("total_train_epoch", int(epochs)),
        "--set",
        override("training_weight_init", str(init_checkpoint)),
        "--set",
        override("output_root", str(model_root) + "/"),
        "--set",
        override("metric_events_path", str(events_path)),
        "--set",
        override("metric_experiment_id", experiment_id),
    ]
    atomic_write_json(
        phase_root / "command.json",
        {"argv": command, "cwd": str(REPO_ROOT), "gpu": int(gpu)},
    )
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONUNBUFFERED"] = "1"
    with (phase_root / "process.log").open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            "{} failed with exit code {}; see {}".format(
                phase_name, completed.returncode, phase_root / "process.log"
            )
        )
    result = best_eval_event(events_path)
    result["metrics"]["planned_epochs"] = float(epochs)
    result["metrics"]["selected_gpu"] = float(gpu)
    atomic_write_json(phase_root / "metrics.json", result)
    return result


def run_route(design, route_name, validation):
    route = design["routes"][route_name]
    gpu = int(route["gpu"])
    phase1_epochs = int(route["phase1_epochs"])
    phase2_epochs = int(route["phase2_epochs"])
    output_root = Path(design["output_root"])
    route_root = output_root / route_name
    route_root.mkdir(parents=True, exist_ok=True)

    lock_handle = (route_root / ".route.lock").open("w")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("route is already running: {}".format(route_name))

    state_path = route_root / "state.json"
    state = {
        "schema_version": 1,
        "experiment_group": design["experiment_group"],
        "route": route_name,
        "gpu": gpu,
        "pid": os.getpid(),
        "status": "running_phase1",
        "phase1_epochs": phase1_epochs,
        "phase2_epochs": phase2_epochs,
        "total_epochs": phase1_epochs + phase2_epochs,
        "hypothesis": route["hypothesis"],
        "source_stage_a_checkpoint": validation["source_stage_a_checkpoint"],
        "source_stage_a_sha256": validation["source_stage_a_sha256"],
    }
    atomic_write_json(state_path, state)
    try:
        phase1 = run_phase(
            route_root,
            "phase1_align",
            Path(validation["configs"]["phase1_config"]),
            phase1_epochs,
            "{}-{}-B1".format(design["experiment_group"], route_name),
            validation["source_stage_a_checkpoint"],
            gpu,
        )
        state.update(status="running_phase2", phase1=phase1)
        atomic_write_json(state_path, state)

        phase2_config_key = route["phase2_config"]
        phase2 = run_phase(
            route_root,
            "phase2_refine",
            Path(validation["configs"][phase2_config_key]),
            phase2_epochs,
            "{}-{}-B2".format(design["experiment_group"], route_name),
            phase1["checkpoint_path"],
            gpu,
        )
        final_metrics = {
            "primary_metric": phase2["primary_metric"],
            "metrics": dict(phase2["metrics"]),
            "route": route_name,
            "hypothesis": route["hypothesis"],
            "phase1": phase1,
            "phase2": phase2,
        }
        final_metrics["metrics"].update(
            {
                "phase1_epochs": float(phase1_epochs),
                "phase2_epochs": float(phase2_epochs),
                "total_planned_epochs": float(phase1_epochs + phase2_epochs),
            }
        )
        atomic_write_json(route_root / "metrics.json", final_metrics)
        state.update(status="completed", phase2=phase2, pid=None)
        atomic_write_json(state_path, state)
    except Exception as exc:
        state.update(status="failed", error=str(exc), traceback=traceback.format_exc(), pid=None)
        atomic_write_json(state_path, state)
        raise
    finally:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        lock_handle.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--routes-config", default=str(DEFAULT_ROUTES))
    parser.add_argument("--route", choices=[
        "r0_switch20_triangle",
        "r1_switch24_triangle",
        "r2_switch28_triangle",
        "r3_switch24_fullpairs",
    ])
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    design = load_design(args.routes_config)
    validation = validate_design(design)
    if args.validate_only:
        print(json.dumps({"design": design, "validation": validation}, indent=2, sort_keys=True))
        return 0
    if not args.route:
        raise ValueError("--route is required unless --validate-only is used")
    run_route(design, args.route, validation)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print("two-stage route failed: {}".format(exc), file=sys.stderr)
        sys.exit(1)
