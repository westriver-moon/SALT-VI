#!/usr/bin/env python3
"""Plan, run, or inspect the C3-to-Stage-B B5 trick grid on GPUs 1 and 2."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from salt_vi.config.validation import validate_runtime_config
from salt_vi.utils.utils import load_train_configs


DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "configs/pipelines/sysu_stage_b_c3_b5_tricks_grid_20260824.yaml"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/home/lab929/ybj/experiments/stage_b/"
    "SALT-VI-c3-b5-tricks-grid-20260824"
)
OUTPUT_ENV = "SALT_B5_TRICKS_OUTPUT_ROOT"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_manifest(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 2:
        raise ValueError("manifest schema_version must be 2")
    return payload


def load_metrics(path: Path) -> list[dict]:
    events = []
    if not path.is_file():
        return events
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        metrics = event.get("metrics")
        if (
            event.get("event_type") == "eval_epoch"
            and isinstance(metrics, dict)
            and "Rank-1" in metrics
        ):
            events.append(event)
    return events


def build_environment(output_root: Path, checkpoint: Path, checkpoint_sha: str) -> dict:
    env = dict(os.environ)
    env[OUTPUT_ENV] = str(output_root)
    env["SALT_STAGE_A_CHECKPOINT"] = str(checkpoint)
    env["SALT_STAGE_A_SHA256"] = checkpoint_sha
    return env


def validate_plan(manifest: dict, env: dict) -> dict:
    stage_b = manifest["stage_b"]
    resolved = {}
    previous = dict(os.environ)
    os.environ.update(env)
    try:
        for name, item in stage_b["variants"].items():
            config_path = PROJECT_ROOT / item["config"]
            config = load_train_configs(str(config_path))
            validate_runtime_config(config)
            if int(config.total_train_epoch) != int(stage_b["total_train_epoch"]):
                raise ValueError(
                    f"{name} total_train_epoch={config.total_train_epoch}, "
                    f"expected {stage_b['total_train_epoch']}"
                )
            resolved[name] = {
                "config": str(config_path),
                "experiment_id": config.metric_experiment_id,
                "events": str(Path(config.metric_events_path)),
                "output_root": str(Path(config.output_root)),
                "sampler_type": str(config.sampler_type),
                "normalized_classifier": bool(config.normalized_classifier),
            }
    finally:
        os.environ.clear()
        os.environ.update(previous)
    return resolved


def query_gpu_processes(gpu: int) -> list[dict]:
    command = [
        "nvidia-smi",
        "-i",
        str(gpu),
        "--query-compute-apps=pid,used_memory,process_name",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"nvidia-smi process query failed for GPU {gpu}: {result.stderr.strip()}"
        )
    processes = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",", 2)]
        if not parts or not parts[0].isdigit():
            continue
        processes.append(
            {
                "pid": int(parts[0]),
                "used_memory_mb": int(parts[1]) if len(parts) > 1 else None,
                "process_name": parts[2] if len(parts) > 2 else None,
            }
        )
    return processes


def query_gpu_memory(gpu: int) -> int:
    command = [
        "nvidia-smi",
        "-i",
        str(gpu),
        "--query-gpu=memory.used",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=True)
    return int(result.stdout.strip().splitlines()[0])


def gpu_snapshot(gpu: int) -> dict:
    return {
        "gpu": gpu,
        "memory_used_mb": query_gpu_memory(gpu),
        "compute_processes": query_gpu_processes(gpu),
        "observed_at": utc_now(),
    }


def train_command(config_path: str, gpu: int) -> list[str]:
    return [
        sys.executable,
        str(PROJECT_ROOT / "scripts/train.py"),
        "--config_select",
        config_path,
        "--set",
        f'CUDA_VISIBLE_DEVICES="{gpu}"',
        "--set",
        'gpu_id="0"',
    ]


def summarize(manifest: dict, resolved: dict, state: dict) -> dict:
    rows = []
    for name, item in manifest["stage_b"]["variants"].items():
        events = load_metrics(Path(resolved[name]["events"]))
        best = max(events, key=lambda event: event["metrics"]["Rank-1"]) if events else None
        latest = events[-1] if events else None
        rows.append(
            {
                "variant": name,
                "purpose": item["purpose"],
                "status": state["jobs"][name]["status"],
                "gpu": state["jobs"][name]["gpu"],
                "best_epoch": best.get("epoch") if best else None,
                "best_metrics": best.get("metrics") if best else None,
                "latest_epoch": latest.get("epoch") if latest else None,
                "latest_metrics": latest.get("metrics") if latest else None,
            }
        )
    completed = [row for row in rows if row["best_metrics"]]
    best = (
        max(completed, key=lambda row: row["best_metrics"]["Rank-1"])
        if completed
        else None
    )
    return {
        "schema_version": 2,
        "experiment_id": manifest["experiment_id"],
        "updated_at": utc_now(),
        "protocol": manifest["protocol"],
        "baseline": manifest["stage_b"]["baseline"],
        "best_variant": best["variant"] if best else None,
        "best_metrics": best["best_metrics"] if best else None,
        "variants": rows,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "run", "status"))
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args(argv)

    manifest = load_manifest(args.manifest.resolve())
    output_root = args.output_root.resolve()
    checkpoint = Path(manifest["stage_a"]["checkpoint"]).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Stage-A checkpoint is missing: {checkpoint}")
    checkpoint_sha = sha256_file(checkpoint)
    expected_sha = manifest["stage_a"].get("checkpoint_sha256")
    if expected_sha and checkpoint_sha != expected_sha:
        raise ValueError(
            f"Stage-A checkpoint SHA mismatch: expected {expected_sha}, got {checkpoint_sha}"
        )
    env = build_environment(output_root, checkpoint, checkpoint_sha)
    resolved = validate_plan(manifest, env)

    scheduling = manifest["scheduling"]
    queues = {
        int(gpu): list(names)
        for gpu, names in scheduling["queues"].items()
    }
    selected = sorted(int(gpu) for gpu in scheduling["selected_gpus"])
    allowed = sorted(int(gpu) for gpu in scheduling["allowed_gpus"])
    if sorted(queues) != selected or selected != [1, 2] or allowed != [1, 2]:
        raise ValueError("this scheduler is intentionally restricted to physical GPUs 1 and 2")
    assigned = [name for names in queues.values() for name in names]
    expected = list(manifest["stage_b"]["variants"])
    if sorted(assigned) != sorted(expected) or len(assigned) != len(set(assigned)):
        raise ValueError("every Stage-B variant must be assigned exactly once")

    plan = {
        "schema_version": 2,
        "experiment_id": manifest["experiment_id"],
        "worktree": str(PROJECT_ROOT),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "output_root": str(output_root),
        "protocol": manifest["protocol"],
        "queues": queues,
        "variants": resolved,
        "safety": {
            "allowed_gpus": allowed,
            "poll_seconds": int(scheduling["poll_seconds"]),
            "max_idle_memory_mb": int(scheduling["max_idle_memory_mb"]),
            "initial_gpu_snapshots": [gpu_snapshot(gpu) for gpu in selected],
        },
    }
    if args.action == "plan":
        print(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    state_path = output_root / "scheduler" / "state.json"
    results_path = output_root / "results.json"
    if args.action == "status":
        state = json.loads(state_path.read_text(encoding="utf-8"))
        print(json.dumps(summarize(manifest, resolved, state), ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    output_root.mkdir(parents=True, exist_ok=False)
    scheduler_root = output_root / "scheduler"
    scheduler_root.mkdir(parents=True)
    atomic_json(output_root / "plan.json", plan)
    state = {
        "schema_version": 2,
        "experiment_id": manifest["experiment_id"],
        "status": "running",
        "started_at": utc_now(),
        "updated_at": utc_now(),
        "queues": queues,
        "jobs": {
            name: {"status": "queued", "gpu": gpu, "pid": None, "exit_code": None}
            for gpu, names in queues.items()
            for name in names
        },
    }
    lock = threading.Lock()
    atomic_json(state_path, state)

    def update(name: str, **values) -> None:
        with lock:
            state["jobs"][name].update(values)
            state["updated_at"] = utc_now()
            atomic_json(state_path, state)

    def wait_for_idle_gpu(gpu: int, name: str) -> dict:
        poll_seconds = int(scheduling["poll_seconds"])
        max_idle_memory_mb = int(scheduling["max_idle_memory_mb"])
        while True:
            snapshot = gpu_snapshot(gpu)
            idle = (
                not snapshot["compute_processes"]
                and snapshot["memory_used_mb"] <= max_idle_memory_mb
            )
            if idle:
                update(name, status="launching", idle_snapshot=snapshot)
                return snapshot
            update(name, status="waiting_for_gpu", blocked_snapshot=snapshot)
            time.sleep(poll_seconds)

    def run_queue(gpu: int, names: list[str]) -> list[int]:
        exit_codes = []
        for name in names:
            wait_for_idle_gpu(gpu, name)
            config_path = resolved[name]["config"]
            log_path = scheduler_root / f"{name}.gpu{gpu}.log"
            command = train_command(config_path, gpu)
            job_env = dict(env)
            job_env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            with log_path.open("ab", buffering=0) as handle:
                process = subprocess.Popen(
                    command,
                    cwd=PROJECT_ROOT,
                    env=job_env,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                )
                update(name, status="running", pid=process.pid, started_at=utc_now())
                code = process.wait()
            update(
                name,
                status="completed" if code == 0 else "failed",
                pid=None,
                exit_code=code,
                finished_at=utc_now(),
            )
            exit_codes.append(code)
            atomic_json(results_path, summarize(manifest, resolved, state))
            if code != 0 and not bool(scheduling.get("continue_on_failure", False)):
                break
        return exit_codes

    with ThreadPoolExecutor(max_workers=len(queues)) as executor:
        futures = [executor.submit(run_queue, gpu, names) for gpu, names in queues.items()]
        all_codes = [code for future in futures for code in future.result()]

    failures = [code for code in all_codes if code != 0]
    state["status"] = "completed" if not failures else "completed_with_failures"
    state["finished_at"] = utc_now()
    state["updated_at"] = utc_now()
    atomic_json(state_path, state)
    atomic_json(results_path, summarize(manifest, resolved, state))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
