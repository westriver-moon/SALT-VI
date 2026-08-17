#!/usr/bin/env python3
"""Run preregistered Stage-A ablations through a shared multi-GPU queue."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = PROJECT_ROOT / "configs/pipelines/sysu_safe_tricks.yaml"
DEFAULT_VARIANTS = ("a0", "a4", "a6", "a1", "a2", "a3", "a5")


def now():
    return datetime.now(timezone.utc).isoformat()


def write_state(path, state):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
    temporary.replace(path)


def available_gpus(gpus, running, jobs_per_gpu):
    gpu_loads = {gpu: 0 for gpu in gpus}
    for item in running.values():
        gpu_loads[item["gpu"]] += 1
    return [gpu for gpu in gpus if gpu_loads[gpu] < jobs_per_gpu]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpus", nargs="+", type=int, default=(1, 2, 3))
    parser.add_argument("--jobs-per-gpu", type=int, default=1)
    parser.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args(argv)

    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    configs = manifest["stage_a"]["variants"]
    unknown = sorted(set(args.variants) - set(configs))
    if unknown:
        parser.error(f"unknown Stage-A variants: {unknown}")
    if len(set(args.gpus)) != len(args.gpus):
        parser.error("GPU indices must be unique")
    if args.jobs_per_gpu < 1:
        parser.error("jobs per GPU must be at least 1")

    output_root = args.output_root.resolve()
    scheduler_dir = output_root / "scheduler"
    scheduler_dir.mkdir(parents=True, exist_ok=True)
    state_path = scheduler_dir / "state.json"
    state = {
        "status": "running",
        "started_at": now(),
        "output_root": str(output_root),
        "gpus": list(args.gpus),
        "jobs_per_gpu": args.jobs_per_gpu,
        "pending": list(args.variants),
        "running": {},
        "completed": [],
        "failed": [],
    }
    write_state(state_path, state)

    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["SALT_SAFE_TRICKS_OUTPUT_ROOT"] = str(output_root)
    running = {}
    pending = list(args.variants)

    while pending or running:
        for gpu in available_gpus(args.gpus, running, args.jobs_per_gpu):
            if not pending:
                break
            variant = pending.pop(0)
            log_path = scheduler_dir / f"{variant}.gpu{gpu}.log"
            log_handle = log_path.open("w", encoding="utf-8")
            command = [
                args.python,
                str(PROJECT_ROOT / "scripts/train.py"),
                "--config_select",
                str(PROJECT_ROOT / configs[variant]),
                "--CUDA_VISIBLE_DEVICES",
                str(gpu),
                "--gpu_id",
                "0",
            ]
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            running[variant] = {
                "process": process,
                "gpu": gpu,
                "log": str(log_path),
                "log_handle": log_handle,
                "started_at": now(),
            }
        state["pending"] = pending
        state["running"] = {
            variant: {
                "gpu": item["gpu"],
                "pid": item["process"].pid,
                "log": item["log"],
                "started_at": item["started_at"],
            }
            for variant, item in running.items()
        }
        write_state(state_path, state)
        if not running:
            break

        time.sleep(5)
        for variant, item in list(running.items()):
            returncode = item["process"].poll()
            if returncode is None:
                continue
            item["log_handle"].close()
            result = {
                "variant": variant,
                "gpu": item["gpu"],
                "returncode": returncode,
                "started_at": item["started_at"],
                "finished_at": now(),
                "log": item["log"],
            }
            state["completed" if returncode == 0 else "failed"].append(result)
            del running[variant]

    state["running"] = {}
    state["pending"] = pending
    state["finished_at"] = now()
    state["status"] = "failed" if state["failed"] else "completed"
    write_state(state_path, state)
    return 1 if state["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
