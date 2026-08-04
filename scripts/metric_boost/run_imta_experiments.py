#!/usr/bin/env python
"""Prepare or safely run IMTA experiments beside existing metric-boost jobs."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import yaml
import fcntl

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import GpuLease, idle_gpu_ids, query_gpu_states, resolve_e4_checkpoint
from run_train_sweep import prepare, run_one

DEFAULT_PLAN = REPO_ROOT / "configs/metric_boost/imta_experiments.yaml"
SCHEDULER_LOCK = REPO_ROOT / "reports/metric_boost/imta_scheduler.lock"


class SchedulerLease:
    def __enter__(self):
        SCHEDULER_LOCK.parent.mkdir(parents=True, exist_ok=True)
        self.handle = SCHEDULER_LOCK.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.handle.close()
            raise RuntimeError("Another IMTA scheduler is already active")
        self.handle.seek(0)
        self.handle.truncate()
        self.handle.write(json.dumps({"pid": os.getpid(), "plan": str(DEFAULT_PLAN)}))
        self.handle.flush()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        self.handle.close()


def lease_free_idle_gpus():
    """Probe the shared metric-boost leases without retaining a GPU."""
    available = []
    for gpu_id in idle_gpu_ids(query_gpu_states()):
        lease = GpuLease(gpu_id)
        try:
            lease.__enter__()
        except RuntimeError:
            continue
        lease.__exit__(None, None, None)
        available.append(gpu_id)
    return available


def run_worker(experiment, checkpoint, gpu_id):
    return run_one(experiment, checkpoint, gpu_id_override=gpu_id)


def load_plan(path):
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    common = dict(payload.get("common_overrides", {}))
    rows = []
    seen = set()
    for row in payload.get("experiments", []):
        experiment_id = str(row["id"])
        if experiment_id in seen:
            raise ValueError(f"Duplicate experiment id: {experiment_id}")
        seen.add(experiment_id)
        overrides = dict(common)
        overrides.update(dict(row.get("overrides", {})))
        rows.append({
            "id": experiment_id,
            "stage": str(payload.get("stage", "IMTA-1")),
            "overrides": overrides,
            "validity": "training experiment; identity-manifold text alignment",
            "dependency": None,
            "description": str(row.get("description", "")),
        })
    if not rows:
        raise ValueError(f"No experiments in {path}")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--only", action="append", default=[])
    args = parser.parse_args()
    if not 1 <= args.max_workers <= 3:
        parser.error("--max-workers must be between 1 and 3")

    experiments = load_plan(args.plan.resolve())
    if args.only:
        requested = set(args.only)
        experiments = [row for row in experiments if row["id"] in requested]
        missing = requested - {row["id"] for row in experiments}
        if missing:
            parser.error(f"Unknown experiment ids: {sorted(missing)}")

    checkpoint = resolve_e4_checkpoint()
    if args.prepare_only:
        outputs = [prepare(row, checkpoint) for row in experiments]
    else:
        outputs, failures = [], []
        pending = list(experiments)
        running = {}
        # Only launch work for GPUs that are hardware-idle and lease-free at
        # this poll. Pending rows remain queued indefinitely while the other
        # metric-boost scheduler owns GPUs 0/1/2.
        with SchedulerLease(), ProcessPoolExecutor(max_workers=min(args.max_workers, len(experiments))) as pool:
            while pending or running:
                for future, metadata in list(running.items()):
                    if not future.done():
                        continue
                    del running[future]
                    try:
                        outputs.append(future.result())
                    except Exception as exc:
                        failures.append({"experiment": metadata["id"], "gpu": metadata["gpu"], "error": repr(exc)})

                capacity = min(args.max_workers, len(experiments)) - len(running)
                if capacity > 0 and pending:
                    busy = {metadata["gpu"] for metadata in running.values()}
                    available = [gpu for gpu in lease_free_idle_gpus() if gpu not in busy]
                    for gpu_id in available[:capacity]:
                        if not pending:
                            break
                        row = pending.pop(0)
                        future = pool.submit(run_worker, row, checkpoint, gpu_id)
                        running[future] = {"id": row["id"], "gpu": gpu_id}
                        print(json.dumps({"event": "launched", "experiment": row["id"], "gpu": gpu_id}), flush=True)

                if pending or running:
                    time.sleep(max(2.0, args.poll_seconds))
        if failures:
            print(json.dumps({"ok": False, "failures": failures}, indent=2, sort_keys=True))
            raise SystemExit(1)

    print(json.dumps({
        "ok": True,
        "mode": "run" if args.run else "prepare-only",
        "experiments": [row["id"] for row in experiments],
        "statuses": [row.get("status") for row in outputs],
        "shared_gpu_lease_root": "reports/metric_boost/gpu_locks",
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
