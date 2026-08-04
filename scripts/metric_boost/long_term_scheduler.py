#!/usr/bin/env python
"""Persistent, resumable stage-aware scheduler for the metric-boost matrix."""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import (
    REPORT_ROOT,
    atomic_write_json,
    experiment_dir,
    experiment_status_path,
    idle_gpu_ids,
    query_gpu_states,
    read_json,
    utc_now,
)
from run_eval_sweep import build_eval_plan
from run_train_sweep import (
    _dependency_ready,
    _evaluation_phase_terminal,
    build_train_plan,
)


try:
    import fcntl
except ImportError:  # pragma: no cover - the production scheduler is Linux-only
    fcntl = None


CONFIRM_TOKEN = "I_UNDERSTAND_LONG_TERM_GPU_SCHEDULER_WILL_START"
TERMINAL_SUCCESS = {"succeeded", "skipped"}
TERMINAL_FAILURE = {"failed", "blocked"}
PHASES = [f"TRAIN-{index}" for index in range(1, 9)]
STATE_PATH = REPORT_ROOT / "long_term_scheduler.json"
LOG_PATH = REPORT_ROOT / "long_term_scheduler.log"
LOCK_PATH = REPORT_ROOT / "long_term_scheduler.lock"
PYTHON = Path(sys.executable)
TRAIN_RUNNER = SCRIPT_DIR / "run_train_sweep.py"
EVAL_RUNNER = SCRIPT_DIR / "run_eval_sweep.py"
SUMMARIZER = SCRIPT_DIR / "summarize_results.py"
MAX_OOM_RETRIES = 2
OOM_RECOVERY_TEST_BATCH_SIZE = 16


def _pid_alive(pid: Optional[int]) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except (OSError, ValueError):
        return False
    return True


def _matching_process(fragment: str) -> Optional[int]:
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        return None
    for path in proc_root.iterdir():
        if not path.name.isdigit():
            continue
        try:
            command = (path / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if fragment in command:
            return int(path.name)
    return None


def _experiment_process(experiment_id: str, status: Mapping[str, Any]) -> Optional[int]:
    runner_pid = status.get("runner_pid")
    if _pid_alive(runner_pid):
        return int(runner_pid)
    runtime = status.get("runtime_config") or str(experiment_dir(experiment_id) / "runtime_config.yaml")
    return _matching_process(str(runtime))


def _status(experiment_id: str) -> Dict[str, Any]:
    return read_json(experiment_status_path(experiment_id), {})


def _phase_items(plan: Sequence[Mapping[str, Any]], phase: str) -> List[Mapping[str, Any]]:
    return [item for item in plan if item["stage"] == phase]


def _current_phase(plan: Sequence[Mapping[str, Any]]) -> Optional[str]:
    for phase in PHASES:
        statuses = [_status(item["id"]).get("status", "pending") for item in _phase_items(plan, phase)]
        if not statuses or all(status in TERMINAL_SUCCESS for status in statuses):
            continue
        return phase
    return None


def _status_counts(plan: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    return dict(Counter(_status(item["id"]).get("status", "missing") for item in plan))


def _phase_launch_allowed(phase: str, evaluation_terminal: bool) -> bool:
    return phase == "TRAIN-1" or evaluation_terminal


def _running_gpu_ids(eval_plan: Sequence[Mapping[str, Any]], train_plan: Sequence[Mapping[str, Any]]) -> List[int]:
    result = set()
    for item in list(eval_plan) + list(train_plan):
        status = _status(item["id"])
        if status.get("status") != "running" or status.get("gpu") is None:
            continue
        if _experiment_process(item["id"], status) is not None:
            result.add(int(status["gpu"]))
    return sorted(result)


def _active_training(plan: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    result = {}
    for item in plan:
        status = _status(item["id"])
        if status.get("status") != "running":
            continue
        pid = _experiment_process(item["id"], status)
        if pid is not None:
            result[item["id"]] = pid
    return result


def _stale_running(plan: Sequence[Mapping[str, Any]]) -> List[str]:
    stale = []
    for item in plan:
        status = _status(item["id"])
        if status.get("status") == "running" and _experiment_process(item["id"], status) is None:
            stale.append(item["id"])
    return stale


def _available_gpu_ids(
    eval_plan: Sequence[Mapping[str, Any]],
    train_plan: Sequence[Mapping[str, Any]],
) -> List[int]:
    reserved = set(_running_gpu_ids(eval_plan, train_plan))
    return [gpu_id for gpu_id in idle_gpu_ids(query_gpu_states()) if gpu_id not in reserved]


def _latest_model_checkpoint(experiment_id: str) -> Optional[Tuple[Path, int]]:
    pattern = re.compile(r"model_Fusion_(\d+)\.pth$")
    candidates = []
    for path in experiment_dir(experiment_id).glob("model_output/**/models/model_Fusion_*.pth"):
        match = pattern.search(path.name)
        if match:
            candidates.append((int(match.group(1)), path.stat().st_mtime, path))
    if not candidates:
        return None
    epoch, _, path = max(candidates, key=lambda item: (item[0], item[1]))
    return path.resolve(), epoch


def _is_cuda_oom(status: Mapping[str, Any]) -> bool:
    log_path = status.get("log_path")
    if not log_path:
        return False
    try:
        return "CUDA out of memory" in Path(log_path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False


def _recover_cuda_oom(experiment_id: str, status: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    if not _is_cuda_oom(status):
        return None
    retry_count = int(status.get("oom_retry_count", 0))
    if retry_count >= MAX_OOM_RETRIES:
        return None
    checkpoint = _latest_model_checkpoint(experiment_id)
    if checkpoint is None:
        return None
    checkpoint_path, saved_epoch = checkpoint
    recovery = {
        "reason": "cuda_oom_during_periodic_validation",
        "resume_checkpoint": str(checkpoint_path),
        "resume_epoch": int(saved_epoch) + 1,
        "test_batch_size": OOM_RECOVERY_TEST_BATCH_SIZE,
        "recovered_at": utc_now(),
    }
    updated = dict(status)
    history = list(updated.get("error_history", []))
    if updated.get("error"):
        history.append(updated["error"])
    updated.update(
        {
            "status": "pending",
            "gpu": None,
            "start_time": None,
            "end_time": None,
            "return_code": None,
            "runner_pid": None,
            "error": None,
            "error_history": history,
            "oom_retry_count": retry_count + 1,
            "oom_recovery": recovery,
        }
    )
    atomic_write_json(experiment_status_path(experiment_id), updated)
    return recovery


def _launch(command: Sequence[str], log_path: Path) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("a", encoding="utf-8")
    try:
        process = subprocess.Popen(
            list(command),
            cwd=str(SCRIPT_DIR.parents[1]),
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    finally:
        log.close()
    return process


class SingletonLease:
    def __init__(self, path: Path):
        self.path = path
        self.handle = None

    def __enter__(self):
        if fcntl is None:
            raise RuntimeError("Long-term scheduler requires Linux fcntl")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another long-term metric-boost scheduler is already active")
        self.handle.seek(0)
        self.handle.truncate()
        self.handle.write(json.dumps({"pid": os.getpid(), "started_at": utc_now()}))
        self.handle.flush()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.handle is not None:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()
        return False


class Scheduler:
    def __init__(self, poll_seconds: float, max_parallel: int):
        self.poll_seconds = poll_seconds
        self.max_parallel = max_parallel
        self.eval_plan = build_eval_plan()
        self.train_plan = build_train_plan()
        self.children: Dict[str, subprocess.Popen] = {}
        self.child_gpus: Dict[str, int] = {}
        self.eval_child: Optional[subprocess.Popen] = None
        self.eval_restart_count = int(read_json(STATE_PATH, {}).get("eval_restart_count", 0))
        self.stop_requested = False

    def _write_state(self, state: str, reason: Optional[str] = None) -> Dict[str, Any]:
        phase = _current_phase(self.train_plan)
        payload = {
            "state": state,
            "reason": reason,
            "pid": os.getpid(),
            "updated_at": utc_now(),
            "current_phase": phase,
            "max_parallel": self.max_parallel,
            "poll_seconds": self.poll_seconds,
            "evaluation_terminal": _evaluation_phase_terminal(),
            "evaluation_status_counts": _status_counts(self.eval_plan),
            "training_status_counts": _status_counts(self.train_plan),
            "active_training": _active_training(self.train_plan),
            "scheduler_children": {
                experiment_id: {"pid": process.pid, "gpu": self.child_gpus.get(experiment_id)}
                for experiment_id, process in self.children.items()
                if process.poll() is None
            },
            "evaluation_runner_pid": (
                self.eval_child.pid if self.eval_child is not None and self.eval_child.poll() is None
                else _matching_process(str(EVAL_RUNNER))
            ),
            "eval_restart_count": self.eval_restart_count,
            "idle_gpu_ids": idle_gpu_ids(query_gpu_states()),
            "reserved_gpu_ids": _running_gpu_ids(self.eval_plan, self.train_plan),
        }
        atomic_write_json(STATE_PATH, payload)
        return payload

    def _reap(self):
        for experiment_id, process in list(self.children.items()):
            return_code = process.poll()
            if return_code is None:
                continue
            del self.children[experiment_id]
            self.child_gpus.pop(experiment_id, None)
            if return_code != 0:
                status = _status(experiment_id)
                if status.get("status") not in TERMINAL_FAILURE:
                    status.update(
                        {
                            "status": "failed",
                            "end_time": utc_now(),
                            "error": f"Scheduler child exited with code {return_code}",
                        }
                    )
                    atomic_write_json(experiment_status_path(experiment_id), status)
        if self.eval_child is not None and self.eval_child.poll() is not None:
            if self.eval_child.returncode != 0:
                self.eval_restart_count += 1
            self.eval_child = None

    def _ensure_evaluation(self):
        if _evaluation_phase_terminal():
            return
        if _matching_process(str(EVAL_RUNNER)) is not None:
            return
        if self.eval_restart_count >= 3:
            raise RuntimeError("Evaluation runner failed three times; refusing an infinite restart loop")
        self.eval_child = _launch(
            [str(PYTHON), str(EVAL_RUNNER), "--run"],
            REPORT_ROOT / "long_term_scheduler_eval.log",
        )

    def _launch_ready_training(self):
        phase = _current_phase(self.train_plan)
        if phase is None:
            return
        if not _phase_launch_allowed(phase, _evaluation_phase_terminal()):
            return
        phase_items = _phase_items(self.train_plan, phase)
        for item in phase_items:
            status = _status(item["id"])
            if status.get("status") == "failed":
                _recover_cuda_oom(item["id"], status)
        failures = [item["id"] for item in phase_items if _status(item["id"]).get("status") in TERMINAL_FAILURE]
        if failures:
            raise RuntimeError(f"Training phase {phase} has terminal failures: {', '.join(failures)}")
        stale = _stale_running(phase_items)
        if stale:
            raise RuntimeError(f"Training phase {phase} has stale running states: {', '.join(stale)}")

        active = _active_training(self.train_plan)
        slots = max(0, self.max_parallel - len(active))
        if slots == 0:
            return
        available_gpus = _available_gpu_ids(self.eval_plan, self.train_plan)
        if not available_gpus:
            return

        for item in phase_items:
            if slots == 0 or not available_gpus:
                break
            experiment_id = item["id"]
            status = _status(experiment_id).get("status", "pending")
            if status in TERMINAL_SUCCESS or status == "running" or experiment_id in self.children:
                continue
            allow_concurrent_train1 = phase == "TRAIN-1" and not _evaluation_phase_terminal()
            if not _dependency_ready(item) and not allow_concurrent_train1:
                continue
            gpu_id = available_gpus.pop(0)
            command = [
                str(PYTHON),
                str(TRAIN_RUNNER),
                "--run",
                "--only",
                experiment_id,
                "--gpu-id",
                str(gpu_id),
            ]
            if allow_concurrent_train1:
                command.append("--allow-concurrent-train1-after-eval0")
            process = _launch(command, experiment_dir(experiment_id) / "scheduler_runner.log")
            self.children[experiment_id] = process
            self.child_gpus[experiment_id] = gpu_id
            slots -= 1

    def run(self):
        while not self.stop_requested:
            self._reap()
            self._ensure_evaluation()
            self._launch_ready_training()
            phase = _current_phase(self.train_plan)
            if phase is None:
                completed = subprocess.run(
                    [str(PYTHON), str(SUMMARIZER)],
                    cwd=str(SCRIPT_DIR.parents[1]),
                    check=False,
                )
                if completed.returncode != 0:
                    raise RuntimeError("Final summary generation failed")
                return self._write_state("completed", "All TRAIN-1..TRAIN-8 experiments succeeded or skipped")
            self._write_state("running")
            time.sleep(self.poll_seconds)
        return self._write_state("stopped", "Stop signal received")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--status-only", action="store_true")
    parser.add_argument("--confirm-launch")
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--max-parallel", type=int, default=4)
    args = parser.parse_args()
    if args.status_only:
        print(json.dumps(read_json(STATE_PATH, {"state": "not-started"}), sort_keys=True))
        return
    if not args.run:
        parser.error("Use --run or --status-only")
    if args.confirm_launch != CONFIRM_TOKEN:
        parser.error(f"--run requires --confirm-launch {CONFIRM_TOKEN}")
    if args.poll_seconds < 1:
        parser.error("--poll-seconds must be at least 1")
    if args.max_parallel < 1:
        parser.error("--max-parallel must be positive")

    scheduler = Scheduler(args.poll_seconds, args.max_parallel)

    def request_stop(signum, frame):
        scheduler.stop_requested = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    with SingletonLease(LOCK_PATH):
        scheduler._write_state("starting")
        try:
            payload = scheduler.run()
        except Exception as exc:
            payload = scheduler._write_state("failed", str(exc))
            print(json.dumps(payload, sort_keys=True), flush=True)
            raise
    print(json.dumps(payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
