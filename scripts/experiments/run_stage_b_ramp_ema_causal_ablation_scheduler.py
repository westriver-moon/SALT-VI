#!/usr/bin/env python3
"""Schedule the two-phase Ramp/EMA ablation on idle allow-listed GPUs."""

from __future__ import annotations

import argparse
import ctypes
from collections import deque
from datetime import datetime, timezone
import json
import math
import os
import signal
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
JOB_RUNNER = PROJECT_ROOT / "scripts/training/run_stage_b_ramp_ema_causal_ablation_job.py"
DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "configs/experiments/stage_b_ramp_ema_causal_ablation_20260825/manifest.yaml"
)
ALLOWED_GPUS = [0, 1, 2, 3]


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


def load_manifest(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 2:
        raise ValueError("ablation manifest schema_version must be 2")
    jobs = payload.get("jobs") or []
    ids = [str(job["id"]) for job in jobs]
    if len(ids) != len(set(ids)):
        raise ValueError("ablation job ids must be unique")
    selected = payload["scheduling"]["selected_gpus"]
    if len(selected) != 2 or len(set(selected)) != 2 or not set(selected).issubset(ALLOWED_GPUS):
        raise ValueError("select exactly two distinct physical GPUs from 0..3")
    scheduling = payload["scheduling"]
    if sorted(scheduling["allowed_gpus"]) != sorted(selected):
        raise ValueError("allowed GPUs must equal the two selected GPUs")
    if scheduling.get("mode") != "shared_queue" or scheduling.get("max_concurrent_jobs") != 2:
        raise ValueError("this scheduler requires a shared queue with a two-job limit")
    if scheduling.get("max_oom_retries") not in (0, 1):
        raise ValueError("max_oom_retries must be 0 or 1")
    if scheduling.get("idle_confirmations", 0) < 1 or scheduling.get("poll_seconds", 0) <= 0:
        raise ValueError("idle confirmations and poll interval must be positive")
    expected = {f"G{group}_s{seed}" for group in range(4) for seed in range(3)} | {f"G{group}_s0" for group in range(4, 8)}
    if set(ids) != expected:
        raise ValueError("manifest must contain the registered 16 jobs exactly")
    if payload["training"]["planned_epochs"] != 30 or payload["training"]["completed_epochs"] != 12:
        raise ValueError("the registered schedule is 30 planned / 12 completed epochs")
    for phase in ("phase1", "phase2"):
        job_order_for_phase(payload, phase)
    return payload


def query_gpu_processes(gpu: int) -> list[dict]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(gpu),
            "--query-compute-apps=pid,used_memory,process_name",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"nvidia-smi process query failed for GPU {gpu}: {result.stderr.strip()}")
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
    result = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(gpu),
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    return int(result.stdout.strip().splitlines()[0])


def gpu_snapshot(gpu: int) -> dict:
    return {
        "gpu": gpu,
        "memory_used_mb": query_gpu_memory(gpu),
        "compute_processes": query_gpu_processes(gpu),
        "observed_at": utc_now(),
    }


def phase_jobs(manifest: dict, phase: str) -> dict[str, dict]:
    return {
        str(job["id"]): dict(job)
        for job in manifest["jobs"]
        if str(job["phase"]) == phase
    }


def job_order_for_phase(manifest: dict, phase: str) -> list[str]:
    key = f"{phase}_order"
    assigned = [str(name) for name in manifest["scheduling"].get(key, [])]
    jobs = phase_jobs(manifest, phase)
    if sorted(assigned) != sorted(jobs) or len(assigned) != len(set(assigned)):
        raise ValueError(f"{key} must assign every {phase} job exactly once")
    return assigned


def interaction(phase1_results: dict[str, dict]) -> dict:
    per_seed = {}
    for seed in (0, 1, 2):
        rows = {}
        for group in ("G0", "G1", "G2", "G3"):
            rows[group] = phase1_results[f"{group}_s{seed}"]["best_by_model"]
        metrics = {}
        for metric in ("Rank-1", "mAP", "mINP"):
            online = (
                rows["G3"]["online"][metric]
                - rows["G1"]["online"][metric]
                - rows["G2"]["online"][metric]
                + rows["G0"]["online"][metric]
            )
            ema = (
                rows["G3"]["ema"][metric]
                - rows["G1"]["online"][metric]
                - rows["G2"]["ema"][metric]
                + rows["G0"]["online"][metric]
            )
            metrics[metric] = {"I_online": online, "I_ema": ema}
        per_seed[str(seed)] = metrics

    summary = {}
    for metric in ("Rank-1", "mAP", "mINP"):
        for kind in ("I_online", "I_ema"):
            values = [per_seed[str(seed)][metric][kind] for seed in (0, 1, 2)]
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / len(values)
            summary[f"{metric}_{kind}"] = {
                "mean": mean,
                "std": math.sqrt(variance),
                "std_ddof": 0,
                "sample_std": math.sqrt(variance * len(values) / (len(values) - 1)),
                "values": values,
            }
    return {"per_seed": per_seed, "summary": summary}


def run_phase(
    phase: str,
    manifest: dict,
    manifest_path: Path,
    output_root: Path,
    state: dict,
    state_path: Path,
    lock: threading.Lock,
) -> list[int]:
    scheduling = manifest["scheduling"]
    selected = scheduling["selected_gpus"]
    pending = deque(name for name in job_order_for_phase(manifest, phase)
                    if state["jobs"][name].get("status") != "completed")
    active = {}
    idle_counts = {gpu: 0 for gpu in selected}
    codes = []
    stop_dispatch = False

    def update(job_id: str, **values) -> None:
        with lock:
            state["jobs"][job_id].update(values)
            state["updated_at"] = utc_now()
            atomic_json(state_path, state)

    while pending or active:
        for gpu, running in list(active.items()):
            code = running["process"].poll()
            if code is None:
                continue
            running["handle"].close()
            job_id = running["job_id"]
            job_dir = output_root / "jobs" / job_id
            result_path = job_dir / "job_result.json"
            result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else {}
            status = "completed" if code == 0 and result.get("status") == "completed" else "failed"
            failure_snapshot = gpu_snapshot(gpu) if status == "failed" else None
            update(job_id, status=status, gpu=gpu, pid=None, exit_code=code,
                   finished_at=utc_now(), result=result, log=str(running["log"]),
                   failure_gpu_snapshot=failure_snapshot,
                   last_running_gpu_snapshot=running.get("last_gpu_snapshot"))
            del active[gpu]
            idle_counts[gpu] = 0
            is_oom = "cuda out of memory" in str(result.get("error", "")).lower()
            attempt = running["attempt"]
            if status == "failed" and is_oom and attempt <= scheduling["max_oom_retries"]:
                # A retry starts from the original Stage-A initialization. Never mix
                # incomplete checkpoints/events into a fresh deterministic attempt.
                archive = output_root / "failed_attempts" / f"{job_id}.attempt{attempt}"
                archive.parent.mkdir(parents=True, exist_ok=True)
                job_dir.rename(archive)
                history = list(state["jobs"][job_id].get("failed_attempts", []))
                history.append({"attempt": attempt, "gpu": gpu, "exit_code": code,
                                "output": str(archive), "log": str(running["log"]),
                                "post_exit_gpu_snapshot": failure_snapshot,
                                "last_running_gpu_snapshot": running.get("last_gpu_snapshot"),
                                "error": result.get("error")})
                update(job_id, status="queued", result={}, failed_attempts=history)
                pending.append(job_id)
            else:
                codes.append(0 if status == "completed" else (code or 1))
                if status == "failed" and not scheduling.get("continue_on_failure", False):
                    stop_dispatch = True

        for gpu in selected:
            if gpu in active:
                snapshot = gpu_snapshot(gpu)
                active[gpu]["last_gpu_snapshot"] = snapshot
                update(active[gpu]["job_id"], last_running_gpu_snapshot=snapshot)
                continue
            if stop_dispatch or not pending:
                continue
            snapshot = gpu_snapshot(gpu)
            idle = not snapshot["compute_processes"] and snapshot["memory_used_mb"] <= scheduling["max_idle_memory_mb"]
            idle_counts[gpu] = idle_counts[gpu] + 1 if idle else 0
            with lock:
                state.setdefault("gpu_observations", {})[str(gpu)] = snapshot
                state["updated_at"] = utc_now()
                atomic_json(state_path, state)
            if idle_counts[gpu] < scheduling["idle_confirmations"]:
                continue
            job_id = pending.popleft()
            attempt = int(state["jobs"][job_id].get("attempt", 0)) + 1
            log_path = output_root / "scheduler" / "logs" / f"{job_id}.gpu{gpu}.attempt{attempt}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, str(JOB_RUNNER), "--manifest", str(manifest_path),
                       "--job-id", job_id, "--gpu", str(gpu), "--output-root", str(output_root)]
            environment = dict(os.environ)
            environment.update(CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER="PCI_BUS_ID")
            handle = log_path.open("ab", buffering=0)
            try:
                process = subprocess.Popen(command, cwd=PROJECT_ROOT, env=environment,
                                           stdout=handle, stderr=subprocess.STDOUT, start_new_session=False)
            except Exception:
                handle.close()
                raise
            active[gpu] = {"process": process, "handle": handle, "job_id": job_id,
                           "attempt": attempt, "log": log_path}
            update(job_id, status="running", gpu=gpu, pid=process.pid, attempt=attempt,
                   started_at=utc_now(), finished_at=None, exit_code=None, result={},
                   gpu_snapshot=snapshot, log=str(log_path))
        if stop_dispatch and not active:
            break
        if pending or active:
            time.sleep(scheduling["poll_seconds"])
    return codes


def build_summary(manifest: dict, state: dict, interaction_payload: dict | None, phase2_decision: dict) -> dict:
    rows = []
    for job in manifest["jobs"]:
        job_id = str(job["id"])
        entry = state["jobs"][job_id]
        result = entry.get("result") or {}
        rows.append(
            {
                "id": job_id,
                "phase": job["phase"],
                "group": job["group"],
                "seed": job["seed"],
                "status": entry.get("status"),
                "gpu": entry.get("gpu"),
                "best_by_model": result.get("best_by_model"),
                "primary_model": result.get("primary_model"),
                "log": entry.get("log"),
            }
        )
    completed = [row for row in rows if row["status"] == "completed" and row["best_by_model"]]
    candidates = []
    for row in completed:
        primary = row["best_by_model"].get(row["primary_model"])
        if primary:
            candidates.append((float(primary["Rank-1"]), row["id"], primary))
    best = max(candidates, default=None)
    return {
        "schema_version": 2,
        "experiment_id": manifest["experiment_id"],
        "updated_at": utc_now(),
        "scheduling_version": manifest["scheduling"].get("version"),
        "selected_gpus": manifest["scheduling"]["selected_gpus"],
        "max_concurrent_jobs": 2,
        "protocol": manifest["protocol"],
        "phase2_decision": phase2_decision,
        "interaction": interaction_payload,
        "best_job": best[1] if best else None,
        "best_metrics": best[2] if best else None,
        "jobs": rows,
    }


def run_scheduler(manifest_path: Path, output_root: Path, results_dir: Path) -> int:
    manifest = load_manifest(manifest_path)
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    scheduler_root = output_root / "scheduler"
    scheduler_root.mkdir(parents=True)
    state_path = scheduler_root / "state.json"
    results_path = output_root / "results.json"
    lock = threading.Lock()

    all_jobs = {str(job["id"]): job for job in manifest["jobs"]}
    state = {
        "schema_version": 2,
        "experiment_id": manifest["experiment_id"],
        "status": "running",
        "started_at": utc_now(),
        "updated_at": utc_now(),
        "scheduling_version": manifest["scheduling"].get("version"),
        "selected_gpus": manifest["scheduling"]["selected_gpus"],
        "max_concurrent_jobs": 2,
        "jobs": {
            job_id: {
                "status": "queued" if job["phase"] == "phase1" else "conditional_pending",
                "gpu": None,
                "pid": None,
                "exit_code": None,
            }
            for job_id, job in all_jobs.items()
        },
    }

    resume_root_value = manifest.get("scheduling", {}).get("resume_from_output_root")
    resume_file_value = manifest.get("scheduling", {}).get("resume_from_state_file")
    if resume_root_value and resume_file_value:
        raise ValueError("choose one resume source")
    if resume_root_value or resume_file_value:
        resume_state_path = (Path(str(resume_file_value)).expanduser().resolve() if resume_file_value else
                             Path(str(resume_root_value)).expanduser().resolve() / "scheduler" / "state.json")
        if not resume_state_path.is_file():
            raise FileNotFoundError(
                f"resume state is missing: {resume_state_path}"
            )
        resume_state = json.loads(resume_state_path.read_text(encoding="utf-8"))
        if resume_state.get("schema_version") != 2:
            raise ValueError("resume state schema_version must be 2")
        resume_jobs = resume_state.get("jobs") or {}
        for job_id in all_jobs:
            previous = resume_jobs.get(job_id) or {}
            previous_result = previous.get("result") or {}
            if (
                previous.get("status") == "completed"
                and previous_result.get("status") == "completed"
                and previous_result.get("best_by_model")
                and previous_result.get("implementation_version") == manifest.get("implementation_version")
                and previous_result.get("completed_epochs") == manifest["training"]["completed_epochs"]
                and all(Path(row.get("checkpoint_path", "")).is_file() for row in previous_result["best_by_model"].values())
            ):
                state["jobs"][job_id].update(
                    {
                        "status": "completed",
                        "gpu": previous.get("gpu"),
                        "pid": None,
                        "exit_code": previous.get("exit_code", 0),
                        "finished_at": previous.get("finished_at"),
                        "result": previous_result,
                        "log": previous.get("log"),
                        "resumed_from": str(resume_state_path),
                    }
                )
    atomic_json(output_root / "plan.json", {
        "schema_version": 2,
        "experiment_id": manifest["experiment_id"],
        "worktree": str(PROJECT_ROOT),
        "implementation_version": manifest["implementation_version"],
        "manifest": str(manifest_path),
        "checkpoint": manifest["stage_a"]["checkpoint"],
        "protocol": manifest["protocol"],
        "training": manifest["training"],
        "scheduling": manifest["scheduling"],
        "phase1_jobs": [job["id"] for job in manifest["jobs"] if job["phase"] == "phase1"],
        "phase2_jobs": [job["id"] for job in manifest["jobs"] if job["phase"] == "phase2"],
    })
    atomic_json(state_path, state)

    phase1_codes = run_phase(
        "phase1", manifest, manifest_path, output_root, state, state_path, lock
    )
    phase1_results = {
        job_id: (state["jobs"][job_id].get("result") or {})
        for job_id in phase_jobs(manifest, "phase1")
    }
    phase1_failed = any(
        state["jobs"][job_id].get("status") != "completed"
        for job_id in phase_jobs(manifest, "phase1")
    )
    interaction_payload = None
    phase2_decision = {
        "condition": "phase1_mean_rank1_interaction_positive",
        "triggered": False,
        "reason": "phase1_failed",
    }

    if not phase1_failed:
        interaction_payload = interaction(phase1_results)
        atomic_json(output_root / "interaction.json", interaction_payload)
        mean_i_ema = float(interaction_payload["summary"]["Rank-1_I_ema"]["mean"])
        phase2_decision = {
            "condition": "phase1_mean_rank1_interaction_positive",
            "triggered": mean_i_ema > 0.0,
            "mean_rank1_I_ema": mean_i_ema,
            "reason": "positive" if mean_i_ema > 0.0 else "non_positive",
        }
        if phase2_decision["triggered"]:
            for job_id in phase_jobs(manifest, "phase2"):
                if state["jobs"][job_id]["status"] != "completed":
                    state["jobs"][job_id]["status"] = "queued"
            atomic_json(state_path, state)
            run_phase(
                "phase2", manifest, manifest_path, output_root, state, state_path, lock
            )
        else:
            for job_id in phase_jobs(manifest, "phase2"):
                state["jobs"][job_id].update(
                    {"status": "skipped", "reason": "phase1 interaction was non-positive"}
                )
            atomic_json(state_path, state)
    else:
        for job_id in phase_jobs(manifest, "phase2"):
            state["jobs"][job_id].update(
                {"status": "skipped", "reason": "phase1 had failed jobs"}
            )
        atomic_json(state_path, state)

    summary = build_summary(manifest, state, interaction_payload, phase2_decision)
    atomic_json(results_path, summary)
    state["status"] = (
        "completed"
        if all(
            entry.get("status") in {"completed", "skipped"}
            for entry in state["jobs"].values()
        )
        else "completed_with_failures"
    )
    state["finished_at"] = utc_now()
    state["updated_at"] = utc_now()
    atomic_json(state_path, state)

    best = summary.get("best_metrics")
    if state["status"] != "completed":
        return 1
    if not best:
        raise RuntimeError("no completed ablation job produced metrics")
    results_dir = results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    runtime_metrics = {
        "primary_metric": float(best["Rank-1"]),
        "metrics": {
            "Rank-1": float(best["Rank-1"]),
            "mAP": float(best["mAP"]),
            "mINP": float(best["mINP"]),
            "completed_jobs": float(sum(entry.get("status") == "completed" for entry in state["jobs"].values())),
            "phase2_triggered": 1.0 if phase2_decision["triggered"] else 0.0,
            "phase1_mean_rank1_I_ema": float(
                interaction_payload["summary"]["Rank-1_I_ema"]["mean"]
            ) if interaction_payload else 0.0,
        },
    }
    atomic_json(results_dir / "metrics.json", runtime_metrics)
    return 0 if state["status"] == "completed" else 1


def configure_process_lifecycle():
    # The generic runtime launches this scheduler in its own process group.
    # On termination, relay to the whole group (including DataLoader workers).
    if os.getpgrp() != os.getpid():
        os.setpgrp()
    def terminate_group(signum, frame):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        os.killpg(os.getpgrp(), signal.SIGTERM)
        os._exit(128 + signum)
    signal.signal(signal.SIGTERM, terminate_group)
    signal.signal(signal.SIGINT, terminate_group)
    parent_pid = os.getppid()
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "cannot register scheduler parent-death signal")
    if os.getppid() != parent_pid:
        terminate_group(signal.SIGTERM, None)


def main(argv=None) -> int:
    configure_process_lifecycle()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "run", "status"))
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest_path = args.manifest.resolve()
    manifest = load_manifest(manifest_path)
    if args.action == "plan":
        print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.action == "status":
        state_path = args.output_root.resolve() / "scheduler" / "state.json"
        print(state_path.read_text(encoding="utf-8"))
        return 0
    results_dir = Path(os.environ.get("AR2_RESULTS_DIR", str(args.output_root / "runtime-results")))
    return run_scheduler(manifest_path, args.output_root, results_dir)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        sys.stderr.flush()
        sys.stdout.flush()
        # Use the registered group handler on unexpected scheduler errors too;
        # otherwise already-launched jobs can survive a failed scheduler.
        os.kill(os.getpid(), signal.SIGTERM)
        raise
