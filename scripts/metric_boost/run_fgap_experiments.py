#!/usr/bin/env python
"""Provenance-gated, lease-safe runner for the three FGAP experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
GIT_ROOT = REPO_ROOT.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import (
    E4_CONFIG_PATH,
    GpuLease,
    atomic_write_json,
    atomic_write_text,
    atomic_write_yaml,
    experiment_dir,
    experiment_status_path,
    idle_gpu_ids,
    load_yaml,
    prepare_runtime_config,
    query_gpu_states,
    read_json,
    resolve_e4_checkpoint,
    sha256_file,
    utc_now,
)

DEFAULT_PLAN = REPO_ROOT / "configs/metric_boost/fgap_experiments.yaml"
PLAN_ROOT = REPO_ROOT / "reports/metric_boost/plans"
REQUIRED_PROVENANCE = (
    "design.md", "runtime_config.yaml", "config_diff.yaml", "code.patch",
    "source_state.json", "environment.json", "dataset_fingerprint.json", "command.txt",
)


def _git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(GIT_ROOT), *args], text=True).strip()


def load_plan(path: Path = DEFAULT_PLAN):
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    common = dict(payload["common_overrides"])
    rows = []
    for row in payload["experiments"]:
        overrides = dict(common)
        overrides.update(dict(row.get("overrides", {})))
        rows.append({
            "id": str(row["id"]),
            "stage": str(payload["stage"]),
            "plan_id": str(payload["plan_id"]),
            "baseline_experiment_id": str(payload["baseline_experiment_id"]),
            "baseline_runtime_config": str(payload["baseline_runtime_config"]),
            "validity": str(payload["selection_validity"]),
            "hypothesis": str(row["hypothesis"]),
            "overrides": overrides,
        })
    if len(rows) != 3 or len({row["id"] for row in rows}) != 3:
        raise ValueError("FGAP plan must contain exactly three unique experiments")
    return rows


def _checkpoint_plan_path(plan_id: str) -> Path:
    return PLAN_ROOT / plan_id / "launch_plan.json"


def prepare_launch_plan(plan_path: Path) -> Path:
    rows = load_plan(plan_path)
    checkpoint = resolve_e4_checkpoint()
    payload = {
        "plan_id": rows[0]["plan_id"],
        "created_at": utc_now(),
        "git_commit_sha": _git("rev-parse", "HEAD"),
        "experiment_ids": [row["id"] for row in rows],
        "e4_checkpoint": checkpoint,
    }
    path = _checkpoint_plan_path(rows[0]["plan_id"])
    atomic_write_json(path, payload)
    return path


def _load_checkpoint_plan(path: Path, expected_plan_id: str) -> Dict[str, Any]:
    payload = read_json(path, {})
    if payload.get("plan_id") != expected_plan_id:
        raise ValueError(f"Checkpoint plan mismatch: {payload.get('plan_id')} != {expected_plan_id}")
    checkpoint = dict(payload["e4_checkpoint"])
    checkpoint_path = Path(checkpoint["checkpoint"])
    if not checkpoint_path.is_file() or sha256_file(checkpoint_path) != checkpoint["sha256"]:
        raise RuntimeError("Resolved E4 checkpoint is missing or its SHA-256 changed")
    return checkpoint


def _wait_for_gpu(experiment_id: str, poll_seconds: float):
    while True:
        for gpu_id in idle_gpu_ids(query_gpu_states()):
            lease = GpuLease(gpu_id)
            try:
                lease.__enter__()
            except RuntimeError:
                continue
            return gpu_id, lease
        status = read_json(experiment_status_path(experiment_id), {})
        status.update({"status": "pending", "waiting_for_gpu": True, "updated_at": utc_now()})
        atomic_write_json(experiment_status_path(experiment_id), status)
        time.sleep(max(2.0, poll_seconds))


def _directory_manifest(path: Path) -> Dict[str, Any]:
    digest = hashlib.sha256()
    count = 0
    total = 0
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        relative = item.relative_to(path).as_posix()
        size = item.stat().st_size
        digest.update(f"{relative}\0{size}\n".encode("utf-8"))
        count += 1
        total += size
    return {"resolved_path": str(path.resolve()), "file_count": count, "size_bytes": total,
            "manifest_algorithm": "sha256(relative_path,NUL,size)", "manifest_sha256": digest.hexdigest()}


def _dataset_fingerprint(runtime: Mapping[str, Any], checkpoint: Mapping[str, Any]):
    result = {"generated_at": utc_now(), "inputs": {}}
    for logical, key in (("sysu_images", "sysu_data_path"), ("sysu_text", "text_data_root")):
        path = Path(str(runtime[key])).resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"Required dataset input missing: {logical}={path}")
        result["inputs"][logical] = _directory_manifest(path)
    checkpoint_path = Path(checkpoint["checkpoint"])
    result["inputs"]["e4_checkpoint"] = {
        "resolved_path": str(checkpoint_path.resolve()), "size_bytes": checkpoint_path.stat().st_size,
        "sha256": checkpoint["sha256"],
    }
    return result


def _config_diff(baseline: Mapping[str, Any], runtime: Mapping[str, Any]):
    changed, unchanged_controls = {}, {}
    controls = {"pa", "loss_names", "cross_modal_hard_weight", "training_mode", "test_modality",
                "visual_pooling", "test_mode", "gall_mode", "gallery_trials"}
    for key in sorted(set(baseline) | set(runtime)):
        if baseline.get(key) != runtime.get(key):
            changed[key] = {"baseline": baseline.get(key), "experiment": runtime.get(key)}
        elif key in controls:
            unchanged_controls[key] = runtime.get(key)
    return {"baseline": "TRAIN-3-H1", "changed": changed, "unchanged_controls": unchanged_controls}


def _append_event(event_path: Path, experiment_id: str, event_type: str, **payload):
    event = {"schema_version": 1, "event_id": f"{experiment_id}:{event_type}:{time.time_ns()}",
             "experiment_id": experiment_id, "event_type": event_type, "timestamp": utc_now(), "attempt": 1}
    event.update(payload)
    with event_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _materialize_provenance(spec, runtime_path: Path, command, checkpoint, gpu_id: int, plan_path: Path):
    run_dir = experiment_dir(spec["id"])
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Immutable manifest already exists: {manifest_path}")
    runtime = load_yaml(runtime_path)
    baseline_path = (REPO_ROOT / spec["baseline_runtime_config"]).resolve()
    baseline = load_yaml(baseline_path)
    head = _git("rev-parse", "HEAD")
    branch = _git("branch", "--show-current")
    tracked_patch = subprocess.check_output(["git", "-C", str(GIT_ROOT), "diff", "--binary", "HEAD"], text=True)
    untracked = _git("ls-files", "--others", "--exclude-standard").splitlines()
    relevant_untracked = [p for p in untracked if p.startswith(("src/salt_vi/engine/", "configs/", "scripts/",
                                                                 "src/salt_vi/optim/", "src/salt_vi/models/", "tests/"))]
    if relevant_untracked:
        raise RuntimeError(f"Algorithm-relevant untracked files refuse launch: {relevant_untracked}")
    changed_last_commit = _git("diff", "--name-only", "HEAD^", "HEAD", "--", "src/salt_vi").splitlines()
    design = (
        f"# {spec['id']}\n\nHypothesis: {spec['hypothesis']}\n\n"
        "Declared baseline: TRAIN-3-H1. Intervention: asymmetric weights over the six cross-modal hard-loss pairs; "
        "P2 additionally unfreezes the final two visual blocks from epoch 3, while P3 uses modality-specific BN.\n\n"
        "Controlled variables: pa=0.5, id+cross_modal_hard, RGB_IR_Text, Fusion evaluation, cls pooling, "
        "SYSU all-search/single-shot/10 trials, seed and E4 initialization.\n\n"
        "Selection rule: report Rank-1, mAP and mINP without promoting test-informed selection to standard validity.\n\n"
        f"Validity: {spec['validity']}. Warm-start chaining is forbidden; initialization is E4 SHA-256 {checkpoint['sha256']}.\n"
    )
    atomic_write_text(run_dir / "design.md", design)
    atomic_write_yaml(run_dir / "config_diff.yaml", _config_diff(baseline, runtime))
    atomic_write_text(run_dir / "code.patch", tracked_patch)
    atomic_write_json(run_dir / "source_state.json", {
        "repository_root": str(GIT_ROOT), "head": head, "branch": branch,
        "baseline_commit": read_json(REPO_ROOT / "reports/metric_boost/runs/TRAIN-3-H1/manifest.json", {}).get("git_commit_sha"),
        "worktree_dirty": bool(_git("status", "--porcelain")), "tracked_patch_empty": not bool(tracked_patch),
        "untracked_paths": untracked, "algorithm_relevant_untracked_paths": relevant_untracked,
        "changed_paths_in_effective_commit": changed_last_commit, "submodules": _git("submodule", "status").splitlines(),
    })
    freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True, errors="replace").splitlines()
    gpu_state = next(state for state in query_gpu_states() if int(state["index"]) == gpu_id)
    atomic_write_json(run_dir / "environment.json", {
        "python_executable": sys.executable, "python_version": sys.version, "platform": platform.platform(),
        "cuda_visible_devices": str(gpu_id), "gpu": gpu_state, "pip_freeze": freeze,
        "secrets_redacted": True,
    })
    atomic_write_json(run_dir / "dataset_fingerprint.json", _dataset_fingerprint(runtime, checkpoint))
    atomic_write_text(run_dir / "command.txt", shlex.join(command) + "\nworking_directory=" + str(REPO_ROOT) + "\n")
    hashes = {name: sha256_file(run_dir / name) for name in REQUIRED_PROVENANCE}
    atomic_write_json(run_dir / "artifact_hashes.json", {"algorithm": "sha256", "files": hashes})
    artifact_hash = sha256_file(run_dir / "artifact_hashes.json")
    diff = _config_diff(baseline, runtime)["changed"]
    manifest = {
        "logging_contract_version": 1, "provenance_contract_version": 1,
        "experiment_id": spec["id"], "stage": spec["stage"], "group": "FGAP", "attempt": 1,
        "parent_experiment_id": None, "baseline_experiment_id": spec["baseline_experiment_id"],
        "config_path": str(plan_path), "runtime_config": str(runtime_path), "output_path": runtime["output_path"],
        "log_path": str(run_dir / "launcher.log"), "baseline_git_commit_sha": read_json(
            REPO_ROOT / "reports/metric_boost/runs/TRAIN-3-H1/manifest.json", {}).get("git_commit_sha"),
        "git_commit_sha": head, "worktree_dirty": bool(_git("status", "--porcelain")), "command": command,
        "working_directory": str(REPO_ROOT), "start_requested_at": utc_now(), "dataset": "sysu",
        "protocol": {"test_mode": "all", "gall_mode": "single", "gallery_trials": 10, "test_modality": "Fusion"},
        "seed": int(runtime["seed"]), "max_epoch": int(runtime["total_train_epoch"]), "epoch_index_origin": 0,
        "planned_metric_names": ["Rank-1", "mAP", "mINP"],
        "planned_loss_names": ["id_loss", "cross_modal_hard_loss", "total_loss"],
        "selection_validity": spec["validity"], "hypothesis": spec["hypothesis"],
        "design_summary": spec["hypothesis"],
        "config_change_summary": f"{len(diff)} resolved keys differ from TRAIN-3-H1",
        "code_change_summary": "Committed FGAP weighted-pair, QBN warm-start, event and provenance implementation",
        "changed_config_keys": diff,
        "changed_code": [{"path": path, "symbols": [], "purpose": "Effective FGAP implementation commit"}
                         for path in changed_last_commit],
        "artifact_hashes_path": str(run_dir / "artifact_hashes.json"),
        "artifact_hashes_sha256": artifact_hash, "provenance_artifacts": hashes,
        "e4_checkpoint": {"path": checkpoint["checkpoint"], "sha256": checkpoint["sha256"]},
        "reproducibility_status": "prelaunch-verified-exact-commit",
    }
    atomic_write_json(manifest_path, manifest)
    validate_provenance(run_dir)
    return manifest


def validate_provenance(run_dir: Path):
    manifest = read_json(run_dir / "manifest.json", {})
    hashes = read_json(run_dir / "artifact_hashes.json", {}).get("files", {})
    for field in ("logging_contract_version", "provenance_contract_version", "experiment_id", "git_commit_sha"):
        if field not in manifest:
            raise ValueError(f"Manifest missing required field: {field}")
    for name in REQUIRED_PROVENANCE:
        path = run_dir / name
        if not path.is_file() or hashes.get(name) != sha256_file(path):
            raise ValueError(f"Provenance hash validation failed: {name}")
    if manifest.get("artifact_hashes_sha256") != sha256_file(run_dir / "artifact_hashes.json"):
        raise ValueError("artifact_hashes.json digest mismatch")
    return True


def run_one(spec, plan_path: Path, checkpoint_plan: Path, poll_seconds: float):
    run_dir = experiment_dir(spec["id"])
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = experiment_status_path(spec["id"])
    if (run_dir / "manifest.json").exists():
        raise FileExistsError(f"Refusing to overwrite formal experiment {spec['id']}")
    checkpoint = _load_checkpoint_plan(checkpoint_plan, spec["plan_id"])
    status = {"experiment_id": spec["id"], "experiment": spec["id"], "phase": spec["stage"],
              "status": "pending", "updated_at": utc_now(), "start_time": None, "end_time": None,
              "runner_pid": os.getpid(), "trainer_pid": None, "gpu": None, "last_completed_epoch": None,
              "return_code": None, "error": None, "Rank-1": None, "mAP": None, "mINP": None,
              "best_epoch": None, "checkpoint": checkpoint["checkpoint"], "validity": spec["validity"]}
    atomic_write_json(status_path, status)
    gpu_id, lease = _wait_for_gpu(spec["id"], poll_seconds)
    events_path = run_dir / "events.jsonl"
    events_path.touch(exist_ok=False)
    try:
        baseline = load_yaml((REPO_ROOT / spec["baseline_runtime_config"]).resolve())
        overrides = dict(baseline)
        overrides.update(spec["overrides"])
        overrides.update({
            "training_weight_init": checkpoint["checkpoint"],
            "output_path": str(run_dir / "model_output") + "/",
            "CUDA_VISIBLE_DEVICES": str(gpu_id), "gpu_id": "0",
            "metric_events_path": str(events_path), "metric_experiment_id": spec["id"], "metric_attempt": 1,
        })
        for forbidden in ("metric_boost_resume_epoch", "auto_resume_training_from_lastest_step"):
            overrides.pop(forbidden, None)
        overrides["resume_train_epoch"] = -1
        overrides["auto_resume_training_from_lastest_step"] = False
        runtime_path = prepare_runtime_config(spec["id"], E4_CONFIG_PATH, overrides, gpu_id=gpu_id)
        command = [sys.executable, str(REPO_ROOT / "scripts" / "train.py"), "--config_select", str(runtime_path)]
        _materialize_provenance(spec, runtime_path, command, checkpoint, gpu_id, plan_path)
        status.update({"status": "running", "waiting_for_gpu": False, "gpu": gpu_id, "start_time": utc_now(),
                       "updated_at": utc_now(), "runtime_config": str(runtime_path), "command": command,
                       "command_shell": shlex.join(command), "log_path": str(run_dir / "launcher.log"),
                       "git_commit_sha": _git("rev-parse", "HEAD"), "warm_start": "E4-only"})
        atomic_write_json(status_path, status)
        _append_event(events_path, spec["id"], "run_started", command=command, git_commit_sha=status["git_commit_sha"],
                      gpu=gpu_id, seed=int(overrides["seed"]), max_epoch=int(overrides["total_train_epoch"]))
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        with (run_dir / "launcher.log").open("a", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=str(REPO_ROOT), stdout=log, stderr=subprocess.STDOUT, env=env)
            status["trainer_pid"] = process.pid
            status["updated_at"] = utc_now()
            atomic_write_json(status_path, status)
            return_code = process.wait()
        status.update({"return_code": return_code, "end_time": utc_now(), "updated_at": utc_now()})
        if return_code != 0:
            status.update({"status": "failed", "error": f"Training returned {return_code}"})
            atomic_write_json(status_path, status)
            _append_event(events_path, spec["id"], "run_finished", status="failed", return_code=return_code,
                          completion_reason="process_exit", final_metrics={})
            raise RuntimeError(status["error"])
        status["status"] = "completed_pending_summary"
        atomic_write_json(status_path, status)
        from summarize_results import write_outputs
        write_outputs()
        status = read_json(status_path, status)
        checkpoint_path = Path(status["checkpoint"]) if status.get("checkpoint") else None
        if checkpoint_path and checkpoint_path.is_file():
            _append_event(events_path, spec["id"], "checkpoint_saved", epoch=int(status["best_epoch"]),
                          path=str(checkpoint_path), sha256=sha256_file(checkpoint_path), is_best=True,
                          selection_metric="Rank-1")
        _append_event(events_path, spec["id"], "run_finished", status=status.get("status"), return_code=0,
                      completion_reason="completed", final_metrics={key: status.get(key) for key in ("Rank-1", "mAP", "mINP")},
                      selected_checkpoint=status.get("checkpoint"))
        return status
    except Exception as exc:
        current = read_json(status_path, status)
        if current.get("status") not in {"failed", "succeeded"}:
            current.update({"status": "blocked" if not (run_dir / "manifest.json").exists() else "failed",
                            "error": repr(exc), "end_time": utc_now(), "updated_at": utc_now()})
            atomic_write_json(status_path, current)
        raise
    finally:
        lease.__exit__(None, None, None)


def launch_tmux_sessions(plan_path: Path, checkpoint_plan: Path, poll_seconds: float):
    launched = []
    for spec in load_plan(plan_path):
        session = spec["id"].lower().replace("_", "-")
        exists = subprocess.run(["tmux", "has-session", "-t", session], stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL).returncode == 0
        if exists:
            raise RuntimeError(f"tmux session already exists: {session}")
        run_dir = experiment_dir(spec["id"])
        run_dir.mkdir(parents=True, exist_ok=True)
        runner_log = run_dir / "scheduler_runner.log"
        command = [sys.executable, str(Path(__file__).resolve()), "--plan", str(plan_path), "run-one",
                   "--experiment", spec["id"], "--checkpoint-plan", str(checkpoint_plan),
                   "--poll-seconds", str(poll_seconds)]
        shell_command = f"exec {shlex.join(command)} >> {shlex.quote(str(runner_log))} 2>&1"
        subprocess.run(["tmux", "new-session", "-d", "-s", session, "-c", str(REPO_ROOT),
                        "/bin/bash", "-lc", shell_command], check=True)
        launched.append({"experiment_id": spec["id"], "session": session,
                         "runner_log": str(runner_log), "command": command})
    return launched


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare-launch-plan")
    launch = sub.add_parser("launch-tmux")
    launch.add_argument("--checkpoint-plan", type=Path, required=True)
    launch.add_argument("--poll-seconds", type=float, default=15.0)
    run = sub.add_parser("run-one")
    run.add_argument("--experiment", required=True)
    run.add_argument("--checkpoint-plan", type=Path, required=True)
    run.add_argument("--poll-seconds", type=float, default=15.0)
    args = parser.parse_args()
    if args.command == "prepare-launch-plan":
        path = prepare_launch_plan(args.plan.resolve())
        print(json.dumps({"ok": True, "launch_plan": str(path)}, sort_keys=True))
        return
    if args.command == "launch-tmux":
        launched = launch_tmux_sessions(args.plan.resolve(), args.checkpoint_plan.resolve(), args.poll_seconds)
        print(json.dumps({"ok": True, "launched": launched}, indent=2, sort_keys=True))
        return
    rows = {row["id"]: row for row in load_plan(args.plan.resolve())}
    if args.experiment not in rows:
        parser.error(f"Unknown FGAP experiment: {args.experiment}")
    result = run_one(rows[args.experiment], args.plan.resolve(), args.checkpoint_plan.resolve(), args.poll_seconds)
    print(json.dumps({"ok": True, "experiment": args.experiment, "status": result.get("status")}, sort_keys=True))


if __name__ == "__main__":
    main()
