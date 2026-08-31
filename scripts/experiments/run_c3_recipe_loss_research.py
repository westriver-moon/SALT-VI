#!/usr/bin/env python3
"""Validate, schedule, and summarize the fixed-input C3 recipe/loss study."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from typing import Iterable

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from salt_vi.config.validation import validate_runtime_config
from salt_vi.diagnostics import run_c3_recipe_diagnostic
from salt_vi.utils.utils import load_train_configs


DEFAULT_MANIFEST = (
    PROJECT_ROOT / "configs/pipelines/c3_recipe_loss_research_20260829.yaml"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/home/lab929/ybj/experiments/stage_a/"
    "SALT-VI-c3-recipe-loss-research-20260829"
)
OUTPUT_ENV = "SALT_C3_RECIPE_OUTPUT_ROOT"
RUNTIME_ONLY_OVERRIDES = {
    "experiment_name",
    "metric_experiment_id",
    "metric_events_path",
    "output_root",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def canonical(value):
    if isinstance(value, (tuple, list)):
        return [canonical(item) for item in value]
    if isinstance(value, dict):
        return {str(key): canonical(item) for key, item in value.items()}
    return value


def load_manifest(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 2:
        raise ValueError("manifest schema_version must be integer 2")
    required = {
        "experiment_id",
        "base_config",
        "protocol",
        "selection",
        "fixed_config",
        "allowed_overrides",
        "variants",
        "diagnostics",
        "scheduling",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError("manifest is missing: " + ", ".join(missing))
    selection = payload["selection"]
    if not isinstance(selection, dict):
        raise ValueError("manifest selection must be a mapping")
    if selection.get("policy") != "best-evaluated-epoch":
        raise ValueError("selection.policy must be best-evaluated-epoch")
    if selection.get("primary_metric") not in {"Rank-1", "mAP", "mINP"}:
        raise ValueError("selection.primary_metric is unsupported")
    if selection.get("mode") != "maximize":
        raise ValueError("selection.mode must be maximize")
    tie_breakers = selection.get("tie_breakers")
    if (
        not isinstance(tie_breakers, list)
        or any(metric not in {"Rank-1", "mAP", "mINP"} for metric in tie_breakers)
        or len(tie_breakers) != len(set(tie_breakers))
        or selection["primary_metric"] in tie_breakers
    ):
        raise ValueError("selection.tie_breakers must be distinct supported metrics")
    if selection.get("final_tie_breaker") != "latest_epoch":
        raise ValueError("selection.final_tie_breaker must be latest_epoch")
    variants = payload["variants"]
    if not isinstance(variants, dict) or not variants:
        raise ValueError("manifest variants must be a non-empty mapping")
    diagnostics = payload["diagnostics"]
    if not isinstance(diagnostics, dict) or not diagnostics:
        raise ValueError("manifest diagnostics must be a non-empty mapping")
    if len(variants) != 13 or len(diagnostics) != 2:
        raise ValueError("C3 study requires exactly 13 trainings and 2 diagnostics")
    allowed = set(payload["allowed_overrides"])
    for name, item in variants.items():
        if not isinstance(item, dict) or not isinstance(item.get("overrides"), dict):
            raise ValueError(f"variant {name} must declare overrides")
        unknown = sorted(set(item["overrides"]) - allowed)
        if unknown:
            raise ValueError(
                f"variant {name} overrides non-recipe keys: {', '.join(unknown)}"
            )
    supported_diagnostics = {"loss_gradient_conflict", "modality_information"}
    for name, item in diagnostics.items():
        if not isinstance(item, dict):
            raise ValueError(f"diagnostic {name} must be a mapping")
        if item.get("type") not in supported_diagnostics:
            raise ValueError(f"diagnostic {name} has an unsupported type")
        if item.get("variant") not in variants:
            raise ValueError(f"diagnostic {name} references an unknown variant")
        if not isinstance(item.get("batches"), int) or item["batches"] < 1:
            raise ValueError(f"diagnostic {name} batches must be a positive integer")
    job_order = payload["scheduling"].get("job_order")
    expected_jobs = set(variants) | set(diagnostics)
    if (
        not isinstance(job_order, list)
        or len(job_order) != len(set(job_order))
        or set(job_order) != expected_jobs
    ):
        raise ValueError(
            "scheduling.job_order must contain every training and diagnostic once"
        )
    payload["_manifest_path"] = str(path.resolve())
    return payload


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def select_variants(
    manifest: dict,
    *,
    names: Iterable[str] | None = None,
    phase: str | None = None,
) -> list[str]:
    variants = manifest["variants"]
    selected = list(names or variants)
    unknown = sorted(set(selected) - set(variants))
    if unknown:
        raise ValueError("unknown variants: " + ", ".join(unknown))
    if phase is not None:
        selected = [name for name in selected if variants[name]["phase"] == phase]
    if not selected:
        raise ValueError("variant selection is empty")
    return selected


def _temporary_environment(**updates):
    class EnvironmentContext:
        def __enter__(self):
            self.previous = dict(os.environ)
            os.environ.update({key: str(value) for key, value in updates.items()})

        def __exit__(self, exc_type, exc, traceback):
            os.environ.clear()
            os.environ.update(self.previous)

    return EnvironmentContext()


def resolve_variant(manifest: dict, name: str, output_root: Path) -> dict:
    base_path = PROJECT_ROOT / manifest["base_config"]
    if not base_path.is_file():
        raise FileNotFoundError(f"base config is missing: {base_path}")
    with _temporary_environment(**{OUTPUT_ENV: output_root}):
        config = load_train_configs(str(base_path))
    overrides = deepcopy(manifest["variants"][name]["overrides"])
    for key, value in overrides.items():
        setattr(config, key, value)

    variant_root = output_root / "runs" / name
    events_path = output_root / "events" / f"{name}.jsonl"
    runtime = {
        "experiment_name": name,
        "metric_experiment_id": f"C3-RECIPE-{name.upper().replace('_', '-')}",
        "metric_events_path": str(events_path),
        "output_root": str(variant_root),
    }
    for key, value in runtime.items():
        setattr(config, key, value)
    validate_runtime_config(config)

    mismatches = []
    for key, expected in manifest["fixed_config"].items():
        actual = canonical(getattr(config, key))
        if actual != canonical(expected):
            mismatches.append(f"{key}: expected {expected!r}, got {actual!r}")
    if mismatches:
        raise ValueError(
            f"variant {name} changed the fixed C3 input/runtime:\n  "
            + "\n  ".join(mismatches)
        )
    return {
        "name": name,
        "phase": manifest["variants"][name]["phase"],
        "purpose": manifest["variants"][name]["purpose"],
        "base_config": str(base_path.relative_to(PROJECT_ROOT)),
        "overrides": overrides,
        "runtime_overrides": runtime,
        "events_path": str(events_path),
        "output_root": str(variant_root),
        "resolved_recipe": {
            key: canonical(getattr(config, key))
            for key in manifest["allowed_overrides"]
        },
        "fixed_config": {
            key: canonical(getattr(config, key))
            for key in manifest["fixed_config"]
        },
    }


def resolve_diagnostic(
    manifest: dict,
    name: str,
    output_root: Path,
    resolved_variants: dict,
) -> dict:
    item = deepcopy(manifest["diagnostics"][name])
    variant = item.pop("variant")
    return {
        "name": name,
        "job_type": "diagnostic",
        "type": item.pop("type"),
        "purpose": item.pop("purpose"),
        "variant": variant,
        "output_path": str(output_root / "diagnostics" / f"{name}.json"),
        "resolved_variant": deepcopy(resolved_variants[variant]),
        **item,
    }


def build_plan(
    manifest: dict,
    names: list[str],
    output_root: Path,
    diagnostic_names: list[str] | None = None,
) -> dict:
    resolved = {
        name: resolve_variant(manifest, name, output_root) for name in names
    }
    diagnostic_names = (
        list(manifest["diagnostics"])
        if diagnostic_names is None
        else list(diagnostic_names)
    )
    unknown_diagnostics = sorted(
        set(diagnostic_names) - set(manifest["diagnostics"])
    )
    if unknown_diagnostics:
        raise ValueError(
            "unknown diagnostics: " + ", ".join(unknown_diagnostics)
        )
    diagnostics = {
        name: resolve_diagnostic(
            manifest,
            name,
            output_root,
            resolved
            if manifest["diagnostics"][name]["variant"] in resolved
            else {
                manifest["diagnostics"][name]["variant"]: resolve_variant(
                    manifest,
                    manifest["diagnostics"][name]["variant"],
                    output_root,
                )
            },
        )
        for name in diagnostic_names
    }
    selected_jobs = set(resolved) | set(diagnostics)
    job_order = [
        name
        for name in manifest["scheduling"]["job_order"]
        if name in selected_jobs
    ]
    return {
        "schema_version": 2,
        "experiment_id": manifest["experiment_id"],
        "created_at": utc_now(),
        "worktree": str(PROJECT_ROOT),
        "base_config": manifest["base_config"],
        "output_root": str(output_root),
        "protocol": manifest["protocol"],
        "selection": canonical(manifest["selection"]),
        "selection_policy": manifest["selection"]["policy"],
        "variants": resolved,
        "diagnostics": diagnostics,
        "job_order": job_order,
        "job_counts": {
            "training": len(resolved),
            "diagnostic": len(diagnostics),
            "total": len(job_order),
        },
    }


def gpu_processes(gpu: int) -> list[dict]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(gpu),
            "--query-compute-apps=pid,used_memory,process_name",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "nvidia-smi query failed")
    rows = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",", 2)]
        if parts and parts[0].isdigit():
            rows.append(
                {
                    "pid": int(parts[0]),
                    "used_memory_mb": int(parts[1]),
                    "process_name": parts[2],
                }
            )
    return rows


def gpu_memory(gpu: int) -> int:
    result = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(gpu),
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(result.stdout.strip().splitlines()[0])


def require_idle_gpu(gpu: int, maximum_memory_mb: int) -> dict:
    processes = gpu_processes(gpu)
    used = gpu_memory(gpu)
    if processes or used > maximum_memory_mb:
        raise RuntimeError(
            f"GPU {gpu} is not idle: used_memory_mb={used}, processes={processes}"
        )
    return {"gpu": gpu, "used_memory_mb": used, "processes": processes}


def yaml_scalar(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def training_command(resolved: dict, gpu: int) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/train.py"),
        "--config_select",
        resolved["base_config"],
        "--CUDA_VISIBLE_DEVICES",
        str(gpu),
        "--gpu_id",
        "0",
    ]
    combined = dict(resolved["overrides"])
    combined.update(resolved["runtime_overrides"])
    for key, value in combined.items():
        command.extend(("--set", f"{key}={yaml_scalar(value)}"))
    return command


def diagnostic_command(
    resolved: dict,
    manifest_path: Path,
    output_root: Path,
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "diagnostic",
        "--manifest",
        str(manifest_path),
        "--output-root",
        str(output_root),
        "--diagnostic",
        resolved["name"],
    ]


def run_logged_process(
    command: list[str],
    log_path: Path,
    *,
    environment: dict | None = None,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return int(completed.returncode)


def evaluation_events(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    events = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("event_type") == "eval_epoch" and isinstance(
            event.get("metrics"), dict
        ):
            events.append(event)
    return events


def metric_rank_key(metrics: dict, selection: dict) -> tuple:
    order = [selection["primary_metric"], *selection["tie_breakers"]]
    return tuple(float(metrics[name]) for name in order) + (int(metrics["epoch"]),)


def best_evaluated_metrics(resolved: dict, selection: dict) -> dict:
    events = evaluation_events(Path(resolved["events_path"]))
    if not events:
        raise RuntimeError(f"{resolved['name']} has no evaluation events")
    candidates = []
    for event in events:
        protocol = event.get("protocol_spec") or {}
        trials = protocol.get("trial_ids") or []
        if (
            int(protocol.get("gallery_trials", len(trials))) != 10
            or len(trials) != 10
        ):
            raise RuntimeError(
                f"{resolved['name']} did not record the required "
                "10-gallery-trial aggregate"
            )
        metrics = event["metrics"]
        for key in ("Rank-1", "mAP", "mINP"):
            if key not in metrics:
                raise RuntimeError(f"{resolved['name']} is missing metric {key}")
        candidates.append(
            {
                "epoch": int(event["epoch"]),
                "Rank-1": float(metrics["Rank-1"]),
                "mAP": float(metrics["mAP"]),
                "mINP": float(metrics["mINP"]),
                "gallery_trials": 10,
                "protocol": event.get("protocol"),
            }
        )
    selected = max(candidates, key=lambda item: metric_rank_key(item, selection))
    return {
        **selected,
        "selection_metric": selection["primary_metric"],
        "evaluated_epochs": sorted({item["epoch"] for item in candidates}),
    }


def run_training(
    resolved: dict,
    gpu: int,
    selection: dict,
    output_root: Path,
) -> dict:
    command = training_command(resolved, gpu)
    started = utc_now()
    log_path = output_root / "launcher_logs" / f"{resolved['name']}.log"
    exit_code = run_logged_process(command, log_path)
    row = {
        "job_type": "training",
        "variant": resolved["name"],
        "phase": resolved["phase"],
        "gpu": gpu,
        "command": command,
        "log_path": str(log_path),
        "started_at": started,
        "finished_at": utc_now(),
        "exit_code": exit_code,
        "status": "completed" if exit_code == 0 else "failed",
        "metrics": None,
    }
    if exit_code == 0:
        try:
            row["metrics"] = best_evaluated_metrics(resolved, selection)
        except Exception as error:
            row["status"] = "failed"
            row["error"] = f"metric extraction failed: {error}"
    return row


def run_diagnostic_job(
    resolved: dict,
    gpu: int,
    manifest_path: Path,
    output_root: Path,
) -> dict:
    command = diagnostic_command(resolved, manifest_path, output_root)
    started = utc_now()
    log_path = output_root / "launcher_logs" / f"{resolved['name']}.log"
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    exit_code = run_logged_process(
        command,
        log_path,
        environment=environment,
    )
    row = {
        "job_type": "diagnostic",
        "diagnostic": resolved["name"],
        "diagnostic_type": resolved["type"],
        "gpu": gpu,
        "command": command,
        "log_path": str(log_path),
        "started_at": started,
        "finished_at": utc_now(),
        "exit_code": exit_code,
        "status": "completed" if exit_code == 0 else "failed",
        "result": None,
    }
    result_path = Path(resolved["output_path"])
    if exit_code == 0:
        if result_path.is_file():
            row["result"] = json.loads(result_path.read_text(encoding="utf-8"))
        else:
            row["status"] = "failed"
            row["error"] = f"diagnostic result is missing: {result_path}"
    return row


def summarize(rows: list[dict], manifest: dict) -> dict:
    completed = [
        row
        for row in rows
        if row.get("job_type") == "training" and row.get("metrics")
    ]
    selection = manifest["selection"]
    best = (
        max(
            completed,
            key=lambda row: metric_rank_key(row["metrics"], selection),
        )
        if completed
        else None
    )
    return {
        "schema_version": 2,
        "experiment_id": manifest["experiment_id"],
        "updated_at": utc_now(),
        "protocol": manifest["protocol"],
        "selection": canonical(selection),
        "selection_policy": selection["policy"],
        "best_variant": best["variant"] if best else None,
        "best_metrics": best["metrics"] if best else None,
        "training_runs": [
            row for row in rows if row.get("job_type") == "training"
        ],
        "diagnostics": [
            row for row in rows if row.get("job_type") == "diagnostic"
        ],
    }


def run_scheduled(
    plan: dict,
    manifest: dict,
    gpus: list[int],
    output_root: Path,
) -> dict:
    if output_root.exists():
        raise FileExistsError(f"fresh output root already exists: {output_root}")
    maximum = int(manifest["scheduling"]["max_idle_memory_mb"])
    snapshots = [require_idle_gpu(gpu, maximum) for gpu in gpus]
    output_root.mkdir(parents=True)
    atomic_json(output_root / "plan.json", {**plan, "gpu_snapshots": snapshots})
    diagnostic_order = [
        name for name in plan["job_order"] if name in plan["diagnostics"]
    ]
    training_order = [
        name for name in plan["job_order"] if name in plan["variants"]
    ]
    diagnostic_queues = {gpu: [] for gpu in gpus}
    training_queues = {gpu: [] for gpu in gpus}
    for index, name in enumerate(diagnostic_order):
        diagnostic_queues[gpus[index % len(gpus)]].append(name)
    for index, name in enumerate(training_order):
        training_queues[gpus[index % len(gpus)]].append(name)
    state = {
        "schema_version": 2,
        "experiment_id": manifest["experiment_id"],
        "status": "running",
        "stage": "diagnostics",
        "started_at": utc_now(),
        "scheduler_pid": os.getpid(),
        "queues": {
            "diagnostics": diagnostic_queues,
            "training": training_queues,
        },
        "jobs": {
            name: {
                "status": "queued",
                "gpu": gpu,
                "job_type": (
                    "training" if name in plan["variants"] else "diagnostic"
                ),
            }
            for gpu, names in {
                gpu: diagnostic_queues[gpu] + training_queues[gpu]
                for gpu in gpus
            }.items()
            for name in names
        },
    }
    state_path = output_root / "scheduler" / "state.json"
    atomic_json(state_path, state)
    lock = threading.Lock()
    completed_rows = {}

    def worker(gpu: int, names: list[str]) -> list[dict]:
        rows = []
        for name in names:
            with lock:
                state["jobs"][name]["status"] = "running"
                state["jobs"][name]["started_at"] = utc_now()
                atomic_json(state_path, state)
            try:
                if name in plan["variants"]:
                    row = run_training(
                        plan["variants"][name],
                        gpu,
                        manifest["selection"],
                        output_root,
                    )
                else:
                    row = run_diagnostic_job(
                        plan["diagnostics"][name],
                        gpu,
                        Path(manifest["_manifest_path"]),
                        output_root,
                    )
            except Exception as error:
                row = {
                    "job_type": state["jobs"][name]["job_type"],
                    "variant": name if name in plan["variants"] else None,
                    "diagnostic": name if name in plan["diagnostics"] else None,
                    "gpu": gpu,
                    "started_at": state["jobs"][name]["started_at"],
                    "finished_at": utc_now(),
                    "exit_code": -1,
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                }
            rows.append(row)
            with lock:
                completed_rows[name] = row
                state["jobs"][name].update(
                    status=row["status"],
                    exit_code=row["exit_code"],
                    finished_at=row["finished_at"],
                )
                if row.get("error"):
                    state["jobs"][name]["error"] = row["error"]
                atomic_json(state_path, state)
                atomic_json(
                    output_root / "results.partial.json",
                    summarize(list(completed_rows.values()), manifest),
                )
        return rows

    def execute(queues: dict[int, list[str]]) -> list[dict]:
        with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
            futures = [
                executor.submit(worker, gpu, names)
                for gpu, names in queues.items()
                if names
            ]
            return [row for future in futures for row in future.result()]

    diagnostic_rows = execute(diagnostic_queues)
    if not all(row["status"] == "completed" for row in diagnostic_rows):
        with lock:
            for name in training_order:
                state["jobs"][name]["status"] = "blocked_by_diagnostic"
            state["status"] = "diagnostic_failed"
            state["stage"] = "finished"
            state["finished_at"] = utc_now()
            atomic_json(state_path, state)
        result = summarize(diagnostic_rows, manifest)
        atomic_json(output_root / "results.json", result)
        return result

    state["stage"] = "training"
    atomic_json(state_path, state)
    training_rows = execute(training_queues)
    rows = diagnostic_rows + training_rows
    result = summarize(rows, manifest)
    state["status"] = (
        "completed" if all(row["status"] == "completed" for row in rows) else "failed"
    )
    state["stage"] = "finished"
    state["finished_at"] = utc_now()
    atomic_json(state_path, state)
    atomic_json(output_root / "results.json", result)
    return result


def collect_existing(plan: dict, manifest: dict, output_root: Path) -> tuple[dict, Path]:
    state_path = output_root / "scheduler" / "state.json"
    if not state_path.is_file():
        raise FileNotFoundError(f"scheduler state is missing: {state_path}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    jobs = state.get("jobs") or {}
    rows = []

    for name, resolved in plan["variants"].items():
        job = jobs.get(name, {})
        row = {
            "job_type": "training",
            "variant": name,
            "phase": resolved["phase"],
            "gpu": job.get("gpu"),
            "log_path": str(output_root / "launcher_logs" / f"{name}.log"),
            "started_at": job.get("started_at"),
            "finished_at": job.get("finished_at"),
            "exit_code": job.get("exit_code"),
            "status": job.get("status", "unknown"),
            "metrics": None,
        }
        try:
            row["metrics"] = best_evaluated_metrics(
                resolved,
                manifest["selection"],
            )
        except Exception as error:
            if row["status"] == "completed":
                row["status"] = "failed"
                row["error"] = f"metric extraction failed: {error}"
        rows.append(row)

    for name, resolved in plan["diagnostics"].items():
        job = jobs.get(name, {})
        result_path = Path(resolved["output_path"])
        rows.append(
            {
                "job_type": "diagnostic",
                "diagnostic": name,
                "diagnostic_type": resolved["type"],
                "gpu": job.get("gpu"),
                "log_path": str(output_root / "launcher_logs" / f"{name}.log"),
                "started_at": job.get("started_at"),
                "finished_at": job.get("finished_at"),
                "exit_code": job.get("exit_code"),
                "status": job.get("status", "unknown"),
                "result": (
                    json.loads(result_path.read_text(encoding="utf-8"))
                    if result_path.is_file()
                    else None
                ),
            }
        )

    result = summarize(rows, manifest)
    result["source"] = "recollected-from-evaluation-events"
    finished = state.get("stage") == "finished"
    result_path = output_root / (
        "results.json" if finished else "results.best_epoch.partial.json"
    )
    atomic_json(result_path, result)

    plan_path = output_root / "plan.json"
    if plan_path.is_file():
        stored_plan = json.loads(plan_path.read_text(encoding="utf-8"))
        stored_plan["selection"] = canonical(manifest["selection"])
        stored_plan["selection_policy"] = manifest["selection"]["policy"]
        stored_plan.pop("reported_epoch", None)
        atomic_json(plan_path, stored_plan)
    return result, result_path


def autoresearch_trial(plan: dict, manifest: dict, variant: str) -> dict:
    gpu_text = os.environ.get("AR2_GPU_ID")
    output_text = os.environ.get("AR2_OUTPUT_DIR")
    results_text = os.environ.get("AR2_RESULTS_DIR")
    if gpu_text is None or output_text is None or results_text is None:
        raise RuntimeError(
            "autoresearch-trial requires AR2_GPU_ID, AR2_OUTPUT_DIR, and AR2_RESULTS_DIR"
        )
    gpu = int(gpu_text)
    resolved = plan["variants"][variant]
    resolved["output_root"] = str(Path(output_text) / variant)
    resolved["events_path"] = str(Path(output_text) / "events" / f"{variant}.jsonl")
    resolved["runtime_overrides"].update(
        output_root=resolved["output_root"],
        metric_events_path=resolved["events_path"],
    )
    row = run_training(
        resolved,
        gpu,
        manifest["selection"],
        Path(output_text),
    )
    if row["status"] != "completed":
        raise RuntimeError(f"training failed with exit code {row['exit_code']}")
    metrics = row["metrics"]
    payload = {
        "primary_metric": metrics["Rank-1"],
        "metrics": {
            "mAP": metrics["mAP"],
            "mINP": metrics["mINP"],
            "epoch": float(metrics["epoch"]),
            "gallery_trials": float(metrics["gallery_trials"]),
        },
    }
    atomic_json(Path(results_text) / "metrics.json", payload)
    return payload


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=(
            "list",
            "plan",
            "run",
            "status",
            "collect",
            "diagnostic",
            "autoresearch-trial",
        ),
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--phase", default=None)
    parser.add_argument("--variants", default=None)
    parser.add_argument("--variant", default=None)
    parser.add_argument("--diagnostic", default=None)
    parser.add_argument("--gpus", default=None)
    args = parser.parse_args(argv)

    manifest = load_manifest(args.manifest.resolve())
    output_root = (
        args.output_root
        or (Path(os.environ["AR2_OUTPUT_DIR"]) if "AR2_OUTPUT_DIR" in os.environ else None)
        or DEFAULT_OUTPUT_ROOT
    ).resolve()
    names = parse_csv(args.variants) if args.variants else None
    if args.variant:
        names = [args.variant]
    diagnostic_names = None
    if args.action == "diagnostic":
        if not args.diagnostic:
            raise ValueError("diagnostic action requires --diagnostic")
        if args.diagnostic not in manifest["diagnostics"]:
            raise ValueError(f"unknown diagnostic: {args.diagnostic}")
        diagnostic_names = [args.diagnostic]
        names = [manifest["diagnostics"][args.diagnostic]["variant"]]
    elif args.action == "autoresearch-trial":
        diagnostic_names = []
    elif args.phase is not None or names is not None:
        diagnostic_names = []
    selected = select_variants(manifest, names=names, phase=args.phase)
    plan = build_plan(
        manifest,
        selected,
        output_root,
        diagnostic_names=diagnostic_names,
    )

    if args.action == "list":
        training_rows = [
            {
                "job": name,
                "job_type": "training",
                "phase": item["phase"],
                "purpose": item["purpose"],
            }
            for name, item in manifest["variants"].items()
        ]
        diagnostic_rows = [
            {
                "job": name,
                "job_type": "diagnostic",
                "diagnostic_type": item["type"],
                "purpose": item["purpose"],
            }
            for name, item in manifest["diagnostics"].items()
        ]
        print(
            json.dumps(
                training_rows + diagnostic_rows,
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.action == "plan":
        print(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.action == "status":
        results = output_root / "results.json"
        state = output_root / "scheduler" / "state.json"
        selected_path = results if results.is_file() else state
        print(selected_path.read_text(encoding="utf-8"))
        return 0
    if args.action == "collect":
        result, result_path = collect_existing(plan, manifest, output_root)
        print(
            json.dumps(
                {"result_path": str(result_path), **result},
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.action == "diagnostic":
        resolved = plan["diagnostics"][args.diagnostic]
        result = run_c3_recipe_diagnostic(
            resolved,
            resolved["resolved_variant"],
            PROJECT_ROOT,
            output_root,
        )
        result_path = Path(resolved["output_path"])
        atomic_json(result_path, result)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.action == "autoresearch-trial":
        if len(selected) != 1:
            raise ValueError("autoresearch-trial requires exactly one --variant")
        print(
            json.dumps(
                autoresearch_trial(plan, manifest, selected[0]),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    gpu_values = args.gpus or ",".join(
        str(value) for value in manifest["scheduling"]["default_gpus"]
    )
    gpus = [int(value) for value in parse_csv(gpu_values)]
    if not gpus or len(gpus) != len(set(gpus)):
        raise ValueError("--gpus must contain one or more distinct GPU indices")
    if set(gpus) != {2, 3}:
        raise ValueError("this launch is pinned to physical GPUs 2 and 3")
    print(
        json.dumps(
            run_scheduled(plan, manifest, gpus, output_root),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
