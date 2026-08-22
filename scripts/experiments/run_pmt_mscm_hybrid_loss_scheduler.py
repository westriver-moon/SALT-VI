#!/usr/bin/env python3
"""Run the preregistered four-way PMT-MSCM hybrid-loss study on two GPUs."""

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
DEFAULT_MANIFEST = (
    PROJECT_ROOT / "configs/pipelines/sysu_pmt_mscm_hybrid_loss.yaml"
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def metric_summary(path: Path) -> dict:
    if not path.is_file():
        return {"path": str(path), "evaluation_count": 0}
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    evaluations = [
        row for row in rows if row.get("event_type") == "eval_epoch"
    ]
    if not evaluations:
        return {"path": str(path), "evaluation_count": 0}
    best = max(
        evaluations,
        key=lambda row: (
            float(row["metrics"]["Rank-1"]),
            float(row["metrics"]["mAP"]),
            float(row["metrics"]["mINP"]),
        ),
    )
    latest = max(evaluations, key=lambda row: int(row["epoch"]))

    def compact(row: dict) -> dict:
        return {
            "epoch": int(row["epoch"]),
            "metrics": {
                name: float(row["metrics"][name])
                for name in ("Rank-1", "mAP", "mINP")
            },
            "protocol": row.get("protocol"),
        }

    return {
        "path": str(path),
        "evaluation_count": len(evaluations),
        "best": compact(best),
        "latest": compact(latest),
    }


def load_manifest(path: Path) -> dict:
    document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if document.get("schema_version") != 2:
        raise ValueError("hybrid-loss scheduler requires schema_version: 2")
    variants = document.get("variants") or {}
    queues = (document.get("schedule") or {}).get("queues") or {}
    queued = [variant for queue in queues.values() for variant in queue]
    if len(queued) != len(set(queued)) or set(queued) != set(variants):
        raise ValueError("every hybrid-loss variant must appear in one GPU queue")
    return document


def public_running(running: dict[int, dict]) -> dict:
    return {
        str(gpu): {
            "variant": item["variant"],
            "pid": item["process"].pid,
            "log": item["log"],
            "started_at": item["started_at"],
        }
        for gpu, item in running.items()
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args(argv)

    manifest_path = args.manifest.expanduser().resolve()
    manifest = load_manifest(manifest_path)
    output_root = args.output_root.expanduser().resolve()
    scheduler_root = output_root / "scheduler"
    state_path = scheduler_root / "state.json"
    results_path = output_root / "results.json"
    if state_path.exists() or results_path.exists():
        raise FileExistsError(
            f"hybrid-loss scheduler output already contains state: {output_root}"
        )
    scheduler_root.mkdir(parents=True, exist_ok=True)

    variants = manifest["variants"]
    queues = {
        int(gpu): list(queue)
        for gpu, queue in manifest["schedule"]["queues"].items()
    }
    state = {
        "schema_version": 2,
        "experiment_id": manifest["experiment_id"],
        "status": "running",
        "started_at": now(),
        "manifest": str(manifest_path),
        "output_root": str(output_root),
        "queues": {str(gpu): list(queue) for gpu, queue in queues.items()},
        "running": {},
        "completed": [],
        "failed": [],
    }
    write_json(state_path, state)

    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    pythonpath_entries = [
        str(PROJECT_ROOT),
        str(PROJECT_ROOT / "src"),
        str(PROJECT_ROOT / "plugins/qwen_imagination"),
    ]
    inherited_pythonpath = environment.get("PYTHONPATH")
    if inherited_pythonpath:
        pythonpath_entries.append(inherited_pythonpath)
    environment["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    environment["SALT_MSCM_HYBRID_OUTPUT_ROOT"] = str(output_root)
    running: dict[int, dict] = {}

    while any(queues.values()) or running:
        for gpu in sorted(queues):
            if gpu in running or not queues[gpu]:
                continue
            variant = queues[gpu].pop(0)
            spec = variants[variant]
            config = (PROJECT_ROOT / spec["config"]).resolve()
            log_path = scheduler_root / f"{variant}.gpu{gpu}.log"
            log_handle = log_path.open("w", encoding="utf-8")
            command = [
                args.python,
                "-u",
                "-m",
                "salt_vi.entrypoints.train",
                "--config_select",
                str(config),
                "--set",
                f'CUDA_VISIBLE_DEVICES="{gpu}"',
                "--set",
                "gpu_id=0",
            ]
            process_environment = dict(environment)
            process_environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                env=process_environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            running[gpu] = {
                "variant": variant,
                "process": process,
                "log": str(log_path),
                "log_handle": log_handle,
                "started_at": now(),
                "command": command,
            }

        state["queues"] = {
            str(gpu): list(queue) for gpu, queue in queues.items()
        }
        state["running"] = public_running(running)
        write_json(state_path, state)
        if not running:
            break

        time.sleep(5)
        for gpu, item in list(running.items()):
            returncode = item["process"].poll()
            if returncode is None:
                continue
            item["log_handle"].close()
            variant = item["variant"]
            metric_path = output_root / variants[variant]["metric_events"]
            result = {
                "variant": variant,
                "gpu": gpu,
                "returncode": returncode,
                "started_at": item["started_at"],
                "finished_at": now(),
                "log": item["log"],
                "command": item["command"],
                "metrics": metric_summary(metric_path),
            }
            state["completed" if returncode == 0 else "failed"].append(result)
            del running[gpu]

    state["queues"] = {
        str(gpu): list(queue) for gpu, queue in queues.items()
    }
    state["running"] = {}
    state["finished_at"] = now()
    state["status"] = "failed" if state["failed"] else "completed"
    write_json(state_path, state)
    write_json(
        results_path,
        {
            "schema_version": 2,
            "experiment_id": manifest["experiment_id"],
            "status": state["status"],
            "completed": state["completed"],
            "failed": state["failed"],
            "selection": manifest["selection"],
        },
    )
    return 1 if state["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
