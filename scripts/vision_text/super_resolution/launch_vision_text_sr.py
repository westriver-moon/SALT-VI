#!/usr/bin/env python3
"""Dynamically launch provenance-validated PMT-MBPatch SR experiments."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from salt_vi.baselines.vision_text.config import load_config
from preflight_vision_text_sr import (
    assert_clean_source,
    atomic_json,
    provenance,
    sha256_file,
    validate_config_contract,
)


CONFIG_NAMES = (
    "sr_a0_original_256.yaml",
    "sr_a1_bicubic_x2.yaml",
    "sr_a2_swinir_rgb_x2.yaml",
    "sr_a3_swinir_both_x2.yaml",
)
CONFIG_DIR = PROJECT_ROOT / "configs/vision_text/super_resolution"


def read_idle_gpus(max_memory_mib, max_utilization):
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    rows = subprocess.check_output(command, text=True).splitlines()
    idle = []
    for row in rows:
        index, memory, utilization = [int(part.strip()) for part in row.split(",")]
        if memory <= max_memory_mib and utilization <= max_utilization:
            idle.append(index)
    return idle


def load_validated_job(config_path, preflight_dir):
    config = load_config(config_path)
    validate_config_contract(config)
    if any(key in config for key in ("text", "language", "caption")):
        raise RuntimeError(f"{config.experiment.id}: text configuration is forbidden")
    expected = provenance(config_path, config)
    selected_path = preflight_dir / f"{config.experiment.id}.json"
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    if not selected.get("valid") or selected.get("provenance") != expected:
        raise RuntimeError(f"{config.experiment.id}: missing or stale provenance-bound preflight")
    attempt = selected.get("selected_attempt", {})
    if not attempt.get("valid") or int(attempt.get("steps", 0)) != 20:
        raise RuntimeError(f"{config.experiment.id}: preflight did not complete 20 finite steps")
    if attempt.get("metrics_after_20_steps") is None:
        raise RuntimeError(f"{config.experiment.id}: preflight evaluation is missing")
    output = Path(config.output.root).resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Refusing to reuse non-empty formal output: {output}")
    return {
        "config": config,
        "config_path": config_path.resolve(),
        "output": output,
        "preflight": selected,
        "preflight_path": selected_path.resolve(),
    }


def launch(job, physical_gpu, commit):
    config = job["config"]
    output = job["output"]
    output.mkdir(parents=True, exist_ok=False)
    chunk_size = int(job["preflight"]["selected_chunk_size"])
    test_batch_size = int(job["preflight"]["selected_test_batch_size"])
    command = [
        sys.executable,
        "-m",
        "salt_vi.baselines.vision_text.train",
        "--config",
        str(job["config_path"]),
        "--device",
        "cuda:0",
        "--output",
        str(output),
        "--override",
        f"model.backbone_chunk_size={chunk_size}",
        "--override",
        f"test.batch_size={test_batch_size}",
    ]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
    log_path = output / "train.stdout.log"
    log_handle = log_path.open("w", encoding="utf-8", buffering=1)
    process = subprocess.Popen(
        command,
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    manifest = {
        "schema_version": 1,
        "experiment_id": config.experiment.id,
        "recipe": "pure-visual PMT-MBPatch Stage-A",
        "contains_text": False,
        "git_commit_sha": commit,
        "physical_gpu_index": physical_gpu,
        "pid": process.pid,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(job["config_path"]),
        "config_sha256": sha256_file(job["config_path"]),
        "preflight_path": str(job["preflight_path"]),
        "preflight_sha256": sha256_file(job["preflight_path"]),
        "selected_chunk_size": chunk_size,
        "selected_test_batch_size": test_batch_size,
        "command": command,
        "log": str(log_path),
    }
    atomic_json(output / "launch_manifest.json", manifest)
    return {"process": process, "log_handle": log_handle, "manifest": manifest}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preflight-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/super_resolution/preflight",
    )
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--max-idle-memory-mib", type=int, default=1024)
    parser.add_argument("--max-idle-utilization", type=int, default=10)
    args = parser.parse_args()

    commit = assert_clean_source("origin/main")
    pending = [
        load_validated_job(CONFIG_DIR / name, args.preflight_dir.resolve())
        for name in CONFIG_NAMES
    ]
    active = {}
    failures = []
    while pending or active:
        for experiment_id, running in list(active.items()):
            returncode = running["process"].poll()
            if returncode is None:
                continue
            running["log_handle"].close()
            del active[experiment_id]
            if returncode:
                failures.append({"experiment_id": experiment_id, "returncode": returncode})

        occupied = {entry["manifest"]["physical_gpu_index"] for entry in active.values()}
        for gpu in read_idle_gpus(args.max_idle_memory_mib, args.max_idle_utilization):
            if gpu in occupied or not pending:
                continue
            job = pending.pop(0)
            running = launch(job, gpu, commit)
            active[job["config"].experiment.id] = running
            occupied.add(gpu)
            print(
                f"launched {job['config'].experiment.id} on physical GPU {gpu} "
                f"pid={running['process'].pid}",
                flush=True,
            )
        if pending or active:
            time.sleep(max(5, args.poll_seconds))

    if failures:
        raise SystemExit(f"Formal SR jobs failed: {failures}")


if __name__ == "__main__":
    main()
