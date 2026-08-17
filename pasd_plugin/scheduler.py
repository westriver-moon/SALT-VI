"""GPU-aware worker scheduler for the unified PASD plugin."""

from __future__ import annotations

import csv
import io
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from .config import PluginConfig
from .generation import consolidate_manifest, load_protocol_records, prepare_build


@dataclass(frozen=True)
class GPUStatus:
    index: int
    free_memory_mib: int
    utilization_percent: int


def query_gpu_status() -> dict[int, GPUStatus]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    )
    statuses = {}
    for row in csv.reader(io.StringIO(result.stdout)):
        index, free_memory, utilization = (int(value.strip()) for value in row)
        statuses[index] = GPUStatus(index, free_memory, utilization)
    return statuses


def eligible(status: GPUStatus, config: PluginConfig) -> bool:
    return (
        status.index in config.gpu_allowlist
        and status.index != 0
        and status.free_memory_mib >= int(config.min_free_memory_gib * 1024)
        and status.utilization_percent <= config.max_gpu_utilization
    )


def run_scheduler(
    config_path: str | Path,
    records_path: str | Path,
    *,
    workers: int = 1,
    poll_seconds: int = 60,
    worker_max_sources: int | None = None,
) -> dict:
    if workers not in (1, 2, 3):
        raise ValueError("workers must be 1, 2, or 3")
    config_path = Path(config_path).expanduser().resolve()
    records_path = Path(records_path).expanduser().resolve()
    config = PluginConfig.from_yaml(config_path)
    records = load_protocol_records(records_path)
    prepare_build(config, records_path)
    selected: list[int] = []
    while len(selected) < workers:
        selected = [status.index for status in query_gpu_status().values() if eligible(status, config)][:workers]
        if len(selected) < workers:
            time.sleep(max(5, min(60, poll_seconds)))
    package_root = Path(__file__).resolve().parent.parent
    processes = []
    for shard_index, physical_gpu in enumerate(selected):
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
        environment["PYTHONPATH"] = str(package_root) + os.pathsep + environment.get("PYTHONPATH", "")
        command = [
            sys.executable,
            "-m",
            "pasd_plugin",
            "_worker",
            "--config",
            str(config_path),
            "--records",
            str(records_path),
            "--physical-gpu",
            str(physical_gpu),
            "--shard-index",
            str(shard_index),
            "--shard-count",
            str(len(selected)),
        ]
        if worker_max_sources is not None:
            command.extend(["--max-sources", str(worker_max_sources)])
        processes.append(subprocess.Popen(command, env=environment, start_new_session=True))
    failures = [process.wait() for process in processes]
    if any(code != 0 for code in failures):
        raise RuntimeError(f"PASD worker failures: {failures}")
    load_build(config)
    summary = consolidate_manifest(config, records)
    summary["physical_gpus"] = selected
    summary["worker_exit_codes"] = failures
    (config.output_root / "scheduler-result.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary
