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

from .config import GenerationConfig
from .generate import (
    consolidate_manifest,
    invalidate_invalid_sources,
    prepare_build,
    source_is_generated,
)
from .tasks import GenerationTask, group_tasks_by_source, load_tasks


@dataclass(frozen=True)
class GPUStatus:
    index: int
    free_memory_mib: int
    utilization_percent: int


def query_gpu_status() -> dict[int, GPUStatus]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    statuses = {}
    for row in csv.reader(io.StringIO(result.stdout)):
        index, free_memory, utilization = (int(value.strip()) for value in row)
        statuses[index] = GPUStatus(index, free_memory, utilization)
    return statuses


def gpu_is_eligible(status: GPUStatus, config: GenerationConfig) -> bool:
    return (
        status.index in config.gpu_allowlist
        and status.index != 0
        and status.free_memory_mib >= int(config.min_free_memory_gib * 1024)
        and status.utilization_percent <= config.max_gpu_utilization
    )


def compute_pids(physical_gpu: int) -> set[int]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(physical_gpu),
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    pids = set()
    for value in result.stdout.splitlines():
        value = value.strip()
        if value.isdigit():
            pids.add(int(value))
    return pids


def has_foreign_compute_process(physical_gpu: int, own_pid: int) -> bool:
    return any(pid != own_pid for pid in compute_pids(physical_gpu))


def generated_source_count(
    groups: list[list[GenerationTask]],
    output_root: Path,
    config: GenerationConfig,
) -> int:
    return sum(source_is_generated(group, output_root, config) for group in groups)


def run_dynamic_scheduler(
    config_path: str | Path,
    records_path: str | Path,
    poll_seconds: int = 60,
    max_workers: int = 3,
    worker_max_sources: int | None = None,
) -> dict:
    if worker_max_sources is not None and worker_max_sources <= 0:
        raise ValueError("worker_max_sources must be positive")
    config_path = Path(config_path).expanduser().resolve()
    records_path = Path(records_path).expanduser().resolve()
    config = GenerationConfig.from_yaml(config_path)
    tasks = load_tasks(records_path)
    groups = group_tasks_by_source(tasks)
    if not groups:
        raise ValueError("records contain no generation tasks")
    for group in groups:
        indices = [task.view_index for task in group]
        expected = config.views_per_source or len(group)
        if len(group) != expected or indices != list(range(expected)):
            source_key = group[0].source_key or str(group[0].image)
            raise ValueError(f"invalid source view group {source_key}: {indices}")
    prepare_build(config, records_path, tasks)
    allowed = tuple(index for index in config.gpu_allowlist if index in (1, 2, 3))
    if 0 in allowed or not allowed or max_workers > 3:
        raise ValueError("scheduler may use only physical GPUs 1, 2, 3 and at most three workers")
    active: dict[int, subprocess.Popen] = {}
    logs = config.output_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    repair_cycles = 0
    while True:
        while generated_source_count(groups, config.output_root, config) < len(groups):
            for gpu, process in list(active.items()):
                return_code = process.poll()
                if return_code is not None:
                    active.pop(gpu)
                    if return_code not in (0, 75):
                        raise RuntimeError(
                            f"PASD worker on physical GPU {gpu} failed with {return_code}"
                        )

            statuses = query_gpu_status()
            for gpu in allowed:
                if len(active) >= max_workers or gpu in active:
                    continue
                status = statuses.get(gpu)
                if status is None or not gpu_is_eligible(status, config):
                    continue
                log_path = logs / f"worker-gpu{gpu}-{int(time.time())}.log"
                log_stream = log_path.open("a", encoding="utf-8")
                worker_environment = os.environ.copy()
                worker_environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
                command = [
                    sys.executable,
                    str(Path(__file__).resolve().parents[1] / "scripts" / "run_worker.py"),
                    "--config",
                    str(config_path),
                    "--records",
                    str(records_path),
                    "--physical-gpu",
                    str(gpu),
                    "--batch-size",
                    "1" if config.views_per_source == 1 else "0",
                ]
                if worker_max_sources is not None:
                    command.extend(["--max-sources", str(worker_max_sources)])
                process = subprocess.Popen(
                    command,
                    env=worker_environment,
                    stdout=log_stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                log_stream.close()
                active[gpu] = process

            time.sleep(min(60, max(5, poll_seconds)))

        for gpu, process in list(active.items()):
            return_code = process.wait()
            active.pop(gpu)
            if return_code not in (0, 75):
                raise RuntimeError(
                    f"PASD worker on physical GPU {gpu} failed with {return_code}"
                )

        summary = consolidate_manifest(
            config.output_root, tasks, config, records_path
        )
        if summary["validated_complete"]:
            summary["repair_cycles"] = repair_cycles
            (config.output_root / "scheduler-result.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            return summary
        invalidate_invalid_sources(config.output_root, summary)
        repair_cycles += 1
