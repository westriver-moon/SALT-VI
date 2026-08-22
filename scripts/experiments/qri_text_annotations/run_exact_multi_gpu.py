#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any


REPOSITORY = Path(__file__).resolve().parents[3]
PLUGIN_ROOT = REPOSITORY / "plugins" / "qwen_imagination"
for candidate in (REPOSITORY, REPOSITORY / "src", PLUGIN_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from qwen_imagination.text_annotation.cli import qwen_server_command  # noqa: E402
from qwen_imagination.text_annotation.config import (  # noqa: E402
    load_text_annotation_config,
)
from qwen_imagination.text_annotation.manifest import atomic_json  # noqa: E402


def parse_gpu_ids(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("gpu ids must be a non-empty unique list")
    return values


def gpu_memory_used() -> dict[int, int]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.used",
        "--format=csv,noheader,nounits",
    ]
    document = subprocess.check_output(command, text=True)
    result = {}
    for line in document.splitlines():
        if not line.strip():
            continue
        index, used = line.split(",", 1)
        result[int(index.strip())] = int(used.strip())
    return result


def wait_for_health(url: str, process: subprocess.Popen, timeout: float) -> None:
    deadline = time.monotonic() + float(timeout)
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Qwen server exited with code {process.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                if int(response.status) == 200:
                    return
        except Exception as error:  # startup is expected to refuse connections
            last_error = error
        time.sleep(1.0)
    raise TimeoutError(f"Qwen server health timeout at {url}: {last_error}")


def stop_process(process: subprocess.Popen, timeout: float = 20.0) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run exact per-image Qwen annotation with one replica per GPU"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--gpu-ids", type=parse_gpu_ids, default=parse_gpu_ids("0,1,2"))
    parser.add_argument("--base-port", type=int, default=18080)
    parser.add_argument("--split", choices=("train", "evaluation", "all"), default="train")
    parser.add_argument("--limit-per-worker", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--skip-busy-check", action="store_true")
    parser.add_argument("--max-used-memory-mib", type=int, default=1024)
    parser.add_argument("--server-start-timeout", type=float, default=240.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def check_runtime() -> None:
    missing = [
        name
        for name in ("torch", "ultralytics")
        if importlib.util.find_spec(name) is None
    ]
    if missing:
        raise RuntimeError(
            f"Python {sys.executable} omits exact-mode dependencies {missing}; "
            "run this launcher with the qri-v1 Python environment"
        )
    try:
        import torch
    except (ImportError, OSError) as error:
        raise RuntimeError(f"PyTorch CUDA runtime cannot load: {error}") from error
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch reports that CUDA is unavailable")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    check_runtime()
    config_path = args.config.expanduser().resolve()
    config = load_text_annotation_config(config_path)
    if config.strategy != "exact":
        raise ValueError("multi-GPU exact runner requires strategy=exact")
    if args.limit_per_worker is not None and args.limit_per_worker < 1:
        raise ValueError("limit-per-worker must be positive")
    if not 1 <= args.base_port <= 65535 - len(args.gpu_ids):
        raise ValueError("base-port is outside the valid range")

    used = gpu_memory_used()
    unknown = [gpu for gpu in args.gpu_ids if gpu not in used]
    if unknown:
        raise ValueError(f"unknown GPU ids: {unknown}")
    busy = {
        gpu: used[gpu]
        for gpu in args.gpu_ids
        if used[gpu] > int(args.max_used_memory_mib)
    }
    if busy and not args.skip_busy_check:
        raise RuntimeError(
            f"refusing to use busy GPUs {busy}; wait, choose other ids, or pass "
            "--skip-busy-check explicitly"
        )

    run_id = time.strftime("%Y%m%d-%H%M%S")
    log_root = config.output_root / "logs" / "exact_multi_gpu" / run_id
    log_root.mkdir(parents=True, exist_ok=True)
    base_environment = dict(os.environ)
    python_path = [str(REPOSITORY), str(REPOSITORY / "src"), str(PLUGIN_ROOT)]
    if base_environment.get("PYTHONPATH"):
        python_path.append(base_environment["PYTHONPATH"])
    base_environment["PYTHONPATH"] = os.pathsep.join(python_path)

    plan: dict[str, Any] = {
        "run_id": run_id,
        "strategy": config.strategy,
        "split": args.split,
        "gpu_ids": args.gpu_ids,
        "num_shards": len(args.gpu_ids),
        "gpu_memory_used_mib": {str(key): value for key, value in used.items()},
        "output_root": str(config.output_root),
        "config": str(config_path),
        "workers": [],
    }
    for shard_index, gpu in enumerate(args.gpu_ids):
        port = args.base_port + shard_index
        worker = {
            "gpu": gpu,
            "port": port,
            "shard_index": shard_index,
            "endpoint": f"http://127.0.0.1:{port}/v1/chat/completions",
            "health_endpoint": f"http://127.0.0.1:{port}/health",
        }
        plan["workers"].append(worker)
    atomic_json(log_root / "plan.json", plan)
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0

    servers: list[subprocess.Popen] = []
    server_logs = []
    workers: list[subprocess.Popen] = []
    worker_logs = []
    interrupted = False

    def handle_signal(signum, _frame):
        nonlocal interrupted
        interrupted = True
        for process in workers:
            stop_process(process, timeout=5)
        for process in servers:
            stop_process(process, timeout=5)
        raise KeyboardInterrupt(f"received signal {signum}")

    previous_sigterm = signal.signal(signal.SIGTERM, handle_signal)
    previous_sigint = signal.signal(signal.SIGINT, handle_signal)
    started = time.monotonic()
    try:
        for item in plan["workers"]:
            replica_config = copy.deepcopy(config)
            replica_config.qwen["port"] = int(item["port"])
            command = qwen_server_command(replica_config)
            environment = dict(base_environment)
            environment["CUDA_VISIBLE_DEVICES"] = str(item["gpu"])
            log_handle = (log_root / f"qwen-gpu{item['gpu']}.log").open(
                "w", encoding="utf-8"
            )
            server_logs.append(log_handle)
            process = subprocess.Popen(
                command,
                cwd=REPOSITORY,
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            servers.append(process)
        for item, process in zip(plan["workers"], servers):
            wait_for_health(
                item["health_endpoint"], process, args.server_start_timeout
            )

        for item in plan["workers"]:
            command = [
                sys.executable,
                "-m",
                "qwen_imagination.text_annotation.cli",
                "--config",
                str(config_path),
                "run",
                "--split",
                args.split,
                "--num-shards",
                str(len(args.gpu_ids)),
                "--shard-index",
                str(item["shard_index"]),
                "--device",
                "cuda:0",
                "--qwen-endpoint",
                item["endpoint"],
            ]
            if args.limit_per_worker is not None:
                command.extend(["--limit", str(args.limit_per_worker)])
            if args.overwrite:
                command.append("--overwrite")
            if args.fail_fast:
                command.append("--fail-fast")
            environment = dict(base_environment)
            environment["CUDA_VISIBLE_DEVICES"] = str(item["gpu"])
            log_handle = (log_root / f"worker-gpu{item['gpu']}.log").open(
                "w", encoding="utf-8"
            )
            worker_logs.append(log_handle)
            workers.append(
                subprocess.Popen(
                    command,
                    cwd=REPOSITORY,
                    env=environment,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            )

        return_codes = [process.wait() for process in workers]
        summary = {
            **plan,
            "elapsed_seconds": time.monotonic() - started,
            "return_codes": return_codes,
            "complete": all(code == 0 for code in return_codes),
            "interrupted": interrupted,
        }
        atomic_json(log_root / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if summary["complete"] else 1
    finally:
        for process in workers:
            stop_process(process)
        for process in servers:
            stop_process(process)
        for handle in worker_logs + server_logs:
            handle.close()
        signal.signal(signal.SIGTERM, previous_sigterm)
        signal.signal(signal.SIGINT, previous_sigint)


if __name__ == "__main__":
    raise SystemExit(main())
