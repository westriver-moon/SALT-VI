"""Command line interface for the unified PASD protocol."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .adapters import build_records
from .config import PluginConfig
from .generation import generate_records, load_build, load_protocol_records
from .scheduler import run_scheduler
from .validation import validate_dataset


def _records_argument(config: PluginConfig, value: str | None) -> Path:
    return Path(value).expanduser().resolve() if value else config.records_path


def _build_records(args: argparse.Namespace) -> int:
    config = PluginConfig.from_yaml(args.config)
    records = build_records(config, args.records)
    print(json.dumps({"dataset": config.dataset, "records": len(records), "records_path": str(_records_argument(config, args.records))}, ensure_ascii=False, indent=2))
    return 0


def _generate(args: argparse.Namespace) -> int:
    config = PluginConfig.from_yaml(args.config)
    result = run_scheduler(
        args.config,
        _records_argument(config, args.records),
        workers=args.workers,
        poll_seconds=args.poll_seconds,
        worker_max_sources=args.worker_max_sources,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["complete"] else 1


def _validate(args: argparse.Namespace) -> int:
    config = PluginConfig.from_yaml(args.config)
    records = load_protocol_records(_records_argument(config, args.records))
    load_build(config)
    result = validate_dataset(config, records)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["complete"] else 1


def _worker(args: argparse.Namespace) -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    if visible != str(args.physical_gpu):
        raise RuntimeError("worker CUDA_VISIBLE_DEVICES does not match its physical GPU")
    config = PluginConfig.from_yaml(args.config)
    load_build(config)
    records = load_protocol_records(_records_argument(config, args.records))
    result = generate_records(
        config,
        records[args.shard_index :: args.shard_count],
        physical_gpu=args.physical_gpu,
        max_sources=args.max_sources,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Unified PASD generator for SYSU-MM01, RegDB, and LLCM")
    subcommands = parser.add_subparsers(dest="command", required=True)
    for name, handler in (("build-records", _build_records), ("generate", _generate), ("validate", _validate)):
        command = subcommands.add_parser(name)
        command.add_argument("--config", required=True)
        command.add_argument("--records")
        command.set_defaults(handler=handler)
    generate = subcommands.choices["generate"]
    generate.add_argument("--workers", type=int, choices=(1, 2, 3), default=1)
    generate.add_argument("--poll-seconds", type=int, default=60)
    generate.add_argument("--worker-max-sources", type=int)
    worker = subcommands.add_parser("_worker", help=argparse.SUPPRESS)
    worker.add_argument("--config", required=True)
    worker.add_argument("--records")
    worker.add_argument("--physical-gpu", required=True, type=int, choices=(1, 2, 3))
    worker.add_argument("--shard-index", required=True, type=int)
    worker.add_argument("--shard-count", required=True, type=int)
    worker.add_argument("--max-sources", type=int)
    worker.set_defaults(handler=_worker)
    args = parser.parse_args(argv)
    return args.handler(args)
