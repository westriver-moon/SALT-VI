"""Shared, version-aware Stage-A launcher for preregistered QRI registries."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def load_registry(path: str | Path) -> dict:
    path = Path(path).expanduser().resolve()
    document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    required = {"generation", "variants", "environment", "base_final_epoch"}
    missing = sorted(required.difference(document))
    if missing:
        raise ValueError(f"QRI registry omits required fields: {missing}")
    return document


def validate_base_completion(path: str | Path, final_epoch: int) -> dict:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"C3-b96 prerequisite event stream is missing: {path}")
    events = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    completed = [
        event
        for event in events
        if event.get("event") == "train_epoch"
        and int(event.get("epoch", -1)) == final_epoch
    ]
    if not completed:
        raise ValueError(
            f"C3-b96 has not recorded train_epoch={final_epoch}; QRI formal launch remains gated"
        )
    return {"events": str(path), "final_epoch": final_epoch, "event": completed[-1]}


def validate_manifest(data_root: Path, relative: str, expected_plugin: str) -> dict:
    path = (data_root / relative).resolve()
    summary_path = path.with_suffix(".json")
    if not path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(f"QRI manifest is incomplete: {path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not summary.get("complete") or int(summary.get("views_per_source", -1)) != 0:
        raise ValueError(
            f"QRI manifest does not satisfy the dynamic complete contract: {summary_path}"
        )
    if summary.get("plugin") != expected_plugin:
        raise ValueError(
            f"QRI manifest plugin {summary.get('plugin')} != registry {expected_plugin}"
        )
    return {"path": str(path), "summary": summary}


def run_registry(registry_path: str | Path, argv=None) -> int:
    manifest = load_registry(registry_path)
    parser = argparse.ArgumentParser(
        description=f"Stage-A launcher for {manifest['generation']['plugin']}"
    )
    parser.add_argument("action", choices=("list", "preflight", "train"))
    parser.add_argument("--variant", choices=tuple(manifest["variants"]))
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--experiment-root", type=Path)
    parser.add_argument("--base-events", type=Path)
    parser.add_argument("--gpu", type=int, choices=(0, 1, 2, 3), default=3)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)

    if args.action == "list":
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0
    from salt_vi.config.validation import validate_runtime_config
    from salt_vi.utils.utils import load_train_configs

    if not args.variant or not args.data_root:
        parser.error(f"{args.action} requires --variant and --data-root")
    if args.action == "train" and (not args.experiment_root or not args.base_events):
        parser.error("train requires --experiment-root and --base-events")

    data_root = args.data_root.expanduser().resolve()
    variant = manifest["variants"][args.variant]
    view = validate_manifest(
        data_root,
        variant["manifest"],
        str(manifest["generation"]["plugin"]),
    )
    base = None
    if args.base_events:
        base = validate_base_completion(
            args.base_events, int(manifest["base_final_epoch"])
        )

    experiment_root = (
        args.experiment_root.expanduser().resolve()
        if args.experiment_root
        else Path("/tmp/qri-preflight")
    )
    names = manifest["environment"]
    env = dict(os.environ)
    env.update(
        {
            str(names["data_root"]): str(data_root),
            str(names["view_manifest"]): view["path"],
            str(names["experiment_root"]): str(experiment_root),
            "SALT_SAFE_TRICKS_OUTPUT_ROOT": str(experiment_root),
        }
    )
    config_path = PROJECT_ROOT / variant["config"]
    previous = dict(os.environ)
    os.environ.update(env)
    try:
        config = load_train_configs(config_path)
        validate_runtime_config(config)
    finally:
        os.environ.clear()
        os.environ.update(previous)

    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/train.py"),
        "--config_select",
        str(config_path),
        "--set",
        f"CUDA_VISIBLE_DEVICES='{args.gpu}'",
        "--set",
        "gpu_id='0'",
    ]
    plan = {
        "plugin": manifest["generation"]["plugin"],
        "variant": args.variant,
        "config": str(config_path),
        "experiment_id": config.metric_experiment_id,
        "manifest": view,
        "base_completion": base,
        "gpu": args.gpu,
        "command": command,
        "automatic_promotion": False,
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    if args.execute:
        if args.action != "train":
            parser.error("--execute is valid only for train")
        return subprocess.run(command, cwd=PROJECT_ROOT, env=env).returncode
    return 0
