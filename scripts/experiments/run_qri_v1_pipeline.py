#!/usr/bin/env python3
"""List, validate, or launch preregistered C3-b96 QRI-v1 Stage-A trials."""

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

from salt_vi.config.validation import validate_runtime_config  # noqa: E402
from salt_vi.utils.utils import load_train_configs  # noqa: E402


REGISTRY_PATH = PROJECT_ROOT / "configs/pipelines/sysu_qri_v1.yaml"


def registry() -> dict:
    return yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))


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
        if event.get("event") == "train_epoch" and int(event.get("epoch", -1)) == final_epoch
    ]
    if not completed:
        raise ValueError(
            f"C3-b96 has not recorded train_epoch={final_epoch}; QRI formal launch remains gated"
        )
    return {"events": str(path), "final_epoch": final_epoch, "event": completed[-1]}


def validate_manifest(data_root: Path, relative: str) -> dict:
    path = (data_root / relative).resolve()
    summary_path = path.with_suffix(".json")
    if not path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(f"QRI manifest is incomplete: {path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not summary.get("complete") or int(summary.get("views_per_source", -1)) != 0:
        raise ValueError(f"QRI manifest does not satisfy the dynamic complete contract: {summary_path}")
    return {"path": str(path), "summary": summary}


def main(argv=None) -> int:
    manifest = registry()
    parser = argparse.ArgumentParser(description=__doc__)
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
    if not args.variant or not args.data_root:
        parser.error(f"{args.action} requires --variant and --data-root")
    if args.action == "train" and (not args.experiment_root or not args.base_events):
        parser.error("train requires --experiment-root and --base-events")

    data_root = args.data_root.expanduser().resolve()
    variant = manifest["variants"][args.variant]
    view = validate_manifest(data_root, variant["manifest"])
    base = None
    if args.base_events:
        base = validate_base_completion(args.base_events, int(manifest["base_final_epoch"]))

    experiment_root = (
        args.experiment_root.expanduser().resolve()
        if args.experiment_root
        else Path("/tmp/qri-preflight")
    )
    env = dict(os.environ)
    env.update(
        {
            "SALT_QRI_DATA_ROOT": str(data_root),
            "SALT_QRI_VIEW_MANIFEST": view["path"],
            "SALT_QRI_EXPERIMENT_ROOT": str(experiment_root),
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


if __name__ == "__main__":
    raise SystemExit(main())
