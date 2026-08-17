#!/usr/bin/env python3
"""List or launch preregistered SYSU safe-trick experiments."""

from __future__ import annotations

import argparse
import hashlib
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

from salt_vi.config.validation import validate_runtime_config
from salt_vi.utils.utils import load_train_configs


MANIFEST_PATH = PROJECT_ROOT / "configs/pipelines/sysu_safe_tricks.yaml"
DEFAULT_OUTPUT_ROOT = Path("/home/lab929/ybj/experiments/SALT-VI-safe-tricks")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest():
    return yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))


def training_command(config_path):
    return [
        sys.executable,
        str(PROJECT_ROOT / "scripts/train.py"),
        "--config_select",
        str(config_path),
    ]


def main(argv=None):
    manifest = load_manifest()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("list", "stage-a", "stage-b"))
    parser.add_argument("--variant", choices=tuple(manifest["stage_b"]["variants"]))
    parser.add_argument("--stage-a-checkpoint", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)

    if args.action == "list":
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0

    env = dict(os.environ)
    env["SALT_SAFE_TRICKS_OUTPUT_ROOT"] = str(args.output_root.resolve())
    if args.action == "stage-a":
        config_path = PROJECT_ROOT / manifest["stage_a"]
    else:
        if not args.variant or not args.stage_a_checkpoint:
            parser.error("stage-b requires --variant and --stage-a-checkpoint")
        checkpoint = args.stage_a_checkpoint.resolve()
        if not checkpoint.is_file():
            parser.error(f"Stage-A checkpoint does not exist: {checkpoint}")
        env["SALT_STAGE_A_CHECKPOINT"] = str(checkpoint)
        env["SALT_STAGE_A_SHA256"] = sha256_file(checkpoint)
        config_path = PROJECT_ROOT / manifest["stage_b"]["variants"][args.variant]

    previous = dict(os.environ)
    os.environ.update(env)
    try:
        config = load_train_configs(str(config_path))
        validate_runtime_config(config)
    finally:
        os.environ.clear()
        os.environ.update(previous)

    command = training_command(config_path)
    plan = {
        "config": str(config_path),
        "experiment_id": config.metric_experiment_id,
        "output_root": env["SALT_SAFE_TRICKS_OUTPUT_ROOT"],
        "command": command,
    }
    if args.action == "stage-b":
        plan["stage_a_checkpoint"] = env["SALT_STAGE_A_CHECKPOINT"]
        plan["stage_a_sha256"] = env["SALT_STAGE_A_SHA256"]
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    if args.execute:
        return subprocess.run(command, cwd=PROJECT_ROOT, env=env).returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
