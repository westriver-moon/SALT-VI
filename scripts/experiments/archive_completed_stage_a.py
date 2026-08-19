#!/usr/bin/env python3
"""Move a completed Stage-A run into the archive and capture its provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_PATH = Path("configs/pipelines/sysu_safe_tricks.yaml")


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def git_file(repo, commit, path):
    return subprocess.check_output(
        ["git", "-C", str(repo), "show", f"{commit}:{path}"], text=True
    )


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def final_eval(path):
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    evaluations = [event for event in events if event["event_type"] == "eval_epoch"]
    return evaluations[-1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--launch-command", required=True)
    args = parser.parse_args(argv)

    run_root = args.run_root.resolve(strict=True)
    archive_root = args.archive_root.resolve()
    if archive_root.exists():
        parser.error(f"archive target already exists: {archive_root}")

    state = json.loads((run_root / "scheduler/state.json").read_text(encoding="utf-8"))
    if state["status"] != "completed" or state["failed"] or state["running"]:
        parser.error("run is not cleanly completed")
    variants = [item["variant"] for item in state["completed"]]

    pipeline = yaml.safe_load(git_file(PROJECT_ROOT, args.code_commit, PIPELINE_PATH))
    source_configs = {name: pipeline["stage_a"]["variants"][name] for name in variants}
    metrics = {
        name: final_eval(run_root / f"events/{name}.jsonl") for name in variants
    }

    archive_root.parent.mkdir(parents=True, exist_ok=True)
    run_root.rename(archive_root)
    provenance = archive_root / "provenance"
    source_dir = provenance / "source_configs"
    source_dir.mkdir(parents=True)
    for name, path in source_configs.items():
        (source_dir / f"{name}.yaml").write_text(
            git_file(PROJECT_ROOT, args.code_commit, path), encoding="utf-8"
        )
    (provenance / "pipeline.yaml").write_text(
        git_file(PROJECT_ROOT, args.code_commit, PIPELINE_PATH), encoding="utf-8"
    )
    (provenance / "scheduler.py").write_text(
        git_file(
            PROJECT_ROOT,
            args.code_commit,
            Path("scripts/experiments/run_stage_a_gpu_scheduler.py"),
        ),
        encoding="utf-8",
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(PROJECT_ROOT),
            "bundle",
            "create",
            str(provenance / f"salt-vi-{args.code_commit[:8]}.bundle"),
            "--all",
        ],
        check=True,
    )
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    (provenance / "environment.json").write_text(
        json.dumps(environment, indent=2), encoding="utf-8"
    )
    (provenance / "pip-freeze.txt").write_text(
        subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "original_run_root": str(run_root),
        "archive_root": str(archive_root),
        "code_commit": args.code_commit,
        "launch_command": args.launch_command,
        "scheduler_state": state,
        "source_configs": source_configs,
        "resolved_configs": {
            name: str(next(archive_root.glob(f"{name}_*/**/configs.yaml")).relative_to(archive_root))
            for name in variants
        },
        "final_evaluations": metrics,
    }
    (provenance / "archive_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    inventory = []
    for path in sorted(archive_root.rglob("*")):
        if path.is_file() and path.name != "inventory.sha256":
            inventory.append(f"{sha256_file(path)}  {path.relative_to(archive_root).as_posix()}")
    (provenance / "inventory.sha256").write_text(
        "\n".join(inventory) + "\n", encoding="utf-8"
    )
    print(json.dumps({"archive_root": str(archive_root), "files": len(inventory)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
