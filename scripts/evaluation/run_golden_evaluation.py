"""Run read-only golden evaluations and write structured freeze evidence.

Each manifest entry selects one checkpoint, one resolved config, and the
exact protocol dimensions recorded in the resulting golden_evaluation.json.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_TRAIN = REPO_ROOT / "scripts" / "train.py"


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args(argv)


def _load_manifest(path):
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("evaluations"), list):
        raise ValueError("golden manifest must contain an 'evaluations' list")
    return payload


def _validate_entry(entry, index, output_root):
    entry_id = str(entry.get("id") or f"evaluation-{index}")
    config_path = Path(entry["config_path"])
    checkpoint_path = Path(entry["checkpoint_path"])
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    if not checkpoint_path.is_absolute():
        checkpoint_path = REPO_ROOT / checkpoint_path
    if not config_path.is_file():
        raise FileNotFoundError(f"{entry_id}: config missing: {config_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"{entry_id}: checkpoint missing: {checkpoint_path}")
    run_dir = output_root / entry_id
    return {
        "id": entry_id,
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint_path),
        "run_dir": str(run_dir),
        "overrides": entry.get("overrides") or {},
    }


def _build_command(entry):
    command = [
        sys.executable,
        str(SCRIPTS_TRAIN),
        "--config_select",
        entry["config_path"],
        "--mode",
        "test",
        "--test_model_path",
        entry["checkpoint_path"],
        "--output_path",
        entry["run_dir"],
        "--set",
        f"golden_evaluation_path={entry['run_dir']}/golden_evaluation.json",
    ]
    for key, value in entry["overrides"].items():
        command.extend(["--set", f"{key}={value}"])
    return command


def _relative_path(path):
    return Path(os.path.relpath(path, REPO_ROOT)).as_posix()


def main(argv=None):
    args = _parse_args(argv)
    manifest = _load_manifest(args.manifest)
    output_root = args.output_root or REPO_ROOT / "reports" / "golden_evaluations"
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_entries = [
        _validate_entry(entry, index, output_root)
        for index, entry in enumerate(manifest["evaluations"])
    ]
    entries = manifest_entries
    if args.only:
        selected = set(args.only)
        entries = [entry for entry in entries if entry["id"] in selected]
    run_results = []
    for entry in entries:
        golden_path = Path(entry["run_dir"]) / "golden_evaluation.json"
        if golden_path.exists() and not args.dry_run:
            run_results.append(
                {
                    "id": entry["id"],
                    "status": "already_exists",
                    "golden_evaluation_path": _relative_path(golden_path),
                }
            )
            continue
        if args.dry_run:
            run_results.append(
                {
                    "id": entry["id"],
                    "status": "dry_run",
                    "command": _build_command(entry),
                }
            )
            continue
        command = _build_command(entry)
        env = os.environ.copy()
        source_path = str(REPO_ROOT / "src")
        env["PYTHONPATH"] = os.pathsep.join(
            item for item in (source_path, env.get("PYTHONPATH")) if item
        )
        process = subprocess.run(command, cwd=str(REPO_ROOT), env=env)
        if process.returncode != 0:
            result = {
                "id": entry["id"],
                "status": "failed",
                "returncode": process.returncode,
                "command": command,
            }
            run_results.append(result)
            if args.fail_fast:
                break
        else:
            result = {
                "id": entry["id"],
                "status": "completed",
                "golden_evaluation_path": str(golden_path),
            }
            run_results.append(result)

    run_by_id = {result["id"]: result for result in run_results}
    results = []
    for entry in manifest_entries:
        golden_path = Path(entry["run_dir"]) / "golden_evaluation.json"
        if golden_path.is_file():
            results.append(
                {
                    "id": entry["id"],
                    "status": "completed",
                    "golden_evaluation_path": _relative_path(golden_path),
                }
            )
        elif entry["id"] in run_by_id:
            results.append(run_by_id[entry["id"]])
        else:
            results.append({"id": entry["id"], "status": "missing"})

    index_path = output_root / "index.json"
    index_path.write_text(
        json.dumps({"schema_version": 1, "results": results}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote golden evaluation index: {index_path}")
    return 0 if all(
        item["status"] in {"completed", "already_exists", "dry_run"}
        for item in run_results
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
