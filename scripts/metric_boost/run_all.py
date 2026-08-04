#!/usr/bin/env python
"""Top-level foreground orchestrator; safe default is preparation only."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import REPORT_ROOT, atomic_write_json, git_commit_sha, utc_now


LAUNCH_CONFIRMATION = "I_UNDERSTAND_GPU_WORK_WILL_START"


def build_commands(mode: str, skip_checkpoint_load: bool = False) -> List[List[str]]:
    python = sys.executable
    preflight = [python, str(SCRIPT_DIR / "preflight.py")]
    if skip_checkpoint_load:
        preflight.append("--skip-checkpoint-load")
    if mode == "prepare-only":
        return [
            preflight,
            [python, str(SCRIPT_DIR / "run_eval_sweep.py"), "--prepare-only"],
            [python, str(SCRIPT_DIR / "run_train_sweep.py"), "--prepare-only"],
            [python, str(SCRIPT_DIR / "summarize_results.py")],
        ]
    if mode == "run-eval":
        return [preflight + ["--require-idle-gpu"], [python, str(SCRIPT_DIR / "run_eval_sweep.py"), "--run"], [python, str(SCRIPT_DIR / "summarize_results.py")]]
    if mode == "run-train":
        return [preflight + ["--require-idle-gpu"], [python, str(SCRIPT_DIR / "run_train_sweep.py"), "--run"], [python, str(SCRIPT_DIR / "summarize_results.py")]]
    raise ValueError(f"Unknown run_all mode: {mode}")


def run_commands(commands: List[List[str]], mode: str) -> Dict[str, object]:
    started = utc_now()
    completed_commands = []
    try:
        for command in commands:
            subprocess.run(command, check=True)
            completed_commands.append(command)
    finally:
        # Summaries are attempted after a failed launch command as well, while
        # preserving the original non-zero exception.
        summary = [sys.executable, str(SCRIPT_DIR / "summarize_results.py")]
        if summary not in completed_commands and mode != "prepare-only":
            subprocess.run(summary, check=False)
    payload = {
        "mode": mode,
        "started_at": started,
        "completed_at": utc_now(),
        "git_commit_sha": git_commit_sha(),
        "commands": completed_commands,
        "foreground_only": True,
        "gpu_work_started": mode != "prepare-only",
    }
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    atomic_write_json(REPORT_ROOT / "orchestration.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--prepare-only", action="store_true", help="Generate all configs, commands, statuses, and reports; start no GPU work")
    group.add_argument("--run-eval", action="store_true", help="Foreground-run the evaluation phase")
    group.add_argument("--run-train", action="store_true", help="Foreground-run the training phase after evaluation is terminal")
    parser.add_argument("--confirm-launch", default="", help="Required exact confirmation token for --run-eval/--run-train")
    parser.add_argument("--skip-checkpoint-load", action="store_true")
    args = parser.parse_args()
    if args.run_eval:
        mode = "run-eval"
    elif args.run_train:
        mode = "run-train"
    else:
        mode = "prepare-only"
    if mode != "prepare-only" and args.confirm_launch != LAUNCH_CONFIRMATION:
        parser.error(
            f"{mode} requires --confirm-launch {LAUNCH_CONFIRMATION}; no GPU work was started"
        )
    payload = run_commands(build_commands(mode, args.skip_checkpoint_load), mode)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

