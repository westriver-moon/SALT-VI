#!/usr/bin/env python3
"""Validate completed SR assets, then preflight all SR ablations on idle RTX 3090s."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scripts.super_resolution.run_sysu_sr_ablation import atomic_json, idle_gpu_indices
from salt_vi.utils.super_resolution.provenance import (
    assert_clean_algorithm_source,
    build_preflight_provenance,
    provenance_matches,
)
from salt_vi.utils.utils import load_train_configs


CONFIGS = (
    REPO_ROOT / "configs/super_resolution/sr_a0_original_288.yaml",
    REPO_ROOT / "configs/super_resolution/sr_a1_bicubic_x2.yaml",
    REPO_ROOT / "configs/super_resolution/sr_a2_swinir_rgb_x2.yaml",
    REPO_ROOT / "configs/super_resolution/sr_a3_swinir_both_x2.yaml",
)
SOURCE_ROOT = Path("/home/cgv841/datasets/SYSU-MM01")
OUTPUT_ROOT = Path("/home/cgv841/datasets/derived/SYSU-MM01-swinir-x2-v2")
REPORT_ROOT = REPO_ROOT / "reports/super_resolution"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    return parser.parse_args()


def preflight_output(config_path):
    config = load_train_configs(str(config_path))
    return REPORT_ROOT / "preflight" / f"{config.metric_experiment_id}.json"


def preflight_is_current(config_path, reference_preflight=None):
    output = preflight_output(config_path)
    if not output.is_file():
        return False
    try:
        result = json.loads(output.read_text(encoding="utf-8"))
        expected = build_preflight_provenance(
            config_path, REPO_ROOT, reference_preflight=reference_preflight
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False
    return bool(result.get("valid")) and provenance_matches(result, expected)


def run_preflight_stage(config_paths, args, status_path, reference_preflight=None):
    pending = [
        path for path in config_paths
        if not preflight_is_current(path, reference_preflight=reference_preflight)
    ]
    running = {}
    atomic_json(status_path, {
        "status": "preflighting",
        "pending": [path.stem for path in pending],
        "reference_preflight": str(reference_preflight) if reference_preflight else None,
        "updated_at_unix": time.time(),
    })
    while pending or running:
        available, _ = idle_gpu_indices(excluded=running)
        while pending and available:
            gpu_index = available.pop(0)
            config_path = pending.pop(0)
            config = load_train_configs(str(config_path))
            driver_log = REPORT_ROOT / "preflight" / f"{config.metric_experiment_id}.driver.log"
            driver_log.parent.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable,
                str(REPO_ROOT / "scripts/super_resolution/preflight_sysu_sr.py"),
                "--config", str(config_path),
                "--gpu-index", str(gpu_index),
            ]
            if reference_preflight is not None:
                command.extend(["--reference-preflight", str(reference_preflight)])
            log = open(driver_log, "w", encoding="utf-8")
            process = subprocess.Popen(
                command, cwd=str(REPO_ROOT), env=dict(os.environ), stdout=log, stderr=subprocess.STDOUT
            )
            running[gpu_index] = {
                "process": process,
                "log": log,
                "config": config_path,
                "experiment_id": config.metric_experiment_id,
                "driver_log": driver_log,
            }
            print(f"started preflight {config.metric_experiment_id} on GPU {gpu_index}", flush=True)

        finished = []
        for gpu_index, run in running.items():
            returncode = run["process"].poll()
            if returncode is None:
                continue
            run["log"].close()
            valid = returncode == 0 and preflight_is_current(
                run["config"], reference_preflight=reference_preflight
            )
            if not valid:
                atomic_json(status_path, {
                    "status": "preflight_failed",
                    "experiment_id": run["experiment_id"],
                    "physical_gpu_index": gpu_index,
                    "returncode": returncode,
                    "log": str(run["driver_log"]),
                    "updated_at_unix": time.time(),
                })
                raise SystemExit(f"Preflight failed: {run['experiment_id']}")
            finished.append(gpu_index)
        for gpu_index in finished:
            del running[gpu_index]
        if pending or running:
            time.sleep(args.poll_seconds)


def main():
    args = parse_args()
    assert_clean_algorithm_source(REPO_ROOT)
    status_path = REPORT_ROOT / "continuation_status.json"
    manifest = OUTPUT_ROOT / "manifest.json"
    atomic_json(status_path, {"status": "waiting_for_sr_manifest", "updated_at_unix": time.time()})
    while not manifest.is_file():
        time.sleep(args.poll_seconds)

    validation_log = REPORT_ROOT / "validation.log"
    validation_command = [
        sys.executable,
        str(REPO_ROOT / "src/salt_vi/utils/super_resolution/validate_sysu_swinir_x2.py"),
        "--source-root", str(SOURCE_ROOT),
        "--output-root", str(OUTPUT_ROOT),
    ]
    atomic_json(status_path, {"status": "validating_sr_assets", "updated_at_unix": time.time()})
    with open(validation_log, "w", encoding="utf-8") as log:
        completed = subprocess.run(
            validation_command, cwd=str(REPO_ROOT), stdout=log, stderr=subprocess.STDOUT
        )
    if completed.returncode != 0:
        atomic_json(status_path, {
            "status": "validation_failed",
            "returncode": completed.returncode,
            "log": str(validation_log),
            "updated_at_unix": time.time(),
        })
        raise SystemExit("SR asset validation failed")

    run_preflight_stage(CONFIGS[:1], args, status_path)
    a0_reference = preflight_output(CONFIGS[0])
    if not preflight_is_current(CONFIGS[0]):
        raise SystemExit("A0 reference preflight is missing or stale")
    run_preflight_stage(CONFIGS[1:], args, status_path, reference_preflight=a0_reference)

    atomic_json(status_path, {"status": "complete", "updated_at_unix": time.time()})


if __name__ == "__main__":
    main()
