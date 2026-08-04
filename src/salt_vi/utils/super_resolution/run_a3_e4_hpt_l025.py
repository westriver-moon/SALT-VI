#!/usr/bin/env python3
"""Run the provenance-gated A3 -> E4 -> HPT-L025 StageB chain."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from datetime import datetime, timezone

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from salt_vi.utils.utils import load_train_configs

RUN_ROOT = REPO_ROOT / "reports" / "a3_e4_hpt_l025"
STATE_PATH = RUN_ROOT / "pipeline_state.json"
E4_CONFIG = REPO_ROOT / "configs" / "stage_b" / "a3_e4_stageb.yaml"
HPT_TEMPLATE = REPO_ROOT / "configs" / "stage_b" / "a3_e4_hpt_l025_template.yaml"
HPT_RUNTIME = RUN_ROOT / "hpt_l025" / "runtime_config.yaml"
A3_CHECKPOINT = REPO_ROOT / "checkpoints" / "stage_a" / "a3_epoch24_tvilfm_full.pth"
A3_MANIFEST = Path(str(A3_CHECKPOINT) + ".manifest.json")
DATA_MANIFEST = Path("/home/cgv841/datasets/derived/SYSU-MM01-swinir-x2-pmt256-v1/manifest.json")
PREFLIGHT_REPORT = RUN_ROOT / "preflight" / "preflight.json"
PYTHON = Path("/home/cgv841/anaconda3/envs/clipreid/bin/python")
VERIFIER = REPO_ROOT / "src" / "salt_vi" / "utils" / "super_resolution" / "verify_a3_stageb_init.py"
CHECKPOINT_PATTERN = re.compile(r"model_Fusion_([0-9]+)\.pth$")


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_state(state):
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    temporary = STATE_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, STATE_PATH)


def verify_phase(config, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log:
        completed = subprocess.run(
            [str(PYTHON), str(VERIFIER), "--config", str(config)],
            cwd=REPO_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        )
    if completed.returncode != 0:
        raise RuntimeError(f"Strict StageB initialization gate failed: {config}")


def run_phase(config, log_path):
    resolved = load_train_configs(str(config))
    visible_devices = str(resolved.CUDA_VISIBLE_DEVICES)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log:
        completed = subprocess.run(
            [str(PYTHON), str(REPO_ROOT / "scripts" / "train.py"), "--config_select", str(config)],
            cwd=REPO_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            env={
                **os.environ,
                "PYTHONUNBUFFERED": "1",
                "CUDA_VISIBLE_DEVICES": visible_devices,
            },
        )
    if completed.returncode != 0:
        raise RuntimeError(f"Training phase failed with return code {completed.returncode}: {config}")


def find_unique_e4_checkpoint():
    candidates = []
    model_root = RUN_ROOT / "e4" / "model_output"
    for path in model_root.rglob("model_Fusion_*.pth"):
        match = CHECKPOINT_PATTERN.fullmatch(path.name)
        if match:
            candidates.append((int(match.group(1)), path.resolve()))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected exactly one A3-E4 Rank-1 checkpoint, got {candidates}")
    return candidates[0]


def write_hpt_runtime(e4_checkpoint):
    HPT_RUNTIME.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        f"extends: {HPT_TEMPLATE}\n"
        f"training_weight_init: {e4_checkpoint}\n"
        f"a3_e4_checkpoint_sha256: {sha256_file(e4_checkpoint)}\n"
    )
    temporary = HPT_RUNTIME.with_suffix(".yaml.tmp")
    temporary.write_text(payload)
    os.replace(temporary, HPT_RUNTIME)


def main(check_only=False):
    for path in (E4_CONFIG, HPT_TEMPLATE, A3_CHECKPOINT, A3_MANIFEST, DATA_MANIFEST, PREFLIGHT_REPORT, PYTHON, VERIFIER):
        if not path.exists():
            raise FileNotFoundError(path)
    preflight = json.loads(PREFLIGHT_REPORT.read_text())
    if preflight.get("formal_training_started") is not False:
        raise RuntimeError("Preflight report does not authorize a fresh formal pipeline launch")
    if not preflight["forward_equivalence"]["exact_equal"] or not preflight["gpu_smoke"]["both_forward_backward_optimizer_step"]:
        raise RuntimeError("Preflight report did not pass forward equivalence and GPU smoke gates")
    implementation_paths = {
        "runner": Path(__file__).resolve(),
        "verifier": VERIFIER,
        "converter": REPO_ROOT / "src" / "salt_vi" / "utils" / "super_resolution" / "convert_stage_a.py",
        "parallel_preflight": REPO_ROOT / "src" / "salt_vi" / "utils" / "super_resolution" / "preflight_a3_e4_multigpu.py",
        "core_build": REPO_ROOT / "src" / "salt_vi" / "engine" / "build.py",
        "main": REPO_ROOT / "scripts" / "train.py",
    }
    for name, path in implementation_paths.items():
        if sha256_file(path) != preflight["implementation"][name]["sha256"]:
            raise RuntimeError(f"Implementation changed after preflight: {name}")
    manifest = json.loads(A3_MANIFEST.read_text())
    if sha256_file(A3_CHECKPOINT) != manifest["target_sha256"]:
        raise RuntimeError("A3 converted checkpoint hash no longer matches its manifest")
    if manifest["source_epoch"] != 24 or manifest["mapping_count"] != 160:
        raise RuntimeError("A3 conversion manifest failed the formal checkpoint gate")

    state = {
        "schema_version": 1,
        "pipeline": "A3 -> E4 -> HPT-L025",
        "status": "running_e4",
        "started_at_utc": now(),
        "pid": os.getpid(),
        "gpu_visible": str(load_train_configs(str(E4_CONFIG)).CUDA_VISIBLE_DEVICES),
        "parallel_strategy": "frozen_visual_chunk_data_parallel",
        "fixed_visual_device_ids": [0, 1, 2, 3],
        "provenance": {
            "a3_checkpoint": str(A3_CHECKPOINT),
            "a3_checkpoint_sha256": sha256_file(A3_CHECKPOINT),
            "a3_manifest_sha256": sha256_file(A3_MANIFEST),
            "data_manifest": str(DATA_MANIFEST),
            "data_manifest_sha256": sha256_file(DATA_MANIFEST),
            "e4_config_sha256": sha256_file(E4_CONFIG),
            "hpt_template_sha256": sha256_file(HPT_TEMPLATE),
            "preflight_report": str(PREFLIGHT_REPORT),
            "preflight_report_sha256": sha256_file(PREFLIGHT_REPORT),
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "verifier_sha256": sha256_file(VERIFIER),
        },
        "e4": {"config": str(E4_CONFIG), "log": str(RUN_ROOT / "e4" / "stdout.log")},
    }
    if check_only:
        verify_phase(E4_CONFIG, RUN_ROOT / "preflight" / "runner_e4_initialization_audit.log")
        print(json.dumps({"initial_gate": "passed", "pipeline": state["pipeline"], "provenance": state["provenance"]}, indent=2))
        return
    save_state(state)
    try:
        verify_phase(E4_CONFIG, RUN_ROOT / "e4" / "initialization_audit.log")
        state["e4"]["initialization_gate"] = "passed"
        save_state(state)
        run_phase(E4_CONFIG, RUN_ROOT / "e4" / "stdout.log")
        epoch, checkpoint = find_unique_e4_checkpoint()
        state["e4"].update({
            "status": "succeeded", "best_rank1_epoch": epoch,
            "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
            "completed_at_utc": now(),
        })
        write_hpt_runtime(checkpoint)
        state["status"] = "running_hpt_l025"
        state["hpt_l025"] = {
            "config": str(HPT_RUNTIME), "config_sha256": sha256_file(HPT_RUNTIME),
            "log": str(RUN_ROOT / "hpt_l025" / "stdout.log"),
            "training_weight_init": str(checkpoint),
        }
        save_state(state)
        verify_phase(HPT_RUNTIME, RUN_ROOT / "hpt_l025" / "initialization_audit.log")
        state["hpt_l025"]["initialization_gate"] = "passed"
        save_state(state)
        run_phase(HPT_RUNTIME, RUN_ROOT / "hpt_l025" / "stdout.log")
        state["status"] = "succeeded"
        state["hpt_l025"].update({"status": "succeeded", "completed_at_utc": now()})
        state["completed_at_utc"] = now()
        save_state(state)
    except BaseException as error:
        state["status"] = "failed"
        state["failed_at_utc"] = now()
        state["error"] = repr(error)
        save_state(state)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    main(check_only=parser.parse_args().check_only)
