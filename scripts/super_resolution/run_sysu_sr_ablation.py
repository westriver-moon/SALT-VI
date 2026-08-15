#!/usr/bin/env python3
"""Provenance-gated dynamic multi-GPU runner for the four SYSU SR ablations."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import time

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
GIT_ROOT = REPO_ROOT.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from salt_vi.utils.super_resolution.build_sysu_swinir_x2 import sha256_file
from salt_vi.utils.super_resolution.provenance import build_preflight_provenance, provenance_matches
from salt_vi.utils.utils import load_train_configs


CONFIGS = (
    REPO_ROOT / "configs/super_resolution/sr_a0_original_288.yaml",
    REPO_ROOT / "configs/super_resolution/sr_a1_bicubic_x2.yaml",
    REPO_ROOT / "configs/super_resolution/sr_a2_swinir_rgb_x2.yaml",
    REPO_ROOT / "configs/super_resolution/sr_a3_swinir_both_x2.yaml",
)


def plain_data(value):
    """Convert EasyDict/config containers into YAML-safe builtin containers."""
    if isinstance(value, dict):
        return {key: plain_data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain_data(item) for item in value]
    return value


def apply_gpu_assignment(runtime, gpu_index):
    """Keep the launcher environment and main.py's config override aligned."""
    runtime["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    runtime["gpu_id"] = "0"
    return runtime


def atomic_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path, payload):
    atomic_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def git(*args):
    return subprocess.check_output(["git", "-C", str(GIT_ROOT), *args], text=True).strip()


def assert_source_state(expected_ref="origin/main"):
    head = git("rev-parse", "HEAD")
    expected_head = git("rev-parse", expected_ref)
    if head != expected_head:
        raise RuntimeError(
            f"Formal SR runs require HEAD {expected_head} from {expected_ref}, found {head}"
        )
    if subprocess.run(["git", "-C", str(GIT_ROOT), "diff", "--quiet"]).returncode != 0:
        raise RuntimeError("Formal SR runs require no tracked worktree changes")
    if subprocess.run(["git", "-C", str(GIT_ROOT), "diff", "--cached", "--quiet"]).returncode != 0:
        raise RuntimeError("Formal SR runs require no staged changes")
    untracked = git("ls-files", "--others", "--exclude-standard").splitlines()
    relevant = [path for path in untracked if path.startswith(("src/salt_vi/engine/", "configs/", "src/salt_vi/data/", "src/salt_vi/models/", "scripts/", "src/salt_vi/utils/"))]
    if relevant:
        raise RuntimeError(f"Algorithm-relevant untracked files: {relevant}")
    return relevant


def gpu_inventory():
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,name,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        text=True,
    )
    inventory = {}
    for line in output.splitlines():
        index, name, memory, utilization = [value.strip() for value in line.split(",")]
        inventory[int(index)] = {
            "name": name,
            "memory_used_mib": int(memory),
            "utilization_percent": int(utilization),
        }
    return inventory


def idle_gpu_indices(candidates=None, excluded=()):
    inventory = gpu_inventory()
    candidates = set(inventory) if candidates is None else set(candidates)
    excluded = set(excluded)
    return sorted(
        index for index, state in inventory.items()
        if index in candidates
        and index not in excluded
        and "RTX 3090" in state["name"]
        and state["memory_used_mib"] < 500
        and state["utilization_percent"] < 10
    ), inventory


def hash_artifacts(run_dir, names):
    return {name: sha256_file(run_dir / name) for name in names}


def materialize(config_path, preflight, untracked, gpu_index):
    config = load_train_configs(str(config_path))
    experiment_id = config.metric_experiment_id
    run_dir = REPO_ROOT / "reports/super_resolution/runs" / experiment_id
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Immutable manifest already exists: {manifest_path}")
    run_dir.mkdir(parents=True, exist_ok=True)
    runtime = plain_data(dict(config))
    runtime["visual_forward_chunk_size"] = int(preflight["selected_chunk_size"])
    runtime["test_batch_size"] = int(preflight["selected_test_batch_size"])
    # main.py reapplies this config value after startup, so it must match the
    # physical assignment made by the scheduler. Inside that visibility mask
    # the model still addresses the assigned card as cuda:0.
    apply_gpu_assignment(runtime, gpu_index)
    runtime["output_path"] = str(run_dir / "model_output") + "/"
    runtime["metric_events_path"] = str(run_dir / "events.jsonl")
    runtime_path = run_dir / "runtime_config.yaml"
    atomic_text(runtime_path, yaml.safe_dump(runtime, sort_keys=True))

    base = plain_data(dict(load_train_configs(str(CONFIGS[0]))))
    diff = {
        key: {"baseline": base.get(key), "experiment": runtime.get(key)}
        for key in sorted(set(base) | set(runtime))
        if base.get(key) != runtime.get(key)
    }
    atomic_text(run_dir / "config_diff.yaml", yaml.safe_dump(diff, sort_keys=True))
    design = (
        f"# {experiment_id}\n\n"
        "Formal seed-0 SYSU-MM01 super-resolution ablation. All groups inherit the corrected-text "
        "FGAP2-P1 configuration and keep PK=8x4, losses, warm start, optimizer, schedule and protocol fixed. "
        "Each run occupies one dynamically assigned idle RTX 3090; groups may run concurrently.\n\n"
        f"SR modalities: {runtime.get('sysu_sr_modalities', [])}; input size: {runtime['img_size']}. "
        "A1-A0 isolates resolution, A2-A1 isolates RGB SwinIR, and A3-A2 isolates IR SwinIR.\n"
    )
    atomic_text(run_dir / "design.md", design)
    atomic_text(run_dir / "code.patch", subprocess.check_output(["git", "-C", str(GIT_ROOT), "diff", "--binary", "HEAD"], text=True))
    source_state = {
        "head": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "untracked_paths": untracked,
        "status": git("status", "--porcelain").splitlines(),
    }
    atomic_json(run_dir / "source_state.json", source_state)
    environment = {
        "python": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "physical_gpu_index": gpu_index,
        "cuda_visible_devices": str(gpu_index),
        "nvidia_smi": subprocess.check_output(["nvidia-smi", "-L"], text=True).splitlines(),
    }
    atomic_json(run_dir / "environment.json", environment)
    source_root = Path(runtime["sysu_data_path"])
    dataset = {
        "source_root": str(source_root.resolve()),
        "train_rgb_sha256": sha256_file(source_root / "train_rgb_resized_img.npy"),
        "train_ir_sha256": sha256_file(source_root / "train_ir_resized_img.npy"),
        "text_manifest_sha256": sha256_file(Path(runtime["text_data_root"]) / "manifest.json"),
        "warm_start_sha256": sha256_file(Path(runtime["training_weight_init"])),
    }
    preflight_inputs = preflight["provenance"]
    dataset.update({
        "preflight_contract_sha256": preflight_inputs["contract_sha256"],
        "train_rgb_label_sha256": preflight_inputs["train_rgb_label"]["sha256"],
        "train_ir_label_sha256": preflight_inputs["train_ir_label"]["sha256"],
        "test_id_sha256": preflight_inputs["test_id"]["sha256"],
        "source_evaluation_tree": preflight_inputs["source_evaluation_tree"],
        "text_assets": preflight_inputs["text_assets"],
    })
    if runtime.get("sysu_sr_modalities"):
        sr_manifest = Path(runtime["sysu_sr_data_root"]) / "manifest.json"
        dataset["sr_manifest"] = str(sr_manifest)
        dataset["sr_manifest_sha256"] = sha256_file(sr_manifest)
    atomic_json(run_dir / "dataset_fingerprint.json", dataset)
    command = [sys.executable, str(REPO_ROOT / "scripts" / "train.py"), "--config_select", str(runtime_path)]
    atomic_text(run_dir / "command.txt", shlex.join(command) + f"\nworking_directory={REPO_ROOT}\n")
    required = ("design.md", "runtime_config.yaml", "config_diff.yaml", "code.patch", "source_state.json", "environment.json", "dataset_fingerprint.json", "command.txt")
    hashes = hash_artifacts(run_dir, required)
    atomic_json(run_dir / "artifact_hashes.json", {"algorithm": "sha256", "files": hashes})
    manifest = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "git_commit_sha": source_state["head"],
        "seed": int(runtime["seed"]),
        "max_epoch": int(runtime["total_train_epoch"]),
        "input_size": runtime["img_size"],
        "sr_modalities": runtime.get("sysu_sr_modalities", []),
        "runtime_config": str(runtime_path),
        "preflight": preflight,
        "physical_gpu_index": gpu_index,
        "command": command,
        "artifact_hashes_sha256": sha256_file(run_dir / "artifact_hashes.json"),
        "created_at_unix": time.time(),
        "selection_rule": "highest Rank-1; report mAP and mINP from the same epoch",
        "validity": "preliminary seed-0 ablation; no cross-seed significance claim",
    }
    atomic_json(manifest_path, manifest)
    os.chmod(manifest_path, 0o444)
    return run_dir, runtime_path, command


def best_rank1(events_path):
    rows = []
    if events_path.exists():
        for line in events_path.read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            if event.get("event_type") == "eval_epoch":
                rows.append(event)
    if not rows:
        return None
    best = max(rows, key=lambda row: row["metrics"]["Rank-1"])
    checkpoint_paths = best.get("checkpoint_paths", {})
    checkpoint_path = checkpoint_paths.get("Rank-1") or checkpoint_paths.get("warm_start")
    result = {
        "epoch": best["epoch"],
        **best["metrics"],
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": None,
    }
    if checkpoint_path and Path(checkpoint_path).is_file():
        result["checkpoint_sha256"] = sha256_file(Path(checkpoint_path))
    return result


def available_preflight(config_path):
    config = load_train_configs(str(config_path))
    path = REPO_ROOT / "reports/super_resolution/preflight" / f"{config.metric_experiment_id}.json"
    if not path.is_file():
        return None
    preflight = json.loads(path.read_text(encoding="utf-8"))
    if not preflight.get("valid"):
        raise RuntimeError(f"Failed preflight for {config.metric_experiment_id}")
    reference = None
    if Path(config_path).resolve() != CONFIGS[0].resolve():
        reference_config = load_train_configs(str(CONFIGS[0]))
        reference = (
            REPO_ROOT / "reports/super_resolution/preflight"
            / f"{reference_config.metric_experiment_id}.json"
        )
    expected = build_preflight_provenance(
        config_path, REPO_ROOT, reference_preflight=reference
    )
    if not provenance_matches(preflight, expected):
        raise RuntimeError(f"Stale preflight provenance for {config.metric_experiment_id}")
    return preflight


def validate_selected_sr_assets(config_paths):
    """Run the semantic validator once before any formal SR-backed run starts."""
    roots = {}
    for config_path in config_paths:
        config = load_train_configs(str(config_path))
        if config.get("sysu_sr_modalities", []):
            roots[Path(config.sysu_sr_data_root).resolve()] = Path(config.sysu_data_path).resolve()
    report_dir = REPO_ROOT / "reports/super_resolution"
    report_dir.mkdir(parents=True, exist_ok=True)
    for output_root, source_root in roots.items():
        command = [
            sys.executable,
            str(REPO_ROOT / "src/salt_vi/utils/super_resolution/validate_sysu_swinir_x2.py"),
            "--source-root", str(source_root),
            "--output-root", str(output_root),
        ]
        completed = subprocess.run(command, cwd=str(REPO_ROOT), text=True, capture_output=True)
        atomic_text(report_dir / "validation.json", completed.stdout)
        if completed.returncode != 0:
            atomic_text(report_dir / "validation.stderr.log", completed.stderr)
            raise RuntimeError(f"SR semantic validation failed for {output_root}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-at", choices=[path.stem for path in CONFIGS], default=CONFIGS[0].stem)
    parser.add_argument("--gpu-indices", help="Comma-separated physical GPU allowlist; default: all RTX 3090 GPUs.")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--expected-ref", default="origin/main",
                        help="Git ref whose commit must exactly equal HEAD for a formal run.")
    args = parser.parse_args()
    untracked = assert_source_state(args.expected_ref)
    frozen_head = git("rev-parse", "HEAD")
    candidates = None
    if args.gpu_indices:
        candidates = {int(value) for value in args.gpu_indices.split(",") if value.strip()}
    inventory = gpu_inventory()
    known_3090s = {index for index, state in inventory.items() if "RTX 3090" in state["name"]}
    candidates = known_3090s if candidates is None else candidates
    if not candidates or not candidates <= known_3090s:
        raise RuntimeError(f"GPU allowlist must contain available RTX 3090 indices: {sorted(known_3090s)}")

    selected_configs = []
    started = False
    for config_path in CONFIGS:
        if config_path.stem == args.start_at:
            started = True
        if not started:
            continue
        selected_configs.append(config_path)

    validate_selected_sr_assets(selected_configs)

    lock_path = REPO_ROOT / "reports/super_resolution/scheduler.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pending = list(selected_configs)
        running = {}
        summary = {}
        failure = None
        while pending or running:
            available, _ = idle_gpu_indices(candidates, excluded=running)
            while pending and available and failure is None:
                ready_index = None
                preflight = None
                for index, candidate in enumerate(pending):
                    candidate_preflight = available_preflight(candidate)
                    if candidate_preflight is not None:
                        ready_index = index
                        preflight = candidate_preflight
                        break
                if ready_index is None:
                    break
                gpu_index = available.pop(0)
                config_path = pending.pop(ready_index)
                config = load_train_configs(str(config_path))
                current_untracked = assert_source_state(args.expected_ref)
                if git("rev-parse", "HEAD") != frozen_head:
                    raise RuntimeError("Source commit changed while the scheduler was running")
                if current_untracked != untracked:
                    raise RuntimeError("Untracked source state changed while the scheduler was running")
                run_dir, runtime_path, command = materialize(
                    config_path, preflight, untracked, gpu_index
                )
                status_path = run_dir / "status.json"
                atomic_json(status_path, {
                    "status": "running",
                    "physical_gpu_index": gpu_index,
                    "started_at_unix": time.time(),
                })
                log = open(run_dir / "launcher.log", "w", encoding="utf-8")
                process = subprocess.Popen(
                    command,
                    cwd=str(REPO_ROOT),
                    env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_index)},
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                running[gpu_index] = {
                    "process": process,
                    "log": log,
                    "run_dir": run_dir,
                    "status_path": status_path,
                    "experiment_id": config.metric_experiment_id,
                }
                print(f"started {config.metric_experiment_id} on physical GPU {gpu_index} (pid={process.pid})", flush=True)

            finished = []
            for gpu_index, run in running.items():
                returncode = run["process"].poll()
                if returncode is None:
                    continue
                run["log"].close()
                result = best_rank1(run["run_dir"] / "events.jsonl")
                atomic_json(run["status_path"], {
                    "status": "complete" if returncode == 0 else "failed",
                    "returncode": returncode,
                    "physical_gpu_index": gpu_index,
                    "finished_at_unix": time.time(),
                    "best_rank1_epoch": result,
                })
                summary[run["experiment_id"]] = result
                finished.append(gpu_index)
                if returncode != 0 and failure is None:
                    failure = RuntimeError(
                        f"Training failed for {run['experiment_id']}; see {run['run_dir'] / 'launcher.log'}"
                    )
            for gpu_index in finished:
                del running[gpu_index]
            if failure is not None and not running:
                raise failure
            if pending or running:
                time.sleep(args.poll_seconds)
        atomic_json(REPO_ROOT / "reports/super_resolution/summary_seed0.json", summary)


if __name__ == "__main__":
    main()
