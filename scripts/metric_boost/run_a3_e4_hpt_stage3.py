#!/usr/bin/env python3
"""Idle-GPU scheduler for A3-E4 SR Stage-3 pair-weight tuning."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shlex
import signal
import subprocess
import sys
import time
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import yaml

try:
    import fcntl
except ImportError:  # pragma: no cover - production scheduling is Linux-only.
    fcntl = None


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
GIT_ROOT = REPO_ROOT.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from salt_vi.utils.utils import load_train_configs
from scripts.metric_boost.run_status import TERMINAL_STATUSES, classify_terminal_status


DEFAULT_PLAN = REPO_ROOT / "configs/metric_boost/a3_e4_hpt_stage3.yaml"
MAIN = REPO_ROOT / "scripts" / "train.py"
CONFIRM_TOKEN = "I_UNDERSTAND_A3_E4_HPT_STAGE3_WILL_START"
TERMINAL = TERMINAL_STATUSES
IDLE_MEMORY_MIB = 2000
IDLE_UTILIZATION_PCT = 20
SHARED_GPU_LOCK_ROOT = Path(
    os.environ.get(
        "SALT_VI_SHARED_GPU_LOCK_ROOT",
        "/home/cgv841/ybj/SALT-VI/reports/metric_boost/gpu_locks",
    )
)

PAIR_NAMES = (
    "RGB-IR",
    "RGB-Fusion",
    "RGB-Text",
    "IR-Fusion",
    "IR-Text",
    "Fusion-Text",
)
EXPECTED_EXPERIMENT_IDS = {
    "A3E4-S3-PAIR-EQUAL",
    "A3E4-S3-PAIR-MILD",
    "A3E4-S3-PAIR-STRONG",
    "A3E4-S3-PAIR-NOTEXT",
}


def utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_text(path: Path, value: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def atomic_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_text(path, yaml.safe_dump(dict(payload), sort_keys=False, allow_unicode=True))


def read_json(path: Path, default: Any = None) -> Any:
    if not Path(path).is_file():
        return default
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_yaml(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.load(handle, Loader=yaml.FullLoader)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected YAML mapping at {path}")
    return payload


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def file_fingerprint(path: Path) -> Dict[str, Any]:
    path = Path(path).resolve()
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256_file(path),
    }


def tree_fingerprint(root: Path) -> Dict[str, Any]:
    root = Path(root).resolve()
    digest = hashlib.sha256()
    count = 0
    total_bytes = 0
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        stat = path.stat()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(4 * 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        count += 1
        total_bytes += stat.st_size
    return {
        "root": str(root),
        "file_count": count,
        "total_bytes": total_bytes,
        "sha256": digest.hexdigest(),
    }


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(GIT_ROOT), *args], text=True).strip()


def source_state() -> Dict[str, Any]:
    if subprocess.run(["git", "-C", str(GIT_ROOT), "diff", "--quiet"]).returncode:
        raise RuntimeError("Formal HPT runs require no tracked worktree changes")
    if subprocess.run(["git", "-C", str(GIT_ROOT), "diff", "--cached", "--quiet"]).returncode:
        raise RuntimeError("Formal HPT runs require no staged changes")
    untracked = git("ls-files", "--others", "--exclude-standard").splitlines()
    relevant_prefixes = (
        "src/salt_vi/entrypoints/train.py",
        "src/salt_vi/engine/",
        "configs/",
        "src/salt_vi/data/",
        "src/salt_vi/models/",
        "scripts/",
        "src/salt_vi/optim/",
        "src/salt_vi/utils/",
    )
    relevant = [path for path in untracked if path.startswith(relevant_prefixes)]
    if relevant:
        raise RuntimeError(f"Algorithm-relevant untracked files prevent formal launch: {relevant}")
    return {
        "head": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "status": git("status", "--porcelain").splitlines(),
        "algorithm_relevant_untracked": relevant,
    }


def load_plan(path: Path = DEFAULT_PLAN) -> Dict[str, Any]:
    plan = load_yaml(path)
    required = {"plan_id", "stage", "base_config", "warm_start", "report_root", "python_bin", "experiments"}
    missing = sorted(required - set(plan))
    if missing:
        raise ValueError(f"Plan is missing required keys: {missing}")
    identifiers = [item["id"] for item in plan["experiments"]]
    if len(identifiers) != 4 or len(identifiers) != len(set(identifiers)):
        raise ValueError("Stage 3 must declare exactly four unique experiment IDs")
    if set(identifiers) != EXPECTED_EXPERIMENT_IDS:
        raise ValueError(
            f"Stage-3 pair structures mismatch: expected {sorted(EXPECTED_EXPERIMENT_IDS)}, "
            f"got {sorted(identifiers)}"
        )
    candidate_gpu_ids = [int(value) for value in plan.get("candidate_gpu_ids", [])]
    if candidate_gpu_ids != [1, 2, 3]:
        raise ValueError("Stage 3 requires candidate_gpu_ids: [1, 2, 3]; GPU 0 is excluded")
    if plan.get("dynamic_launch") is not True:
        raise ValueError("Stage 3 requires dynamic_launch: true")
    expected_warm_start_role = "shared-a3-e4-epoch21-initialization"
    if plan.get("warm_start_role") != expected_warm_start_role:
        raise ValueError(f"Stage 3 warm_start_role must be {expected_warm_start_role!r}")
    for item in plan["experiments"]:
        weights = item.get("overrides", {}).get("cross_modal_pair_weights")
        validate_pair_weights(weights)
    return plan


def validate_pair_weights(weights: Any) -> Dict[str, float]:
    if not isinstance(weights, Mapping):
        raise TypeError("Each Stage-3 experiment must declare cross_modal_pair_weights")
    missing = sorted(set(PAIR_NAMES) - set(weights))
    unknown = sorted(set(weights) - set(PAIR_NAMES))
    if missing or unknown:
        raise ValueError(f"Pair-weight keys mismatch: missing={missing}, unknown={unknown}")
    result = {name: float(weights[name]) for name in PAIR_NAMES}
    if any(not math.isfinite(value) or value < 0.0 for value in result.values()) or sum(result.values()) <= 0.0:
        raise ValueError(f"Invalid Stage-3 pair weights: {result}")
    return result


def build_experiments(plan: Mapping[str, Any]) -> list[Dict[str, Any]]:
    fixed = dict(plan.get("fixed_overrides", {}))
    result = []
    for item in plan["experiments"]:
        overrides = dict(fixed)
        overrides.update(dict(item.get("overrides", {})))
        result.append({**dict(item), "resolved_overrides": overrides})
    return result


def enforce_protocol(runtime: Mapping[str, Any]) -> None:
    required = {
        "mode": "train",
        "dataset": "sysu",
        "training_mode": "RGB_IR_Text",
        "test_modality": "Fusion",
        "test_mode": "all",
        "gall_mode": "single",
        "gallery_trials": 10,
        "Fix_Visual": True,
        "fusion_way": "parameter_add",
        "loss_names": "id,cross_modal_hard",
        "img_h": 512,
        "img_w": 256,
        "sysu_sr_exact_size": True,
        "fixed_visual_data_parallel": False,
        "CAT_EVAL": False,
        "test_flip_tta": False,
        "rerank": False,
        "ensemble_mode": "none",
        "seed": 0,
        "lr_txt": 7.5e-6,
        "cross_modal_hard_weight": 1.25,
        "pa": 0.5,
        "learnable_pa": False,
        "uni_BN": False,
        "qbn_freeze_running_stats_epoch": -1,
    }
    mismatches = {
        key: {"expected": value, "actual": runtime.get(key)}
        for key, value in required.items()
        if runtime.get(key) != value
    }
    if runtime.get("test_multi_scale") != [[512, 256]]:
        mismatches["test_multi_scale"] = {
            "expected": [[512, 256]],
            "actual": runtime.get("test_multi_scale"),
        }
    if list(runtime.get("sysu_source_size", [])) != [256, 128]:
        mismatches["sysu_source_size"] = {"expected": [256, 128], "actual": runtime.get("sysu_source_size")}
    if list(runtime.get("sysu_sr_modalities", [])) != ["rgb", "ir"]:
        mismatches["sysu_sr_modalities"] = {"expected": ["rgb", "ir"], "actual": runtime.get("sysu_sr_modalities")}
    branches = runtime.get("pmt_patch_embed")
    branches = dict(branches) if branches is not None else None
    expected_patch = {
        "anchor_branch": 0,
        "branches": [
            {"patch_size": [16, 16], "stride_size": [12, 12]},
            {"patch_size": [16, 8], "stride_size": [12, 6]},
        ],
    }
    if branches != expected_patch:
        mismatches["pmt_patch_embed"] = {"expected": expected_patch, "actual": branches}
    try:
        validate_pair_weights(runtime.get("cross_modal_pair_weights"))
    except (TypeError, ValueError) as exc:
        mismatches["cross_modal_pair_weights"] = str(exc)
    if mismatches:
        raise ValueError(f"A3-E4 Stage-3 protocol violation: {mismatches}")


def gpu_inventory() -> Dict[int, Dict[str, Any]]:
    raw = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    inventory: Dict[int, Dict[str, Any]] = {}
    uuid_to_index = {}
    for row in csv.reader(raw.splitlines()):
        index, uuid, name, total, used, utilization = [value.strip() for value in row]
        gpu_index = int(index)
        inventory[gpu_index] = {
            "index": gpu_index,
            "uuid": uuid,
            "name": name,
            "memory_total_mib": int(float(total)),
            "memory_used_mib": int(float(used)),
            "utilization_percent": int(float(utilization)),
            "compute_process_count": 0,
        }
        uuid_to_index[uuid] = gpu_index
    processes = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if processes.returncode == 0:
        for row in processes.stdout.splitlines():
            if not row.strip():
                continue
            uuid = row.split(",", 1)[0].strip()
            if uuid in uuid_to_index:
                inventory[uuid_to_index[uuid]]["compute_process_count"] += 1
    return inventory


def idle_from_inventory(
    inventory: Mapping[int, Mapping[str, Any]],
    candidates: Optional[Iterable[int]] = None,
    excluded: Iterable[int] = (),
) -> list[int]:
    allowed = set(inventory) if candidates is None else set(candidates)
    excluded = set(excluded)
    return sorted(
        index
        for index, state in inventory.items()
        if index in allowed
        and index not in excluded
        and "RTX 3090" in str(state["name"])
        and int(state["memory_used_mib"]) < IDLE_MEMORY_MIB
        and int(state["utilization_percent"]) < IDLE_UTILIZATION_PCT
        and int(state.get("compute_process_count", 0)) == 0
    )


def shared_input_fingerprint(
    report_root: Path,
    baseline: Mapping[str, Any],
) -> Dict[str, Any]:
    output = report_root / "shared_input_fingerprint.json"
    existing = read_json(output)
    if existing:
        return existing
    data_root = Path(str(baseline["sysu_data_path"])).resolve()
    text_root = Path(str(baseline["text_data_root"])).resolve()
    sr_root = Path(str(baseline["sysu_sr_data_root"])).resolve()
    training_files = [
        sr_root / "train_rgb_swinir_x2_img.npy",
        data_root / "train_rgb_resized_label.npy",
        sr_root / "train_ir_swinir_x2_img.npy",
        data_root / "train_ir_resized_label.npy",
    ]
    text_files = [
        text_root / "Blip_RGB/train_text_Blip_RGB.npy",
        text_root / "Blip_RGB/train_text_label_Blip_RGB.npy",
        text_root / "Blip_RGB/id_caption_map_Blip_RGB.json",
        text_root / "Blip_IR/train_text_Blip_IR.npy",
        text_root / "Blip_IR/train_text_label_Blip_IR.npy",
        text_root / "Blip_IR/id_caption_map_Blip_IR.json",
    ]
    required = training_files + text_files + [
        data_root / "exp/test_id.txt",
        sr_root / "manifest.json",
        Path(str(baseline["training_weight_init"])),
        Path(str(baseline["pmt_pretrained"])),
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required A3-E4 Stage-3 input files are missing: {missing}")
    payload = {
        "created_at": utc_now(),
        "files": [file_fingerprint(path) for path in required],
        "evaluation_cameras": [
            tree_fingerprint(sr_root / "eval" / f"cam{camera}") for camera in range(1, 7)
        ],
    }
    atomic_json(output, payload)
    return payload


def environment_fingerprint(python_bin: Path, gpu_index: int) -> Dict[str, Any]:
    probe = subprocess.check_output(
        [
            str(python_bin),
            "-c",
            (
                "import json, torch; "
                "print(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,"
                "'cudnn':torch.backends.cudnn.version()}))"
            ),
        ],
        text=True,
    ).strip()
    packages = subprocess.check_output([str(python_bin), "-m", "pip", "freeze"], text=True).splitlines()
    return {
        "captured_at": utc_now(),
        "python_bin": str(python_bin),
        "python_version": subprocess.check_output([str(python_bin), "--version"], text=True, stderr=subprocess.STDOUT).strip(),
        "platform": platform.platform(),
        "torch_cuda": json.loads(probe),
        "pip_freeze": packages,
        "physical_gpu_index": gpu_index,
        "nvidia_smi": subprocess.check_output(["nvidia-smi", "-L"], text=True).splitlines(),
    }


def config_diff(baseline: Mapping[str, Any], runtime: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        key: {"baseline": baseline.get(key), "experiment": runtime.get(key)}
        for key in sorted(set(baseline) | set(runtime))
        if baseline.get(key) != runtime.get(key)
    }


def plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def resolved_config(path: Path) -> Dict[str, Any]:
    return plain(vars(load_train_configs(str(path.resolve()))))


def materialize(
    plan: Mapping[str, Any],
    experiment: Mapping[str, Any],
    gpu_index: int,
    frozen_source: Mapping[str, Any],
    shared_inputs: Mapping[str, Any],
) -> Dict[str, Any]:
    report_root = Path(str(plan["report_root"])).resolve()
    run_dir = report_root / "runs" / experiment["id"]
    manifest_path = run_dir / "manifest.json"
    status_path = run_dir / "status.json"
    if manifest_path.is_file():
        manifest = read_json(manifest_path)
        if manifest.get("git_commit_sha") != frozen_source["head"]:
            raise RuntimeError(f"Existing manifest source mismatch for {experiment['id']}")
        return {
            "run_dir": run_dir,
            "runtime_path": Path(manifest["runtime_config"]),
            "command": manifest["command"],
            "status_path": status_path,
        }
    run_dir.mkdir(parents=True, exist_ok=False)
    base_path = Path(str(plan["base_config"]))
    if not base_path.is_absolute():
        base_path = REPO_ROOT / base_path
    baseline = resolved_config(base_path)
    runtime_spec = {
        "extends": str(base_path.resolve()),
        "training_weight_init": str(Path(str(plan["warm_start"])).resolve()),
        "a3_e4_checkpoint_sha256": str(plan["warm_start_sha256"]),
        **dict(plan.get("fixed_overrides", {})),
        **dict(experiment.get("overrides", {})),
        "CUDA_VISIBLE_DEVICES": str(gpu_index),
        "gpu_id": "0",
        "DataParallel": False,
        "fixed_visual_data_parallel": False,
        "fixed_visual_device_ids": [0],
        "output_path": str(run_dir / "model_output") + "/",
        "metric_events_path": str(run_dir / "events.jsonl"),
        "metric_experiment_id": experiment["id"],
    }
    runtime_path = run_dir / "runtime_config.yaml"
    atomic_yaml(runtime_path, runtime_spec)
    runtime = resolved_config(runtime_path)
    enforce_protocol(runtime)
    atomic_yaml(run_dir / "config_diff.yaml", config_diff(baseline, runtime))
    design = (
        f"# {experiment['id']}\n\n"
        f"Plan: `{plan['plan_id']}`; group: `{experiment['group']}`.\n\n"
        f"{experiment['description']}\n\n"
        "This is one structure in the A3-E4 SR Stage-3 static pair-weight study. "
        "The Stage-2 winning lr_txt and global hard-loss weight are fixed. Every structure "
        "starts afresh from the exact same E4 epoch-21 checkpoint and runs on one physical GPU.\n"
    )
    atomic_text(run_dir / "design.md", design)
    atomic_text(
        run_dir / "code.patch",
        subprocess.check_output(["git", "-C", str(GIT_ROOT), "diff", "--binary", "HEAD"], text=True),
    )
    atomic_json(run_dir / "source_state.json", frozen_source)
    python_bin = Path(str(plan["python_bin"]))
    atomic_json(run_dir / "environment.json", environment_fingerprint(python_bin, gpu_index))
    atomic_json(run_dir / "dataset_fingerprint.json", shared_inputs)
    command = [str(python_bin), str(MAIN), "--config_select", str(runtime_path)]
    atomic_text(run_dir / "command.txt", shlex.join(command) + f"\nworking_directory={REPO_ROOT}\n")
    required = (
        "design.md",
        "runtime_config.yaml",
        "config_diff.yaml",
        "code.patch",
        "source_state.json",
        "environment.json",
        "dataset_fingerprint.json",
        "command.txt",
    )
    artifact_hashes = {name: sha256_file(run_dir / name) for name in required}
    atomic_json(run_dir / "artifact_hashes.json", {"algorithm": "sha256", "files": artifact_hashes})
    manifest = {
        "logging_contract_version": 1,
        "schema_version": 1,
        "experiment_id": experiment["id"],
        "plan_id": plan["plan_id"],
        "stage": plan["stage"],
        "config_path": str(DEFAULT_PLAN.relative_to(REPO_ROOT)),
        "runtime_config": str(runtime_path),
        "git_commit_sha": frozen_source["head"],
        "dataset": "sysu",
        "protocol": "all-search-single-shot-10-trial",
        "seed": int(runtime["seed"]),
        "max_epoch": int(runtime["total_train_epoch"]),
        "epoch_index_origin": 0,
        "planned_metric_names": ["Rank-1", "mAP", "mINP"],
        "planned_loss_names": ["id_loss", "cross_modal_hard_loss", "total_loss"],
        "selection_validity": plan["selection_validity"],
        "selection_rule": plan["selection_rule"],
        "physical_gpu_index": gpu_index,
        "command": command,
        "artifact_hashes_sha256": sha256_file(run_dir / "artifact_hashes.json"),
        "created_at": utc_now(),
    }
    atomic_json(manifest_path, manifest)
    os.chmod(manifest_path, 0o444)
    atomic_json(
        status_path,
        {
            "experiment_id": experiment["id"],
            "status": "pending",
            "updated_at": utc_now(),
            "start_time": None,
            "end_time": None,
            "runner_pid": None,
            "gpu": gpu_index,
            "last_completed_epoch": None,
            "return_code": None,
            "error": None,
        },
    )
    return {"run_dir": run_dir, "runtime_path": runtime_path, "command": command, "status_path": status_path}


def best_result(events_path: Path) -> Optional[Dict[str, Any]]:
    rows = []
    if events_path.is_file():
        for line in events_path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event_type") == "eval_epoch" and event.get("metrics", {}).get("Rank-1") is not None:
                rows.append(event)
    if not rows:
        return None
    best = max(rows, key=lambda row: float(row["metrics"]["Rank-1"]))
    checkpoint_paths = best.get("checkpoint_paths", {})
    checkpoint = checkpoint_paths.get("Rank-1")
    return {
        "best_epoch": best.get("epoch"),
        "Rank-1": best["metrics"].get("Rank-1"),
        "mAP": best["metrics"].get("mAP"),
        "mINP": best["metrics"].get("mINP"),
        "checkpoint": checkpoint,
        "checkpoint_sha256": sha256_file(Path(checkpoint)) if checkpoint and Path(checkpoint).is_file() else None,
    }


def pid_alive(pid: Optional[int]) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except (OSError, ValueError):
        return False
    return True


class Scheduler:
    def __init__(
        self,
        plan_path: Path,
        poll_seconds: float,
        candidates: Optional[set[int]],
        max_parallel: int,
    ):
        self.plan_path = plan_path.resolve()
        self.plan = load_plan(self.plan_path)
        self.experiments = build_experiments(self.plan)
        self.report_root = Path(str(self.plan["report_root"])).resolve()
        self.poll_seconds = poll_seconds
        configured_candidates = set(int(value) for value in self.plan["candidate_gpu_ids"])
        self.candidates = configured_candidates if candidates is None else set(candidates)
        if not self.candidates or not self.candidates.issubset(configured_candidates):
            raise ValueError(
                f"Stage 3 GPU candidates must be a non-empty subset of "
                f"{sorted(configured_candidates)}; got {sorted(self.candidates)}"
            )
        if 0 in self.candidates:
            raise ValueError("GPU 0 is explicitly excluded from Stage 3")
        if max_parallel > len(self.candidates):
            raise ValueError("max_parallel cannot exceed the number of candidate GPUs")
        self.max_parallel = max_parallel
        self.running: Dict[int, Dict[str, Any]] = {}
        self.stop_requested = False
        self.frozen_source = source_state()
        base_path = Path(str(self.plan["base_config"]))
        if not base_path.is_absolute():
            base_path = REPO_ROOT / base_path
        self.baseline = resolved_config(base_path)
        probe = dict(self.baseline)
        probe.update(dict(self.plan["fixed_overrides"]))
        probe.update({
            "training_weight_init": str(Path(str(self.plan["warm_start"])).resolve()),
            "fixed_visual_data_parallel": False,
        })
        enforce_protocol(probe)
        warm_start = Path(str(self.plan["warm_start"])).resolve()
        if not warm_start.is_file() or sha256_file(warm_start) != str(self.plan["warm_start_sha256"]):
            raise RuntimeError("Shared A3-E4 epoch-21 warm-start is missing or its SHA-256 changed")
        self.baseline = probe
        self.shared_inputs = shared_input_fingerprint(self.report_root, self.baseline)

    @property
    def state_path(self) -> Path:
        return self.report_root / "scheduler_state.json"

    def status(self, experiment_id: str) -> Dict[str, Any]:
        return read_json(self.report_root / "runs" / experiment_id / "status.json", {})

    def write_state(self, state: str, reason: Optional[str] = None) -> None:
        inventory = gpu_inventory()
        statuses = {item["id"]: self.status(item["id"]).get("status", "missing") for item in self.experiments}
        atomic_json(
            self.state_path,
            {
                "plan_id": self.plan["plan_id"],
                "state": state,
                "reason": reason,
                "scheduler_pid": os.getpid(),
                "updated_at": utc_now(),
                "source": self.frozen_source,
                "status_by_experiment": statuses,
                "running": {
                    run["experiment_id"]: {"gpu": gpu, "pid": run["pid"]}
                    for gpu, run in self.running.items()
                },
                "idle_gpu_ids": idle_from_inventory(inventory, self.candidates, self.running),
                "gpu_inventory": inventory,
            },
        )

    def pending(self) -> list[Dict[str, Any]]:
        result = []
        active_ids = {run["experiment_id"] for run in self.running.values()}
        for item in self.experiments:
            status = self.status(item["id"]).get("status")
            if item["id"] in active_ids or status in TERMINAL or status == "running":
                continue
            result.append(item)
        return result

    def try_gpu_lease(self, gpu_index: int):
        SHARED_GPU_LOCK_ROOT.mkdir(parents=True, exist_ok=True)
        handle = (SHARED_GPU_LOCK_ROOT / f"gpu-{gpu_index}.lock").open("a+", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            return None
        inventory = gpu_inventory()
        if gpu_index not in idle_from_inventory(inventory, {gpu_index}, self.running):
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()
            return None
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps({"pid": os.getpid(), "gpu": gpu_index, "acquired_at": utc_now()}))
        handle.flush()
        return handle

    def launch_available(self) -> None:
        """Fill each currently idle candidate GPU with one pending experiment."""
        if len(self.running) >= self.max_parallel:
            return
        inventory = gpu_inventory()
        available = idle_from_inventory(inventory, self.candidates, self.running)
        for gpu_index in available:
            if len(self.running) >= self.max_parallel:
                break
            pending = self.pending()
            if not pending:
                break
            lease = self.try_gpu_lease(gpu_index)
            if lease is None:
                continue
            experiment = pending[0]
            current_source = source_state()
            if current_source["head"] != self.frozen_source["head"]:
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
                lease.close()
                raise RuntimeError("Source commit changed while HPT scheduler was active")
            try:
                materialized = materialize(
                    self.plan, experiment, gpu_index, self.frozen_source, self.shared_inputs
                )
                log = (materialized["run_dir"] / "launcher.log").open("a", encoding="utf-8")
                process = subprocess.Popen(
                    materialized["command"], cwd=str(REPO_ROOT),
                    env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_index)},
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            except Exception:
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
                lease.close()
                raise
            launch_time = utc_now()
            atomic_json(materialized["status_path"], {
                "experiment_id": experiment["id"], "status": "running",
                "updated_at": launch_time, "start_time": launch_time, "end_time": None,
                "runner_pid": process.pid, "gpu": gpu_index, "last_completed_epoch": None,
                "return_code": None, "error": None,
                "runtime_config": str(materialized["runtime_path"]),
                "log_path": str(materialized["run_dir"] / "launcher.log"),
            })
            self.running[gpu_index] = {
                "experiment_id": experiment["id"], "process": process, "pid": process.pid,
                "lease": lease, "log": log, **materialized,
            }
            print(f"started {experiment['id']} on physical GPU {gpu_index} (pid={process.pid})", flush=True)

    def reap(self) -> None:
        finished = []
        for gpu_index, run in self.running.items():
            return_code = run["process"].poll()
            if return_code is None:
                continue
            run["log"].close()
            result = best_result(run["run_dir"] / "events.jsonl")
            terminal_status = classify_terminal_status(
                return_code,
                bool(result),
                user_stop_requested=(run["run_dir"] / "user_stop.json").is_file(),
            )
            status = {
                "experiment_id": run["experiment_id"],
                "status": terminal_status,
                "updated_at": utc_now(),
                "start_time": self.status(run["experiment_id"]).get("start_time"),
                "end_time": utc_now(),
                "runner_pid": run["pid"],
                "gpu": gpu_index,
                "last_completed_epoch": result.get("best_epoch") if result else None,
                "return_code": return_code,
                "error": None if terminal_status in {"succeeded", "stopped_by_user"} else (
                    f"Training returned {return_code}" if return_code else "No eval_epoch metrics were emitted"
                ),
            }
            if result:
                status.update(result)
            atomic_json(run["status_path"], status)
            fcntl.flock(run["lease"].fileno(), fcntl.LOCK_UN)
            run["lease"].close()
            finished.append(gpu_index)
        for gpu_index in finished:
            del self.running[gpu_index]

    def run(self) -> None:
        self.report_root.mkdir(parents=True, exist_ok=True)
        atomic_yaml(self.report_root / "resolved_plan.yaml", self.plan)
        while not self.stop_requested:
            self.reap()
            self.launch_available()
            if not self.pending() and not self.running:
                statuses = {item["id"]: self.status(item["id"]).get("status") for item in self.experiments}
                failures = [key for key, value in statuses.items() if value in {"failed", "blocked"}]
                stopped = [key for key, value in statuses.items() if value in {"stopped_by_user", "cancelled"}]
                state = "completed_with_failures" if failures else "completed_with_stopped_runs" if stopped else "completed"
                self.write_state(state, ", ".join(failures or stopped) or None)
                return
            self.write_state("running")
            time.sleep(self.poll_seconds)
        self.write_state("stopped", "Signal received; running trainers were left untouched")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--confirm-launch")
    parser.add_argument("--gpu-indices", help="Comma-separated physical GPU allowlist; default all RTX 3090 GPUs")
    parser.add_argument("--max-parallel", type=int, default=3)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    args = parser.parse_args()
    plan = load_plan(args.plan)
    report_root = Path(str(plan["report_root"])).resolve()
    if args.status:
        print(json.dumps(read_json(report_root / "scheduler_state.json", {"state": "not-started"}), ensure_ascii=False, sort_keys=True))
        return
    if args.preflight:
        base_path = Path(str(plan["base_config"]))
        if not base_path.is_absolute():
            base_path = REPO_ROOT / base_path
        baseline = resolved_config(base_path)
        baseline.update(dict(plan["fixed_overrides"]))
        baseline.update({
            "training_weight_init": str(Path(str(plan["warm_start"])).resolve()),
            "fixed_visual_data_parallel": False,
        })
        resolved_experiments = []
        for experiment in build_experiments(plan):
            runtime = dict(baseline)
            runtime.update(dict(experiment["resolved_overrides"]))
            enforce_protocol(runtime)
            resolved_experiments.append({
                "id": experiment["id"],
                "lr_txt": runtime["lr_txt"],
                "cross_modal_hard_weight": runtime["cross_modal_hard_weight"],
                "cross_modal_pair_weights": validate_pair_weights(runtime["cross_modal_pair_weights"]),
                "training_weight_init": runtime["training_weight_init"],
            })
        warm_start = Path(str(plan["warm_start"])).resolve()
        inventory = gpu_inventory()
        idle_gpu_ids = idle_from_inventory(inventory, plan["candidate_gpu_ids"])
        payload = {
            "plan_id": plan["plan_id"],
            "source": source_state(),
            "experiment_count": len(build_experiments(plan)),
            "experiments": resolved_experiments,
            "gpu_inventory": inventory,
            "python_exists": Path(str(plan["python_bin"])).is_file(),
            "warm_start_exists": warm_start.is_file(),
            "warm_start_sha256_matches": warm_start.is_file() and sha256_file(warm_start) == str(plan["warm_start_sha256"]),
            "idle_gpu_ids": idle_gpu_ids,
            "candidate_gpu_ids": plan["candidate_gpu_ids"],
            "dynamic_launch": plan["dynamic_launch"],
        }
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return
    if not args.run:
        parser.error("Choose --run, --preflight, or --status")
    if fcntl is None:
        parser.error("--run requires Linux fcntl")
    if args.confirm_launch != CONFIRM_TOKEN:
        parser.error(f"--run requires --confirm-launch {CONFIRM_TOKEN}")
    if args.poll_seconds < 1 or args.max_parallel < 1:
        parser.error("poll-seconds and max-parallel must be positive")
    candidates = None
    if args.gpu_indices:
        candidates = {int(value) for value in args.gpu_indices.split(",") if value.strip()}
    scheduler = Scheduler(args.plan, args.poll_seconds, candidates, args.max_parallel)

    def request_stop(signum, frame):
        scheduler.stop_requested = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    report_root.mkdir(parents=True, exist_ok=True)
    lock_path = report_root / "scheduler.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another A3-E4 Stage-3 scheduler is already active") from exc
        atomic_text(report_root / "scheduler.log", f"scheduler_pid={os.getpid()} started_at={utc_now()}\n")
        scheduler.write_state("starting")
        scheduler.run()


if __name__ == "__main__":
    main()
