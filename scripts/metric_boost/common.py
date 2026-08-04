"""Shared, resumable, safety-first helpers for metric-boost experiments."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import subprocess
from contextlib import AbstractContextManager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import yaml

try:
    import fcntl
except ImportError:  # pragma: no cover - the experiment server is Linux.
    fcntl = None


REPO_ROOT = Path(__file__).resolve().parents[2]
E4_CONFIG_PATH = REPO_ROOT / "configs/stage_b/vit_source_core_sysu_no_sff_parameter_add_pa05.yaml"
REPORT_ROOT = REPO_ROOT / "reports/metric_boost"
CONFIG_ROOT = REPO_ROOT / "configs/metric_boost"
EXPERIMENT_MANIFEST = CONFIG_ROOT / "experiments.yaml"
EXPECTED_E4 = {"Rank-1": 0.81620, "mAP": 0.78663, "mINP": 0.67711}
E4_TOLERANCE = 5e-4
GPU_IDLE_MEMORY_MIB = 2000
GPU_IDLE_UTIL_PCT = 20

BEST_RE = re.compile(
    r"Best Fusion_RGB mINP:\s*([0-9.eE+-]+),\s*Best mAP:\s*([0-9.eE+-]+),\s*Best Rank1:\s*([0-9.eE+-]+)"
)
EPOCH_RE = re.compile(r"Time:\s*[0-9:-]+;\s*Epoch:\s*(\d+);")
RESULT_RE = re.compile(
    r"mINP:\s*([0-9.eE+-]+)\s*\n?mAP:\s*([0-9.eE+-]+)\s*\n?\s*Rank:\s*\[?([0-9.eE+-]+)"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_yaml(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        # The project's canonical default.yaml contains explicit python/tuple
        # tags, so FullLoader is required for parity with tools.utils.
        payload = yaml.load(handle, Loader=yaml.FullLoader)
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise TypeError(f"Expected YAML mapping at {path}, got {type(payload).__name__}")
    return payload


def atomic_write_text(path: Path, value: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def atomic_write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_write_text(path, yaml.safe_dump(dict(payload), sort_keys=False, allow_unicode=True))


def read_json(path: Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.is_file():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def git_commit_sha(repo_root: Path = REPO_ROOT) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(repo_root), text=True
    ).strip()


def resolve_sysu_text_root(configured: Optional[str] = None) -> Path:
    candidates = []
    if configured:
        candidates.append(Path(configured).expanduser())
    environment = os.environ.get("SALT_VI_SYSU_TEXT_ROOT")
    if environment:
        candidates.append(Path(environment).expanduser())
    candidates.extend(
        [
            REPO_ROOT / "datasets/sysu/Text",
            Path("/home/cgv841/ybj/SALT-VI/datasets/sysu/Text"),
        ]
    )
    checked = []
    for candidate in candidates:
        candidate = candidate.resolve()
        if str(candidate) in checked:
            continue
        checked.append(str(candidate))
        if (candidate / "Blip_RGB").is_dir() and (candidate / "Blip_IR").is_dir():
            return candidate
    raise FileNotFoundError(f"Unable to locate shared SYSU text assets; checked {checked}")


def resolve_pmt_pretrained(configured: Optional[str]) -> Path:
    if not configured:
        raise FileNotFoundError("pmt_pretrained is not configured")
    value = Path(configured).expanduser()
    candidates = [value] if value.is_absolute() else [
        REPO_ROOT / value,
        REPO_ROOT.parent / value,
        Path("/home/cgv841/ybj") / value,
    ]
    checked = []
    for candidate in candidates:
        candidate = candidate.resolve()
        if str(candidate) in checked:
            continue
        checked.append(str(candidate))
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Unable to locate PMT pretrained weights; checked {checked}")


def metrics_match_reference(
    metrics: Mapping[str, float],
    expected: Mapping[str, float] = EXPECTED_E4,
    tolerance: float = E4_TOLERANCE,
) -> bool:
    return all(abs(float(metrics[key]) - float(expected[key])) <= tolerance for key in expected)


def metric_error(metrics: Mapping[str, float], expected: Mapping[str, float] = EXPECTED_E4) -> float:
    return sum(abs(float(metrics[key]) - float(expected[key])) for key in expected)


def resolve_e4_checkpoint(
    config_path: Path = E4_CONFIG_PATH,
    expected: Mapping[str, float] = EXPECTED_E4,
    tolerance: float = E4_TOLERANCE,
) -> Dict[str, Any]:
    """Locate E4 from its configured output tree and corroborating log metrics."""
    config_path = Path(config_path).resolve()
    config = load_yaml(config_path)
    output_root = Path(str(config.get("output_path", ""))).expanduser()
    if not output_root.is_absolute():
        output_root = (REPO_ROOT / output_root).resolve()
    if not output_root.is_dir():
        raise FileNotFoundError(f"E4 output_path does not exist: {output_root}")

    candidates: List[Dict[str, Any]] = []
    for log_path in sorted(output_root.rglob("log.log")):
        current_epoch: Optional[int] = None
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
            epoch_match = EPOCH_RE.search(line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
            best_match = BEST_RE.search(line)
            if not best_match or current_epoch is None:
                continue
            metrics = {
                "mINP": float(best_match.group(1)),
                "mAP": float(best_match.group(2)),
                "Rank-1": float(best_match.group(3)),
            }
            checkpoint = log_path.parent.parent / "models" / f"model_Fusion_{current_epoch}.pth"
            if not checkpoint.is_file():
                continue
            candidates.append(
                {
                    "checkpoint": str(checkpoint.resolve()),
                    "source_log": str(log_path.resolve()),
                    "best_epoch": current_epoch,
                    "metrics": metrics,
                    "metric_error": metric_error(metrics, expected),
                    "metrics_match_reference": metrics_match_reference(metrics, expected, tolerance),
                }
            )

    if not candidates:
        raise FileNotFoundError(f"No log-backed E4 checkpoint found below {output_root}")
    candidates.sort(
        key=lambda item: (
            not item["metrics_match_reference"],
            item["metric_error"],
            -float(item["metrics"]["Rank-1"]),
        )
    )
    selected = dict(candidates[0])
    if not selected["metrics_match_reference"]:
        raise RuntimeError(
            "E4 checkpoint candidates do not match the configured reference within "
            f"{tolerance}: {json.dumps(candidates, ensure_ascii=False)}"
        )
    checkpoint_path = Path(selected["checkpoint"])
    selected.update(
        {
            "sha256": sha256_file(checkpoint_path),
            "size_bytes": checkpoint_path.stat().st_size,
            "candidate_count": len(candidates),
            "all_candidates": candidates,
            "config": str(config_path),
        }
    )
    return selected


def nearby_checkpoints(checkpoint: Path, offsets: Sequence[int] = (-4, -2, 0, 2, 4)) -> List[Dict[str, Any]]:
    checkpoint = Path(checkpoint)
    match = re.search(r"_(\d+)\.pth$", checkpoint.name)
    if not match:
        raise ValueError(f"Checkpoint name has no epoch suffix: {checkpoint.name}")
    best_epoch = int(match.group(1))
    prefix = checkpoint.name[: match.start(1)]
    result = []
    for offset in offsets:
        epoch = best_epoch + int(offset)
        if epoch < 0:
            continue
        candidate = checkpoint.with_name(f"{prefix}{epoch}.pth")
        result.append({"epoch": epoch, "offset": int(offset), "path": str(candidate), "exists": candidate.is_file()})
    return result


def protocol_guard(config: Mapping[str, Any]) -> None:
    required = {
        "dataset": "sysu",
        "test_mode": "all",
        "gall_mode": "single",
        "test_modality": "Fusion",
    }
    mismatches = {
        key: {"actual": config.get(key), "expected": expected}
        for key, expected in required.items()
        if config.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"SYSU all-search single-shot protocol violation: {mismatches}")
    if int(config.get("gallery_trials", 10)) != 10:
        raise ValueError("SYSU results must average exactly 10 gallery trials")


def parse_gpu_csv(raw: str) -> List[Dict[str, Any]]:
    states = []
    for row in csv.reader(raw.splitlines()):
        if not row or not any(item.strip() for item in row):
            continue
        if len(row) != 5:
            raise ValueError(f"Expected five nvidia-smi columns, got {row!r}")
        index, name, total, used, utilization = [item.strip() for item in row]
        states.append(
            {
                "index": int(index),
                "name": name,
                "memory_total_mib": int(float(total)),
                "memory_used_mib": int(float(used)),
                "utilization_pct": int(float(utilization)),
            }
        )
    return states


def query_gpu_states() -> List[Dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    states = parse_gpu_csv(subprocess.check_output(command, text=True))
    uuid_rows = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        text=True,
    ).splitlines()
    uuid_to_index = {}
    for row in uuid_rows:
        index, uuid = [part.strip() for part in row.split(",", 1)]
        uuid_to_index[uuid] = int(index)
    process_counts = {int(state["index"]): 0 for state in states}
    process_query = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if process_query.returncode == 0:
        for row in process_query.stdout.splitlines():
            if not row.strip():
                continue
            uuid = row.split(",", 1)[0].strip()
            if uuid in uuid_to_index:
                index = uuid_to_index[uuid]
                process_counts[index] = process_counts.get(index, 0) + 1
    for state in states:
        state["compute_process_count"] = process_counts.get(int(state["index"]), 0)
    return states


def idle_gpu_ids(
    states: Iterable[Mapping[str, Any]],
    memory_limit_mib: int = GPU_IDLE_MEMORY_MIB,
    utilization_limit_pct: int = GPU_IDLE_UTIL_PCT,
) -> List[int]:
    return [
        int(state["index"])
        for state in states
        if int(state["memory_used_mib"]) < memory_limit_mib
        and int(state["utilization_pct"]) < utilization_limit_pct
        and int(state.get("compute_process_count", 0)) == 0
    ]


class GpuLease(AbstractContextManager):
    """Advisory per-GPU lock held for the lifetime of one foreground subprocess."""

    def __init__(self, gpu_id: int, lock_root: Path = REPORT_ROOT / "gpu_locks"):
        self.gpu_id = int(gpu_id)
        self.lock_root = Path(lock_root)
        self.handle = None

    def __enter__(self):
        if fcntl is None:
            raise RuntimeError("GpuLease requires fcntl on the Linux experiment server")
        self.lock_root.mkdir(parents=True, exist_ok=True)
        lock_path = self.lock_root / f"gpu-{self.gpu_id}.lock"
        self.handle = lock_path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.handle.close()
            self.handle = None
            raise RuntimeError(f"GPU {self.gpu_id} already has a metric-boost lease")
        self.handle.seek(0)
        self.handle.truncate()
        self.handle.write(json.dumps({"pid": os.getpid(), "gpu": self.gpu_id, "acquired_at": utc_now()}))
        self.handle.flush()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.handle is not None:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()
            self.handle = None
        return False


def experiment_dir(experiment_id: str) -> Path:
    return REPORT_ROOT / "runs" / experiment_id


def experiment_status_path(experiment_id: str) -> Path:
    return experiment_dir(experiment_id) / "status.json"


def succeeded(experiment_id: str) -> bool:
    payload = read_json(experiment_status_path(experiment_id), {})
    return payload.get("status") == "succeeded"


def prepare_runtime_config(
    experiment_id: str,
    base_config_path: Path,
    overrides: Mapping[str, Any],
    gpu_id: Optional[int] = None,
) -> Path:
    payload = load_yaml(REPO_ROOT / "src/salt_vi/config/default.yaml")
    payload.update(load_yaml(base_config_path))
    payload.update(dict(overrides))
    if payload.get("dataset") == "sysu":
        payload["text_data_root"] = str(resolve_sysu_text_root(payload.get("text_data_root")))
    if payload.get("pretrain_choice") == "PMT_VIT":
        payload["pmt_pretrained"] = str(resolve_pmt_pretrained(payload.get("pmt_pretrained")))
    if gpu_id is not None:
        payload["CUDA_VISIBLE_DEVICES"] = str(int(gpu_id))
        payload["gpu_id"] = "0"
    protocol_guard(payload)
    run_dir = experiment_dir(experiment_id)
    runtime_path = run_dir / "runtime_config.yaml"
    atomic_write_yaml(runtime_path, payload)
    return runtime_path


def base_status(experiment_id: str, phase: str, validity: str) -> Dict[str, Any]:
    return {
        "experiment": experiment_id,
        "phase": phase,
        "status": "pending",
        "validity": validity,
        "git_commit_sha": git_commit_sha(),
        "checkpoint": None,
        "best_epoch": None,
        "Rank-1": None,
        "mAP": None,
        "mINP": None,
        "delta_rank1_pp": None,
        "delta_map_pp": None,
        "delta_minp_pp": None,
        "gpu": None,
        "start_time": None,
        "end_time": None,
        "command": None,
        "runtime_config": None,
        "log_path": None,
        "error": None,
    }


def validate_experiment_manifest(eval_plan, train_plan, path: Path = EXPERIMENT_MANIFEST) -> Dict[str, Any]:
    manifest = load_yaml(path)
    expected_order = [f"EVAL-{index}" for index in range(7)] + [f"TRAIN-{index}" for index in range(1, 9)]
    if manifest.get("order") != expected_order:
        raise ValueError(f"Experiment manifest order differs from required order: {manifest.get('order')}")
    counts = manifest.get("expanded_counts", {})
    if int(counts.get("evaluation", -1)) != len(eval_plan):
        raise ValueError(f"Manifest evaluation count {counts.get('evaluation')} != generated {len(eval_plan)}")
    if int(counts.get("training", -1)) != len(train_plan):
        raise ValueError(f"Manifest training count {counts.get('training')} != generated {len(train_plan)}")
    identifiers = [item["id"] for item in list(eval_plan) + list(train_plan)]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Experiment IDs are not unique")
    protocol = manifest.get("protocol", {})
    required_protocol = {"dataset": "SYSU-MM01", "test_mode": "all", "gall_mode": "single", "gallery_trials": 10}
    mismatches = {key: protocol.get(key) for key, value in required_protocol.items() if protocol.get(key) != value}
    if mismatches:
        raise ValueError(f"Manifest protocol mismatch: {mismatches}")
    gpu_policy = manifest.get("gpu_policy", {})
    if gpu_policy.get("max_jobs_per_gpu") != 1 or not gpu_policy.get("foreground_only"):
        raise ValueError("Manifest must enforce one foreground job per GPU")
    return manifest
