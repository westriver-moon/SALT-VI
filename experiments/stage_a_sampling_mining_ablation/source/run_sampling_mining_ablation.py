#!/usr/bin/env python
import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml


CONFIGS = [
    "configs/stage_a/sampling_mining_ablation/s0_pk8x4_current_replace_hard.yaml",
    "configs/stage_a/sampling_mining_ablation/s1_pk8x4_auto_replace_hard.yaml",
    "configs/stage_a/sampling_mining_ablation/s2_pk16x2_auto_replace_hard.yaml",
    "configs/stage_a/sampling_mining_ablation/s3_pk4x8_auto_replace_hard.yaml",
    "configs/stage_a/sampling_mining_ablation/h1_pk8x4_auto_replace_wrt.yaml",
    "configs/stage_a/sampling_mining_ablation/h5_pk8x4_auto_replace_crossmodal_hard.yaml",
]
OUT_ROOT = Path("train_outputs/sampling_mining_ablation")
STATUS_PATH = OUT_ROOT / "status.json"


def now():
    return datetime.now().isoformat(timespec="seconds")


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.load(f, Loader=yaml.FullLoader) or {}


def exp_name(config_path):
    return Path(config_path).stem


def read_status():
    if STATUS_PATH.exists():
        return json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    return {}


def write_status(status):
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    tmp = STATUS_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(STATUS_PATH)


def max_logged_epoch(output_path):
    root = Path(output_path)
    epochs = []
    for log_path in list(root.rglob("log.log")) + list(root.rglob("train.log")):
        try:
            text = log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for match in re.finditer(r"Epoch:\s*(\d+)", text):
            epochs.append(int(match.group(1)))
    return max(epochs) if epochs else None


def is_done_by_log(config_path):
    cfg = load_yaml(config_path)
    total = int(cfg.get("total_train_epoch", 0))
    output_path = cfg.get("output_path", "")
    if not total or not output_path:
        return False
    epoch = max_logged_epoch(output_path)
    return epoch is not None and epoch + 1 >= total


def query_gpus(selected, max_mem, max_util):
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    output = subprocess.check_output(cmd, text=True)
    allowed = None
    if selected:
        allowed = {int(item) for item in selected.split(",") if item.strip()}
    gpus = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        idx = int(float(parts[0]))
        mem = int(float(parts[1]))
        util = 0 if parts[2].upper() == "N/A" else int(float(parts[2]))
        if allowed is not None and idx not in allowed:
            continue
        if mem < max_mem and util < max_util:
            gpus.append(idx)
    return gpus


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument("--gpus", default="")
    parser.add_argument("--rerun-failed", action="store_true")
    parser.add_argument("--max-mem", type=int, default=2000)
    parser.add_argument("--max-util", type=int, default=20)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()

    status = read_status()
    queue = []
    for cfg_path in CONFIGS:
        name = exp_name(cfg_path)
        entry = status.get(name, {"config": cfg_path, "status": "pending"})
        entry["config"] = cfg_path
        if entry.get("status") == "done" or is_done_by_log(cfg_path):
            entry["status"] = "done"
            status[name] = entry
            continue
        if entry.get("status") == "failed" and not args.rerun_failed:
            status[name] = entry
            continue
        entry.update({"status": "pending", "return_code": None})
        status[name] = entry
        queue.append(cfg_path)
    write_status(status)

    running = {}
    while queue or running:
        for name, proc_info in list(running.items()):
            proc = proc_info["proc"]
            rc = proc.poll()
            if rc is None:
                continue
            proc_info["log_handle"].close()
            entry = status[name]
            entry["end_time"] = now()
            entry["return_code"] = rc
            entry["status"] = "done" if rc == 0 else "failed"
            status[name] = entry
            del running[name]
            write_status(status)

        used_gpus = {info["gpu"] for info in running.values()}
        while queue and len(running) < args.max_parallel:
            free_gpus = [gpu for gpu in query_gpus(args.gpus, args.max_mem, args.max_util) if gpu not in used_gpus]
            if not free_gpus:
                break
            gpu = free_gpus[0]
            cfg_path = queue.pop(0)
            name = exp_name(cfg_path)
            log_dir = OUT_ROOT / name
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / "launcher.log"
            runtime_config = log_dir / "config_for_gpu.yaml"
            cfg_data = load_yaml(cfg_path)
            cfg_data["CUDA_VISIBLE_DEVICES"] = str(gpu)
            cfg_data["gpu_id"] = "0"
            runtime_config.write_text(yaml.safe_dump(cfg_data, sort_keys=False, allow_unicode=True), encoding="utf-8")
            log_handle = log_path.open("ab")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            cmd = [sys.executable, str(REPO_ROOT / "scripts" / "train.py"), "--config_select", str(runtime_config)]
            proc = subprocess.Popen(cmd, stdout=log_handle, stderr=subprocess.STDOUT, env=env)
            status[name] = {
                "config": cfg_path,
                "runtime_config": str(runtime_config),
                "status": "running",
                "gpu": gpu,
                "start_time": now(),
                "end_time": None,
                "return_code": None,
                "command": "CUDA_VISIBLE_DEVICES={} {}".format(gpu, " ".join(cmd)),
            }
            running[name] = {"proc": proc, "gpu": gpu, "log_handle": log_handle}
            used_gpus.add(gpu)
            write_status(status)

        if queue or running:
            time.sleep(args.poll_seconds)

    return 1 if any(item.get("status") == "failed" for item in status.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
