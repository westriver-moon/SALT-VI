import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
import re

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = Path("/home/cgv841/ybj/SALT-VI/experiments/stageb-fusion-ablation")
CHECK_INTERVAL_SECONDS = 60
GPU_IDLE_MEMORY_MIB = 2000
GPU_IDLE_UTIL_PCT = 20
EPOCH_RE = re.compile(r"Time:\s*([0-9:-]+);\s*Epoch:\s*(\d+);")
BEST_RE = re.compile(
    r"Best Fusion_RGB mINP:\s*([0-9.]+), Best mAP:\s*([0-9.]+), Best Rank1:\s*([0-9.]+)"
)
TIME_RE = re.compile(r"Time:\s*([0-9:-]+);")

EXPERIMENTS = [
    {
        "experiment_name": "e3_no_sff_norm_add",
        "config_path": "configs/stage_b/vit_source_core_sysu_no_sff_norm_add.yaml",
    },
    {
        "experiment_name": "e4_no_sff_parameter_add_pa05",
        "config_path": "configs/stage_b/vit_source_core_sysu_no_sff_parameter_add_pa05.yaml",
    },
    {
        "experiment_name": "e5_no_sff_cross_attention",
        "config_path": "configs/stage_b/vit_source_core_sysu_no_sff_cross_attention.yaml",
    },
]


def now_iso():
    return datetime.now().isoformat(timespec="seconds")


def parse_log_time(value):
    return datetime.strptime(value, "%Y-%m-%d-%H:%M:%S")


def format_duration(seconds):
    seconds = int(max(0, seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def read_yaml(path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_yaml(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


def read_json(path):
    if not path.is_file():
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(tmp_path, path)


def run_command(command):
    return subprocess.run(command, check=True, capture_output=True, text=True)


def is_pid_running(pid):
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except (ProcessLookupError, ValueError, OSError):
        return False
    return True


def query_gpu_states():
    result = run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    gpus = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        index, memory_used, util = [item.strip() for item in line.split(",")]
        gpus.append(
            {
                "index": index,
                "memory_used": int(float(memory_used)),
                "utilization": int(float(util)),
            }
        )
    return gpus


def idle_gpu_ids(excluded_gpu_ids):
    idle = []
    for gpu in query_gpu_states():
        if gpu["index"] in excluded_gpu_ids:
            continue
        if gpu["memory_used"] < GPU_IDLE_MEMORY_MIB and gpu["utilization"] < GPU_IDLE_UTIL_PCT:
            idle.append(gpu["index"])
    return idle


def exp_dir(experiment_name):
    return EXPERIMENT_ROOT / experiment_name


def status_file(experiment_name):
    return exp_dir(experiment_name) / "status.json"


def launcher_log_file(experiment_name):
    return exp_dir(experiment_name) / "launcher.log"


def return_code_file(experiment_name):
    return exp_dir(experiment_name) / "return_code.txt"


def prepared_config_file(experiment_name):
    return exp_dir(experiment_name) / "config_runtime.yaml"


def ensure_pending_status(experiment):
    exp_dir(experiment["experiment_name"]).mkdir(parents=True, exist_ok=True)
    path = status_file(experiment["experiment_name"])
    if path.is_file():
        return
    payload = {
        "experiment_name": experiment["experiment_name"],
        "config_path": experiment["config_path"],
        "prepared_config_path": None,
        "gpu_id": None,
        "pid": None,
        "start_time": None,
        "end_time": None,
        "status": "pending",
        "return_code": None,
        "best_epoch": None,
        "best_rank1": None,
        "best_map": None,
        "best_minp": None,
        "best_model_path": None,
        "training_time": None,
        "launcher_log": str(launcher_log_file(experiment["experiment_name"])),
    }
    write_json(path, payload)


def prepare_runtime_config(experiment, gpu_id):
    source_path = REPO_ROOT / experiment["config_path"]
    payload = read_yaml(source_path)
    payload["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    payload["gpu_id"] = "0"
    path = prepared_config_file(experiment["experiment_name"])
    write_yaml(path, payload)
    return path, payload


def parse_log_metrics(log_path, output_root):
    best_epoch = None
    best_rank1 = None
    best_map = None
    best_minp = None
    current_epoch = None
    timestamps = []
    pending_new_best = False

    if log_path.is_file():
        for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            epoch_match = EPOCH_RE.search(line)
            if epoch_match:
                timestamps.append(parse_log_time(epoch_match.group(1)))
                current_epoch = int(epoch_match.group(2))
                continue

            time_match = TIME_RE.search(line)
            if time_match:
                timestamps.append(parse_log_time(time_match.group(1)))

            if "New Best" in line:
                pending_new_best = True
                continue

            best_match = BEST_RE.search(line)
            if best_match:
                if pending_new_best or best_epoch is None:
                    best_minp = float(best_match.group(1))
                    best_map = float(best_match.group(2))
                    best_rank1 = float(best_match.group(3))
                    best_epoch = current_epoch
                pending_new_best = False

    best_model_path = None
    if output_root and output_root.exists() and best_epoch is not None:
        matches = sorted(output_root.rglob(f"model_Fusion_{best_epoch}.pth"))
        if matches:
            best_model_path = str(matches[-1])

    training_time = None
    if len(timestamps) >= 2:
        training_time = format_duration((max(timestamps) - min(timestamps)).total_seconds())

    return {
        "best_epoch": best_epoch,
        "best_rank1": best_rank1,
        "best_map": best_map,
        "best_minp": best_minp,
        "best_model_path": best_model_path,
        "training_time": training_time,
    }


def finalize_status(experiment_name, payload, state, return_code):
    output_root = None
    prepared_config_path = payload.get("prepared_config_path")
    if prepared_config_path and Path(prepared_config_path).is_file():
        output_root = Path(read_yaml(prepared_config_path).get("output_path", ""))

    parsed = parse_log_metrics(launcher_log_file(experiment_name), output_root)
    payload["status"] = state
    payload["return_code"] = return_code
    payload["end_time"] = now_iso()
    payload["best_epoch"] = parsed["best_epoch"]
    payload["best_rank1"] = parsed["best_rank1"]
    payload["best_map"] = parsed["best_map"]
    payload["best_minp"] = parsed["best_minp"]
    payload["best_model_path"] = parsed["best_model_path"]
    payload["training_time"] = parsed["training_time"]
    write_json(status_file(experiment_name), payload)


def refresh_running_status(experiment):
    name = experiment["experiment_name"]
    payload = read_json(status_file(name))
    if payload is None:
        ensure_pending_status(experiment)
        payload = read_json(status_file(name))

    if payload["status"] != "running":
        return payload
    if is_pid_running(payload.get("pid")):
        return payload

    rc_path = return_code_file(name)
    if rc_path.is_file():
        raw = rc_path.read_text(encoding="utf-8").strip()
        return_code = int(raw) if raw else None
        finalize_status(name, payload, "succeeded" if return_code == 0 else "failed", return_code)
    else:
        finalize_status(name, payload, "stale_failed", None)
    return read_json(status_file(name))


def launch_experiment(experiment, gpu_id):
    name = experiment["experiment_name"]
    exp_dir(name).mkdir(parents=True, exist_ok=True)
    prepared_config, config_payload = prepare_runtime_config(experiment, gpu_id)

    rc_path = return_code_file(name)
    if rc_path.exists():
        rc_path.unlink()

    log_path = launcher_log_file(name)
    log_handle = open(log_path, "a", encoding="utf-8")
    command = (
        f"cd {shlex.quote(str(REPO_ROOT))} && "
        f"export CUDA_VISIBLE_DEVICES={shlex.quote(str(gpu_id))} && "
        f"{shlex.quote(sys.executable)} scripts/train.py --config_select {shlex.quote(str(prepared_config))}; "
        f"rc=$?; printf '%s' \"$rc\" > {shlex.quote(str(rc_path))}; exit $rc"
    )
    process = subprocess.Popen(
        ["bash", "-lc", command],
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    payload = {
        "experiment_name": name,
        "config_path": experiment["config_path"],
        "prepared_config_path": str(prepared_config),
        "gpu_id": str(gpu_id),
        "pid": process.pid,
        "start_time": now_iso(),
        "end_time": None,
        "status": "running",
        "return_code": None,
        "best_epoch": None,
        "best_rank1": None,
        "best_map": None,
        "best_minp": None,
        "best_model_path": None,
        "training_time": None,
        "launcher_log": str(log_path),
        "output_root": config_payload.get("output_path"),
    }
    write_json(status_file(name), payload)
    print(f"[launch] {name} gpu={gpu_id} pid={process.pid} config={prepared_config}", flush=True)
    return process, log_handle


def summarize_runs():
    summary_script = REPO_ROOT / "scripts" / "summarize_stage_b_fusion_ablation.py"
    print(f"[summary] {summary_script}", flush=True)
    subprocess.run([sys.executable, str(summary_script)], cwd=str(REPO_ROOT), check=True)


def main():
    EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)
    for experiment in EXPERIMENTS:
        ensure_pending_status(experiment)

    launched = {}
    log_handles = {}
    failure_seen = False

    try:
        while True:
            pending = []
            running_gpu_ids = set()

            for experiment in EXPERIMENTS:
                payload = refresh_running_status(experiment)
                state = payload["status"]
                if state in ("failed", "stale_failed"):
                    failure_seen = True
                if state == "running" and payload.get("gpu_id") is not None:
                    running_gpu_ids.add(str(payload["gpu_id"]))
                elif state == "pending":
                    pending.append(experiment)

            if not failure_seen and pending:
                for gpu_id in idle_gpu_ids(running_gpu_ids):
                    if not pending:
                        break
                    experiment = pending.pop(0)
                    process, log_handle = launch_experiment(experiment, gpu_id)
                    launched[experiment["experiment_name"]] = process
                    log_handles[experiment["experiment_name"]] = log_handle
                    running_gpu_ids.add(str(gpu_id))

            if failure_seen and pending:
                print("[halt] detected failed experiment; pending launches are blocked.", flush=True)

            for name, process in list(launched.items()):
                return_code = process.poll()
                if return_code is None:
                    continue
                payload = read_json(status_file(name))
                if payload and payload["status"] == "running":
                    finalize_status(name, payload, "succeeded" if return_code == 0 else "failed", return_code)
                    if return_code != 0:
                        failure_seen = True
                handle = log_handles.pop(name, None)
                if handle is not None:
                    handle.close()
                launched.pop(name, None)

            snapshot = {}
            terminal = True
            for experiment in EXPERIMENTS:
                payload = read_json(status_file(experiment["experiment_name"]))
                snapshot[experiment["experiment_name"]] = payload["status"]
                state = payload["status"]
                if state == "running":
                    terminal = False
                if state == "pending" and not failure_seen:
                    terminal = False
            print(f"[status] {snapshot}", flush=True)

            if terminal:
                break

            time.sleep(CHECK_INTERVAL_SECONDS)

        summarize_runs()
        print("[done] stage_b fusion ablation complete.", flush=True)
    finally:
        for handle in log_handles.values():
            handle.close()


if __name__ == "__main__":
    main()
