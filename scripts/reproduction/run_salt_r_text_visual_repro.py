#!/usr/bin/env python3
"""Run the isolated SALT_R_TEXT_VISUAL reproduction and publish schema-v2 metrics."""

import json
import math
import os
from pathlib import Path
import subprocess
import sys


def required_env(name):
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError("missing required environment variable: {}".format(name))
    return value


def best_eval_event(events_path):
    best = None
    with events_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    "invalid JSONL at {}:{}: {}".format(events_path, line_number, exc)
                )
            if event.get("event_type") != "eval_epoch":
                continue
            metrics = event.get("metrics") or {}
            try:
                rank1 = float(metrics["Rank-1"])
                map_value = float(metrics["mAP"])
                minp = float(metrics["mINP"])
            except (KeyError, TypeError, ValueError):
                continue
            if not all(math.isfinite(value) for value in (rank1, map_value, minp)):
                continue
            candidate = {
                "primary_metric": rank1,
                "metrics": {
                    "Rank-1": rank1,
                    "mAP": map_value,
                    "mINP": minp,
                    "best_epoch": float(event.get("epoch", -1)),
                },
            }
            if best is None or rank1 > best[0]:
                best = (rank1, candidate)
    if best is None:
        raise RuntimeError("no finite eval_epoch metrics found in {}".format(events_path))
    return best[1]


def atomic_write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.{}".format(os.getpid()))
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


def validate_inputs(repo_root, config_path):
    src_root = repo_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    from salt_vi.utils.utils import load_train_configs

    config = load_train_configs(str(config_path))
    required_paths = {
        "config": config_path,
        "training_weight_init": Path(config.training_weight_init),
        "pmt_pretrained": Path(config.pmt_pretrained),
        "sysu_data_path": Path(config.sysu_data_path),
        "sysu_sr_data_root": Path(config.sysu_sr_data_root),
        "text_data_root": Path(config.text_data_root),
    }
    missing = [name for name, path in required_paths.items() if not path.exists()]
    if missing:
        raise RuntimeError("missing reproduction inputs: {}".format(", ".join(missing)))
    return {
        "metric_experiment_id": config.metric_experiment_id,
        "required_paths": {name: str(path) for name, path in required_paths.items()},
    }


def main():
    repo_root = Path(__file__).resolve().parents[2]
    config_path = repo_root / "configs" / "reproduction" / "salt_r_text_visual_84_gpu.yaml"
    if "--validate-only" in sys.argv[1:]:
        print(json.dumps(validate_inputs(repo_root, config_path), ensure_ascii=False, indent=2))
        return 0

    output_dir = Path(required_env("AR2_OUTPUT_DIR")).resolve()
    results_dir = Path(required_env("AR2_RESULTS_DIR")).resolve()
    gpu_id = required_env("AR2_GPU_ID")
    events_path = output_dir / "events.jsonl"
    model_output_root = output_dir / "model_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        str(repo_root / "scripts" / "train.py"),
        "--config_select",
        str(config_path),
        "--set",
        "CUDA_VISIBLE_DEVICES={}".format(gpu_id),
        "--set",
        "gpu_id=0",
        "--set",
        "output_path={}/".format(model_output_root),
        "--set",
        "metric_events_path={}".format(events_path),
        "--set",
        "metric_experiment_id=SALTVI-R-TEXT-VISUAL-84-REPRO",
    ]
    completed = subprocess.run(command, cwd=str(repo_root), check=False)
    if completed.returncode != 0:
        return completed.returncode

    payload = best_eval_event(events_path)
    payload["metrics"]["selected_gpu"] = float(gpu_id)
    atomic_write_json(results_dir / "metrics.json", payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
