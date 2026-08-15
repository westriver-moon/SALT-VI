#!/usr/bin/env python3
"""Run Stage-B from the best PASD RN50 Stage-A checkpoint."""

import json
import math
import os
from pathlib import Path
import subprocess
import sys


EXPERIMENT_ID = "SALTVI-STAGEB-RN50-PASD-R-TEXT-VISUAL-30-20260815"
TOTAL_EPOCHS = 30
CHECKPOINT = Path(
    "/home/lab929/ybj/SALT-VI/checkpoints/stage_a/experiments/"
    "stage_a_tvilfm_rn50_pasd_direct_20260813/best/model_IR_epoch_115.pth"
)


def required_env(name):
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError("missing required environment variable: {}".format(name))
    return value


def atomic_write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.{}".format(os.getpid()))
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


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
                epoch = int(event.get("epoch", -1))
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
                    "best_epoch": float(epoch),
                },
            }
            if best is None or rank1 > best[0]:
                best = (rank1, candidate)
    if best is None:
        raise RuntimeError("no finite eval_epoch metrics found in {}".format(events_path))
    return best[1]


def load_and_validate_config(repo_root, config_path):
    src_root = repo_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

    from salt_vi.config.validation import validate_runtime_config
    from salt_vi.utils.utils import load_train_configs

    config = validate_runtime_config(load_train_configs(str(config_path)))
    expected = {
        "pretrain_choice": "RN50_ORI",
        "prj_output_dim": 2048,
        "training_mode": "RGB_IR_Text",
        "joint_mode": "uni",
        "captioner_name": "Blip",
        "fusion_way": "parameter_add",
        "pa": 0.5,
        "Fix_Visual": True,
        "optimizer": "AdamW",
        "lr_txt": 7.5e-06,
        "total_train_epoch": TOTAL_EPOCHS,
        "batch_size": 32,
        "num_pos": 4,
        "retrieval_backend": "legacy",
    }
    mismatches = {
        name: {"expected": value, "actual": getattr(config, name, None)}
        for name, value in expected.items()
        if getattr(config, name, None) != value
    }
    if set(config.sysu_sr_modalities) != {"rgb", "ir"}:
        mismatches["sysu_sr_modalities"] = {
            "expected": ["rgb", "ir"],
            "actual": list(config.sysu_sr_modalities),
        }
    if mismatches:
        raise RuntimeError("invalid experiment configuration: {}".format(mismatches))

    required_paths = {
        "training_weight_init": Path(config.training_weight_init),
        "sysu_data_path": Path(config.sysu_data_path),
        "sysu_sr_data_root": Path(config.sysu_sr_data_root),
        "sysu_sr_view_manifest": Path(config.sysu_sr_view_manifest),
        "text_data_root": Path(config.text_data_root),
    }
    missing = [name for name, path in required_paths.items() if not path.exists()]
    if missing:
        raise RuntimeError("missing Stage-B inputs: {}".format(", ".join(missing)))

    return config, {
        "checkpoint": str(Path(config.training_weight_init).resolve()),
        "planned_epochs": int(config.total_train_epoch),
        "batch_size_per_modality": int(config.batch_size),
        "identities_per_modality_batch": int(config.batch_size) // int(config.num_pos),
        "required_paths": {name: str(path) for name, path in required_paths.items()},
    }


def validate_checkpoint_compatibility(config):
    import torch
    from salt_vi.engine import build_model
    from salt_vi.entrypoints.train import _load_compatible_state_dict

    # The canonical training entrypoint derives this before constructing the
    # model. Supply a non-writing placeholder when validating in isolation.
    config.output_path = str(Path("/tmp") / "saltvi-stageb-rn50-compat")
    model = build_model(config)
    _load_compatible_state_dict(model, str(CHECKPOINT), torch.device("cpu"))
    return {
        "checkpoint_compatible": True,
        "model_parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
    }


def main():
    repo_root = Path(__file__).resolve().parents[2]
    config_path = (
        repo_root
        / "configs"
        / "experiments"
        / "stage_b_rn50_pasd_r_text_visual_30"
        / "train.yaml"
    )
    config, validation = load_and_validate_config(repo_root, config_path)
    if "--validate-only" in sys.argv[1:]:
        validation.update(validate_checkpoint_compatibility(config))
        print(json.dumps(validation, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    output_dir = Path(required_env("AR2_OUTPUT_DIR")).resolve()
    results_dir = Path(required_env("AR2_RESULTS_DIR")).resolve()
    gpu_id = required_env("AR2_GPU_ID")
    events_path = output_dir / "events.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        str(repo_root / "scripts" / "train.py"),
        "--config_select",
        str(config_path),
        "--set",
        "CUDA_VISIBLE_DEVICES='{}'".format(gpu_id),
        "--set",
        "gpu_id='0'",
    ]
    completed = subprocess.run(command, cwd=str(repo_root), check=False)
    if completed.returncode != 0:
        return completed.returncode

    payload = best_eval_event(events_path)
    payload["metrics"].update(
        {
            "selected_gpu": float(gpu_id),
            "planned_epochs": float(TOTAL_EPOCHS),
            "stage_a_epoch": 115.0,
            "batch_size_per_modality": 32.0,
            "effective_cross_modal_images": 64.0,
            "identities_per_modality_batch": 8.0,
        }
    )
    atomic_write_json(results_dir / "metrics.json", payload)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print("RN50 Stage-B experiment failed: {}".format(exc), file=sys.stderr)
        sys.exit(1)
