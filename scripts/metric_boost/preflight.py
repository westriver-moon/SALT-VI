#!/usr/bin/env python
"""Read-only preflight for the SYSU metric-boost program.

The command deliberately treats a busy server as a valid preparation state:
``preparation_ready`` may be true while ``launch_ready`` is false. It never
starts, stops, signals, or waits for a GPU process.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import (
    E4_CONFIG_PATH,
    EXPECTED_E4,
    REPORT_ROOT,
    REPO_ROOT,
    atomic_write_json,
    atomic_write_text,
    git_commit_sha,
    idle_gpu_ids,
    load_yaml,
    nearby_checkpoints,
    protocol_guard,
    query_gpu_states,
    resolve_e4_checkpoint,
    resolve_pmt_pretrained,
    resolve_sysu_text_root,
    utc_now,
)


DEFAULT_CONFIG_PATH = REPO_ROOT / "src/salt_vi/config/default.yaml"


def merged_config(config_path: Path) -> Dict[str, Any]:
    payload = load_yaml(DEFAULT_CONFIG_PATH)
    payload.update(load_yaml(config_path))
    payload.setdefault("gallery_trials", 10)
    return payload


def _array_metadata(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "exists": False}
    value = np.load(str(path), mmap_mode="r", allow_pickle=False)
    return {
        "path": str(path.resolve()),
        "exists": True,
        "shape": list(value.shape),
        "dtype": str(value.dtype),
    }


def inspect_sysu_assets(config: Mapping[str, Any]) -> Dict[str, Any]:
    data_root = Path(str(config["sysu_data_path"])).resolve()
    text_root = resolve_sysu_text_root(config.get("text_data_root"))
    arrays = {
        "train_rgb_images": data_root / "train_rgb_resized_img.npy",
        "train_rgb_labels": data_root / "train_rgb_resized_label.npy",
        "train_ir_images": data_root / "train_ir_resized_img.npy",
        "train_ir_labels": data_root / "train_ir_resized_label.npy",
        "train_rgb_text": text_root / "Blip_RGB/train_text_Blip_RGB.npy",
        "train_rgb_text_labels": text_root / "Blip_RGB/train_text_label_Blip_RGB.npy",
        "train_rgb_llm_text": text_root / "Blip_RGB/train_llm_text_Blip_RGB.npy",
        "train_ir_text": text_root / "Blip_IR/train_text_Blip_IR.npy",
        "train_ir_text_labels": text_root / "Blip_IR/train_text_label_Blip_IR.npy",
        "train_ir_llm_text": text_root / "Blip_IR/train_llm_text_Blip_IR.npy",
    }
    metadata = {name: _array_metadata(path) for name, path in arrays.items()}
    required_json = [
        text_root / "Blip_RGB/id_caption_map_Blip_RGB.json",
        text_root / "Blip_IR/id_caption_map_Blip_IR.json",
        text_root / "Blip_RGB/caption_dict_Blip_RGB.json",
        text_root / "Blip_IR/caption_dict_Blip_IR.json",
    ]
    json_assets = [{"path": str(path), "exists": path.is_file()} for path in required_json]

    errors = []
    missing = [name for name, item in metadata.items() if not item["exists"]]
    if missing:
        errors.append(f"Missing SYSU array assets: {missing}")
    missing_json = [item["path"] for item in json_assets if not item["exists"]]
    if missing_json:
        errors.append(f"Missing SYSU text JSON assets: {missing_json}")
    if not missing:
        pairs = [
            ("train_rgb_images", "train_rgb_labels"),
            ("train_ir_images", "train_ir_labels"),
            ("train_rgb_images", "train_rgb_text"),
            ("train_rgb_images", "train_rgb_text_labels"),
            ("train_rgb_images", "train_rgb_llm_text"),
            ("train_ir_images", "train_ir_text"),
            ("train_ir_images", "train_ir_text_labels"),
            ("train_ir_images", "train_ir_llm_text"),
        ]
        for left, right in pairs:
            left_count = int(metadata[left]["shape"][0])
            right_count = int(metadata[right]["shape"][0])
            if left_count != right_count:
                errors.append(f"Asset alignment mismatch: {left}={left_count}, {right}={right_count}")
    return {
        "data_root": str(data_root),
        "text_root": str(text_root),
        "arrays": metadata,
        "json_assets": json_assets,
        "aligned": not errors,
        "errors": errors,
    }


def inspect_checkpoint_loadability(checkpoint_path: Path) -> Dict[str, Any]:
    import torch

    payload = torch.load(str(checkpoint_path), map_location="cpu")
    if isinstance(payload, dict) and "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
        wrapper = "model_state_dict"
    elif isinstance(payload, dict) and "state_dict" in payload:
        state_dict = payload["state_dict"]
        wrapper = "state_dict"
    else:
        state_dict = payload
        wrapper = "raw"
    if not isinstance(state_dict, dict):
        raise TypeError(f"Checkpoint does not contain a state dictionary: {type(state_dict).__name__}")
    tensor_count = sum(1 for value in state_dict.values() if hasattr(value, "shape"))
    sample_shapes = {
        key: list(value.shape)
        for key, value in list(state_dict.items())[:20]
        if hasattr(value, "shape")
    }
    return {
        "loadable_on_cpu": True,
        "wrapper": wrapper,
        "key_count": len(state_dict),
        "tensor_count": tensor_count,
        "sample_shapes": sample_shapes,
    }


def git_profile() -> Dict[str, Any]:
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=str(REPO_ROOT), text=True).strip()
    top_level = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=str(REPO_ROOT), text=True).strip()
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=str(REPO_ROOT), text=True).splitlines()
    owned_fragments = ("autoresearch-results", "reports/metric_boost", "reports/metric_boost")
    tracked_dirty = [
        line for line in status if not any(fragment in line for fragment in owned_fragments)
    ]
    return {
        "top_level": top_level,
        "project_root": str(REPO_ROOT),
        "branch": branch,
        "commit": git_commit_sha(),
        "tracked_or_untracked_changes": tracked_dirty,
        "isolated_metric_boost_branch": branch == "codex/metric-boost-prep",
    }


def render_markdown(payload: Mapping[str, Any]) -> str:
    checkpoint = payload["e4_checkpoint"]
    gpu_lines = [
        f"- GPU {item['index']} {item['name']}: {item['memory_used_mib']} MiB, {item['utilization_pct']}%"
        for item in payload["gpu"]["states"]
    ]
    return "\n".join(
        [
            "# Metric Boost Preflight",
            "",
            f"- Preparation ready: **{payload['preparation_ready']}**",
            f"- Launch ready: **{payload['launch_ready']}**",
            f"- Launch blocked reason: {payload['launch_blocked_reason'] or 'none'}",
            f"- Git: `{payload['git']['branch']}@{payload['git']['commit']}`",
            f"- E4 epoch: `{checkpoint['best_epoch']}`",
            f"- E4 checkpoint: `{checkpoint['checkpoint']}`",
            f"- E4 SHA-256: `{checkpoint['sha256']}`",
            f"- E4 metrics: Rank-1 `{checkpoint['metrics']['Rank-1']:.5f}`, mAP `{checkpoint['metrics']['mAP']:.5f}`, mINP `{checkpoint['metrics']['mINP']:.5f}`",
            f"- SYSU assets aligned: **{payload['sysu_assets']['aligned']}**",
            "",
            "## GPU snapshot (read-only)",
            "",
            *gpu_lines,
            "",
            "No process was started, stopped, signalled, or waited on by this preflight.",
            "",
        ]
    )


def run_preflight(check_checkpoint: bool = True) -> Dict[str, Any]:
    config = merged_config(E4_CONFIG_PATH)
    config["pmt_pretrained"] = str(resolve_pmt_pretrained(config.get("pmt_pretrained")))
    protocol_guard(config)
    checkpoint = resolve_e4_checkpoint()
    checkpoint["nearby_checkpoints"] = nearby_checkpoints(Path(checkpoint["checkpoint"]))
    if check_checkpoint:
        checkpoint["load_audit"] = inspect_checkpoint_loadability(Path(checkpoint["checkpoint"]))
    else:
        checkpoint["load_audit"] = {"loadable_on_cpu": None, "skipped": True}
    sysu_assets = inspect_sysu_assets(config)
    gpu_states = query_gpu_states()
    idle = idle_gpu_ids(gpu_states)
    git = git_profile()
    errors = list(sysu_assets["errors"])
    if not git["isolated_metric_boost_branch"]:
        errors.append(f"Expected isolated branch codex/metric-boost-prep, got {git['branch']}")
    if git["tracked_or_untracked_changes"]:
        errors.append(f"Worktree has non-autoresearch changes: {git['tracked_or_untracked_changes']}")
    preparation_ready = not errors
    launch_ready = preparation_ready and bool(idle)
    return {
        "generated_at": utc_now(),
        "preparation_ready": preparation_ready,
        "launch_ready": launch_ready,
        "launch_blocked_reason": None if launch_ready else (
            "; ".join(errors) if errors else "No GPU meets memory<2000 MiB and utilization<20%"
        ),
        "errors": errors,
        "expected_e4": EXPECTED_E4,
        "protocol": {
            "dataset": "SYSU-MM01",
            "search": "all-search",
            "gallery": "single-shot",
            "gallery_trials": 10,
            "test_labels_for_tuning": False,
        },
        "e4_checkpoint": checkpoint,
        "pmt_pretrained": {
            "path": config["pmt_pretrained"],
            "exists": Path(config["pmt_pretrained"]).is_file(),
            "size_bytes": Path(config["pmt_pretrained"]).stat().st_size,
        },
        "sysu_assets": sysu_assets,
        "gpu": {"states": gpu_states, "idle_gpu_ids": idle},
        "git": git,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "executable": sys.executable,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=REPORT_ROOT)
    parser.add_argument("--skip-checkpoint-load", action="store_true")
    parser.add_argument("--require-idle-gpu", action="store_true")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    payload = run_preflight(check_checkpoint=not args.skip_checkpoint_load)
    if not args.no_write:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(args.output_dir / "preflight.json", payload)
        atomic_write_text(args.output_dir / "preflight.md", render_markdown(payload))
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    if not payload["preparation_ready"]:
        raise SystemExit(2)
    if args.require_idle_gpu and not payload["launch_ready"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
