#!/usr/bin/env python3
"""Run a provenance-bound 20-step PMT-MBPatch SYSU SR preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn as nn
from torch.cuda import amp


PROJECT_ROOT = Path(__file__).resolve().parents[3]
GIT_ROOT = PROJECT_ROOT.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from salt_vi.baselines.vision_text.config import load_config
from salt_vi.baselines.vision_text.data.sampler import build_label_positions
from salt_vi.baselines.vision_text.engine.evaluator import evaluate_sysu
from salt_vi.baselines.vision_text.engine.trainer import (
    build_epoch_loader,
    build_train_datasets,
    compute_pmt_losses,
    make_optimizer,
    set_cosine_lr,
)
from salt_vi.baselines.vision_text.losses import DCL, MSEL, TripletLoss
from salt_vi.baselines.vision_text.model import build_pmt_model
from salt_vi.baselines.vision_text.utils.seed import set_seed


ALGORITHM_FILES = (
    "src/salt_vi/baselines/vision_text/config/defaults.py",
    "src/salt_vi/baselines/vision_text/data/dataset.py",
    "src/salt_vi/baselines/vision_text/data/transforms.py",
    "src/salt_vi/baselines/vision_text/engine/evaluator.py",
    "src/salt_vi/baselines/vision_text/engine/trainer.py",
    "src/salt_vi/baselines/vision_text/model/pmt_model.py",
    "src/salt_vi/baselines/vision_text/model/vision_transformer.py",
    "scripts/vision_text/super_resolution/preflight_pmt_sr.py",
    "src/salt_vi/utils/super_resolution/build_sysu_swinir_x2.py",
)


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload):
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def git(*args):
    return subprocess.check_output(["git", "-C", str(GIT_ROOT), *args], text=True).strip()


def assert_clean_source(expected_ref=None):
    head = git("rev-parse", "HEAD")
    if expected_ref and head != git("rev-parse", expected_ref):
        raise RuntimeError(f"HEAD {head} does not match {expected_ref}")
    if subprocess.run(["git", "-C", str(GIT_ROOT), "diff", "--quiet"]).returncode:
        raise RuntimeError("Tracked worktree changes are not allowed")
    if subprocess.run(["git", "-C", str(GIT_ROOT), "diff", "--cached", "--quiet"]).returncode:
        raise RuntimeError("Staged changes are not allowed")
    relevant_prefixes = ("src/salt_vi/baselines/vision_text/", "scripts/vision_text/", "src/salt_vi/utils/super_resolution/")
    untracked = git("ls-files", "--others", "--exclude-standard").splitlines()
    relevant = [path for path in untracked if path.startswith(relevant_prefixes)]
    if relevant:
        raise RuntimeError(f"Algorithm-relevant untracked files: {relevant}")
    return head


def plain(value):
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


def validate_config_contract(config):
    errors = []
    if int(config.seed) != 0:
        errors.append("seed must be 0")
    if (int(config.data.source_height), int(config.data.source_width)) != (256, 128):
        errors.append("common LR source must be 256x128")
    target = (int(config.data.height), int(config.data.width))
    if target not in {(256, 128), (512, 256)}:
        errors.append(f"unsupported target size {target}")
    if int(config.data.batch_size_per_modality) != 32 or int(config.data.num_pos) != 4:
        errors.append("PMT PK layout must remain 8x4 per modality")
    branches = plain(config.model.patch_embed.branches)
    expected_branches = [
        {"patch_size": [16, 16], "stride_size": [12, 12]},
        {"patch_size": [16, 8], "stride_size": [12, 6]},
    ]
    if branches != expected_branches or int(config.model.patch_embed.anchor_branch) != 0:
        errors.append("Stage-A MBPatch structure does not match the validated best config")
    if int(config.model.embed_dim) != 768 or not bool(config.model.gradient_checkpointing):
        errors.append("PMT embed_dim=768 and gradient checkpointing are required")
    if int(config.train.max_epoch) != 24 or str(config.train.optimizer) != "AdamW":
        errors.append("PMT 24-epoch AdamW recipe changed")
    if int(config.test.training_trials) != 10:
        errors.append("formal training evaluation must average 10 trials")
    modalities = list(config.data.get("sr_modalities", []) or [])
    if modalities not in ([], ["rgb"], ["rgb", "ir"]):
        errors.append(f"invalid SR modalities {modalities}")
    if target == (256, 128) and modalities:
        errors.append("A0 cannot use derived SR assets")
    if errors:
        raise ValueError("; ".join(errors))


def validate_sr_manifest(config):
    modalities = list(config.data.get("sr_modalities", []) or [])
    if not modalities:
        return None
    manifest_path = Path(config.data.sr_root) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": 4,
        "source_size_hw": [256, 128],
        "output_size_hw": [512, 256],
        "source_resampling": "bilinear",
    }
    mismatches = {
        key: {"expected": value, "observed": manifest.get(key)}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(f"SR manifest contract mismatch: {mismatches}")
    if set(manifest.get("modalities", [])) != {"rgb", "ir"}:
        raise ValueError("SR manifest must contain both modalities")
    return {"path": str(manifest_path.resolve()), "sha256": sha256_file(manifest_path)}


def provenance(config_path, config):
    pretrained = Path(config.model.pretrained)
    if not pretrained.is_absolute():
        pretrained = PROJECT_ROOT / pretrained
    sr_manifest = validate_sr_manifest(config)
    payload = {
        "schema_version": 1,
        "git_commit_sha": git("rev-parse", "HEAD"),
        "config_path": str(config_path.resolve()),
        "resolved_config_sha256": canonical_sha256(plain(config)),
        "algorithm_files": {
            path: sha256_file(GIT_ROOT / path)
            for path in ALGORITHM_FILES
        },
        "imagenet_pretrained": {
            "path": str(pretrained.resolve()),
            "sha256": sha256_file(pretrained),
        },
        "sr_manifest": sr_manifest,
    }
    payload["contract_sha256"] = canonical_sha256(payload)
    return payload


def serialize_metrics(average):
    values = np.asarray([average["rank1"], average["mAP"], average["mINP"]], dtype=float)
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
        raise ValueError(f"Invalid retrieval metrics: {values}")
    return {"Rank-1": float(values[0]), "mAP": float(values[1]), "mINP": float(values[2])}


def run_worker(config_path, output, steps, chunk_size, test_batch_size):
    config = load_config(config_path)
    validate_config_contract(config)
    # Bind the preflight to the immutable experiment config. Runtime-only memory
    # fallbacks are reported separately and must not change the experiment identity.
    run_provenance = provenance(config_path, config)
    config.model.backbone_chunk_size = int(chunk_size)
    config.test.batch_size = int(test_batch_size)
    set_seed(int(config.seed))
    device = torch.device("cuda:0")
    gray, _rgb, color_pos, thermal_pos = build_train_datasets(config, config.data.root)
    loader = build_epoch_loader(config, gray, color_pos, thermal_pos)
    model = build_pmt_model(config, num_classes=int(config.model.num_classes)).to(device)
    pretrained = Path(config.model.pretrained)
    if not pretrained.is_absolute():
        pretrained = PROJECT_ROOT / pretrained
    model.load_imagenet_pretrained(pretrained)
    optimizer = make_optimizer(config, model)
    set_cosine_lr(config, optimizer, 1)
    scaler = amp.GradScaler(enabled=bool(config.train.amp))
    criterion_id = nn.CrossEntropyLoss()
    criterion_tri = TripletLoss(float(config.train.triplet_margin), feat_norm="no")
    criterion_msel = MSEL(int(config.data.num_pos), feat_norm="no")
    criterion_dcl = DCL(int(config.data.num_pos), feat_norm="no")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    losses = []
    overflow_count = 0
    completed_steps = 0
    model.train()
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        with amp.autocast(enabled=bool(config.train.amp)):
            result = compute_pmt_losses(
                config, model, batch, device, 1,
                criterion_id, criterion_tri, criterion_msel, criterion_dcl,
            )
        total_loss = result["loss"]
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        if not gradients:
            raise RuntimeError("No gradients were produced")
        finite = all(bool(torch.isfinite(gradient).all().item()) for gradient in gradients)
        scaler.step(optimizer)
        scaler.update()
        if finite:
            overflow_count = 0
            losses.append(float(total_loss.detach().cpu()))
            completed_steps += 1
        else:
            overflow_count += 1
            if overflow_count >= 8:
                raise FloatingPointError("AMP overflow persisted for 8 consecutive steps")
        if completed_steps >= steps:
            break
    if completed_steps != steps:
        raise RuntimeError(f"Only completed {completed_steps}/{steps} finite training steps")
    torch.cuda.synchronize(device)
    train_seconds = time.monotonic() - started
    train_peak = int(torch.cuda.max_memory_allocated(device))

    eval_started = time.monotonic()
    average, _ = evaluate_sysu(
        model,
        config.data.root,
        int(config.data.height),
        int(config.data.width),
        mode=config.test.mode,
        gallery_mode=config.test.gallery_mode,
        trials=1,
        batch_size=int(config.test.batch_size),
        num_workers=int(config.test.num_workers),
        device=device,
        output_dir=output.parent / f"{output.stem}.evaluation",
        logger=lambda _message: None,
        source_height=int(config.data.source_height),
        source_width=int(config.data.source_width),
        sr_data_dir=config.data.get("sr_root"),
        sr_modalities=config.data.get("sr_modalities", []),
    )
    torch.cuda.synchronize(device)
    peak = int(torch.cuda.max_memory_allocated(device))
    token_count = int(model.base.patch_embed.num_patches + 1)
    expected_tokens = 211 if (int(config.data.height), int(config.data.width)) == (256, 128) else 883
    if token_count != expected_tokens:
        raise RuntimeError(
            f"Unexpected PMT token count {token_count}; expected {expected_tokens} "
            f"for {int(config.data.height)}x{int(config.data.width)}"
        )
    result = {
        "valid": True,
        "provenance": run_provenance,
        "config": str(config_path.resolve()),
        "experiment_id": config.experiment.id,
        "steps": completed_steps,
        "backbone_chunk_size": int(chunk_size),
        "test_batch_size": int(test_batch_size),
        "token_count": token_count,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_min": min(losses),
        "loss_max": max(losses),
        "metrics_after_20_steps": serialize_metrics(average),
        "train_peak_bytes": train_peak,
        "peak_bytes": peak,
        "train_seconds": float(train_seconds),
        "evaluation_seconds": float(time.monotonic() - eval_started),
        "amp_final_scale": float(scaler.get_scale()),
        "errors": [],
    }
    atomic_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--max-gib", type=float, default=22.0)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--test-batch-size", type=int)
    args = parser.parse_args()
    assert_clean_source()
    config = load_config(args.config)
    output = args.output or (
        PROJECT_ROOT / "outputs/super_resolution/preflight" / f"{config.experiment.id}.json"
    )
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.worker:
        try:
            run_worker(args.config.resolve(), output, args.steps, args.chunk_size, args.test_batch_size)
        except Exception as error:
            atomic_json(output, {
                "valid": False,
                "config": str(args.config.resolve()),
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            })
            raise
        return

    expected_provenance = provenance(args.config.resolve(), config)
    attempts = []
    limit = int(args.max_gib * 1024 ** 3)
    for chunk_size, test_batch_size in ((8, 8), (4, 4), (2, 2)):
        attempt_path = output.with_name(f"{output.stem}.chunk{chunk_size}.test{test_batch_size}.json")
        log_path = attempt_path.with_suffix(".log")
        command = [
            sys.executable, str(Path(__file__).resolve()), "--worker",
            "--config", str(args.config.resolve()),
            "--output", str(attempt_path),
            "--steps", str(args.steps),
            "--chunk-size", str(chunk_size),
            "--test-batch-size", str(test_batch_size),
        ]
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(command, cwd=str(PROJECT_ROOT), env=env, stdout=log, stderr=subprocess.STDOUT)
        attempt = json.loads(attempt_path.read_text(encoding="utf-8")) if attempt_path.exists() else {}
        attempt.update({"returncode": completed.returncode, "log": str(log_path)})
        attempts.append(attempt)
        if (
            completed.returncode == 0
            and attempt.get("valid")
            and attempt.get("provenance") == expected_provenance
            and int(attempt.get("peak_bytes", limit + 1)) <= limit
        ):
            atomic_json(output, {
                "valid": True,
                "provenance": expected_provenance,
                "physical_gpu_index": args.gpu_index,
                "selected_chunk_size": chunk_size,
                "selected_test_batch_size": test_batch_size,
                "max_peak_bytes": limit,
                "selected_attempt": attempt,
                "attempts": attempts,
            })
            print(output)
            return
    atomic_json(output, {
        "valid": False,
        "provenance": expected_provenance,
        "max_peak_bytes": limit,
        "attempts": attempts,
    })
    raise SystemExit("No 3090-safe PMT SR preflight combination passed")


if __name__ == "__main__":
    main()
