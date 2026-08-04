#!/usr/bin/env python3
"""Convert the formal A3 vision-text checkpoint into the canonical SALT-VI format."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from salt_vi.engine.build import CLIP2ReID
from salt_vi.utils.utils import load_train_configs

EXPECTED_STAGE_A_PARAMETER_COUNT = 160
EXPECTED_STAGE_A_EPOCH = 24
EXPECTED_SOURCE_SHA256 = "75975a07c4fcd6b44649d67252fe4bdff9ac132d7c174b2fe64c114f5770952a"
EXPECTED_POSITION_SHAPE = (1, 883, 768)
POSITION_SOURCE_KEY = "base.pos_embed"
POSITION_TARGET_KEY = "base_model.visual.vit.pos_embed"


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def map_stage_a_key(key):
    if key.startswith("base."):
        return "base_model.visual.vit." + key[len("base."):]
    if key.startswith("bottleneck."):
        return "classifier.BN." + key[len("bottleneck."):]
    if key.startswith("classifier."):
        return "classifier.classifier." + key[len("classifier."):]
    raise KeyError(f"Unsupported StageA parameter key: {key}")


def validate_formal_a3_checkpoint(checkpoint):
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise RuntimeError("A3 checkpoint must embed its resolved configuration")
    expected = {
        "experiment.id": "PMSR-A3-swinir-both-x2",
        "experiment.seed": 0,
        "data.height": 512,
        "data.width": 256,
        "data.source_height": 256,
        "data.source_width": 128,
        "data.sr_modalities": ["rgb", "ir"],
        "train.max_epoch": 24,
    }
    for dotted_key, expected_value in expected.items():
        value = config
        for part in dotted_key.split("."):
            if not isinstance(value, dict) or part not in value:
                raise RuntimeError(f"A3 checkpoint config is missing {dotted_key}")
            value = value[part]
        if value != expected_value:
            raise RuntimeError(
                f"Formal A3 config mismatch for {dotted_key}: expected {expected_value!r}, got {value!r}"
            )
    return config


def build_mapping(source_state, target_state, expected_count=EXPECTED_STAGE_A_PARAMETER_COUNT):
    if len(source_state) != expected_count:
        raise RuntimeError(f"StageA parameter count mismatch: expected {expected_count}, got {len(source_state)}")
    mapping = {}
    for source_key, source_tensor in source_state.items():
        target_key = map_stage_a_key(source_key)
        if target_key in mapping.values():
            raise RuntimeError(f"Duplicate target mapping: {target_key}")
        if target_key not in target_state:
            raise RuntimeError(f"Mapped target key is missing: {source_key} -> {target_key}")
        target_tensor = target_state[target_key]
        if source_tensor.shape != target_tensor.shape:
            raise RuntimeError(
                f"Shape mismatch for {source_key} -> {target_key}: "
                f"{tuple(source_tensor.shape)} != {tuple(target_tensor.shape)}"
            )
        if source_tensor.dtype != target_tensor.dtype:
            raise RuntimeError(
                f"Dtype mismatch for {source_key} -> {target_key}: "
                f"{source_tensor.dtype} != {target_tensor.dtype}"
            )
        mapping[source_key] = target_key
    if set(mapping) != set(source_state):
        raise RuntimeError("Mapping did not consume every source parameter")
    return mapping


def git_provenance():
    def run(*args):
        return subprocess.check_output(args, cwd=REPO_ROOT, text=True).strip()
    try:
        return {
            "commit": run("git", "rev-parse", "HEAD"),
            "dirty": bool(run("git", "status", "--porcelain")),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def json_safe(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return repr(value)


def atomic_torch_save(state, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".tmp.{os.getpid()}")
    try:
        torch.save(state, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_json_save(payload, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".tmp.{os.getpid()}")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def convert(source, config_path, output, manifest_path, allow_overwrite=False):
    source = source.resolve()
    config_path = config_path.resolve()
    output = output.resolve()
    manifest_path = manifest_path.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    for destination in (output, manifest_path):
        if destination.exists() and not allow_overwrite:
            raise FileExistsError(f"Refusing to overwrite {destination}; pass --allow-overwrite explicitly")

    source_hash_before = sha256_file(source)
    if source_hash_before != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(
            f"Formal A3 best.pth SHA256 mismatch: expected {EXPECTED_SOURCE_SHA256}, "
            f"got {source_hash_before}"
        )
    checkpoint = torch.load(source, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise RuntimeError("Expected native vision-text checkpoint with model_state_dict")
    epoch = int(checkpoint.get("epoch", -1))
    if epoch != EXPECTED_STAGE_A_EPOCH:
        raise RuntimeError(f"Expected formal A3 best epoch {EXPECTED_STAGE_A_EPOCH}, got {epoch}")
    source_config = validate_formal_a3_checkpoint(checkpoint)
    source_state = checkpoint["model_state_dict"]
    if tuple(source_state[POSITION_SOURCE_KEY].shape) != EXPECTED_POSITION_SHAPE:
        raise RuntimeError(
            f"A3 position embedding must remain {EXPECTED_POSITION_SHAPE}; "
            f"got {tuple(source_state[POSITION_SOURCE_KEY].shape)}"
        )

    config = load_train_configs(str(config_path))
    if list(config.img_size) != [512, 256] or not config.pmt_patch_embed:
        raise RuntimeError("Bridge config must use 512x256 and explicit MBPatch")
    if list(config.sysu_sr_modalities) != ["rgb", "ir"]:
        raise RuntimeError("A3 bridge config must use both RGB and IR SwinIR inputs")
    if not bool(config.sysu_sr_exact_size):
        raise RuntimeError("A3 bridge config must enforce exact-size SR inputs")
    torch.manual_seed(int(config.seed))
    model = CLIP2ReID(config, num_classes=int(config.pid_num))
    target_state = model.state_dict()
    if tuple(target_state[POSITION_TARGET_KEY].shape) != EXPECTED_POSITION_SHAPE:
        raise RuntimeError(
            f"SALT-VI position embedding must be {EXPECTED_POSITION_SHAPE}; "
            f"got {tuple(target_state[POSITION_TARGET_KEY].shape)}; interpolation is forbidden"
        )

    mapping = build_mapping(source_state, target_state)
    converted = {key: tensor.detach().cpu().clone() for key, tensor in target_state.items()}
    for source_key, target_key in mapping.items():
        converted[target_key] = source_state[source_key].detach().cpu().clone()
        if not torch.equal(converted[target_key], source_state[source_key].detach().cpu()):
            raise RuntimeError(f"Non-bitwise mapping detected: {source_key} -> {target_key}")
    model.load_state_dict(converted, strict=True)

    atomic_torch_save(converted, output)
    reloaded = torch.load(output, map_location="cpu")
    model.load_state_dict(reloaded, strict=True)
    for source_key, target_key in mapping.items():
        if not torch.equal(reloaded[target_key], source_state[source_key].detach().cpu()):
            raise RuntimeError(f"Round-trip mismatch: {source_key} -> {target_key}")
    if source_hash_before != sha256_file(source):
        raise RuntimeError("Source checkpoint changed during conversion")

    manifest = {
        "schema_version": 1,
        "conversion": "vision-text A3 native -> SALT-VI image-only",
        "source_checkpoint": str(source),
        "source_sha256": source_hash_before,
        "source_epoch": epoch,
        "source_best_mAP": checkpoint.get("best_mAP"),
        "source_selection": {
            "artifact": "formal best.pth",
            "saved_epoch": epoch,
            "observed_save_metric": "mAP",
            "config_declared_selection_rule": source_config["experiment"].get("selection_rule"),
            "note": "The declared Rank-1 rule and the mAP-selected best.pth disagree; this conversion uses the formal existing best.pth artifact without inventing an unavailable epoch-22 checkpoint.",
        },
        "source_config": json_safe(source_config),
        "target_checkpoint": str(output),
        "target_sha256": sha256_file(output),
        "target_state_key_count": len(converted),
        "bridge_config": str(config_path),
        "bridge_config_sha256": sha256_file(config_path),
        "converter": str(Path(__file__).resolve()),
        "converter_sha256": sha256_file(Path(__file__).resolve()),
        "mapping_count": len(mapping),
        "mapping": mapping,
        "position_embedding_shape": list(EXPECTED_POSITION_SHAPE),
        "position_embedding_interpolated": False,
        "code": git_provenance(),
        "strict_reload": True,
        "bitwise_round_trip": True,
    }
    atomic_json_save(manifest, manifest_path)
    return manifest


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--allow-overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    manifest_path = args.manifest or Path(str(args.output) + ".manifest.json")
    manifest = convert(args.source, args.config, args.output, manifest_path, args.allow_overwrite)
    print(json.dumps({key: manifest[key] for key in ("target_checkpoint", "target_sha256", "mapping_count", "strict_reload")}, indent=2))


if __name__ == "__main__":
    main()
