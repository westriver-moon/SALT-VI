#!/usr/bin/env python3
"""Copy a code-only legacy snapshot into SALT-VI; never copy model weights/data."""

import hashlib
import json
import shutil
import subprocess
from pathlib import Path


ROOT = Path("/home/cgv841/ybj")
PROJECT = ROOT / "SALT-VI"
DEST = PROJECT / "vendor" / "legacy_code"
RUNTIME = PROJECT / "runtime"

TVI_DIRS = ["core", "data_loader", "network", "solver", "generators", "tools", "base_model", "scripts"]
PMT_DIRS = ["pmt_sysu", "scripts"]
WEIGHT_SUFFIXES = {".pth", ".pt", ".ckpt", ".bin", ".safetensors"}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git_state():
    try:
        commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
        status = subprocess.check_output(["git", "-C", str(ROOT), "status", "--short", "--branch"], text=True).splitlines()
        return {"commit": commit, "status": status[:200]}
    except Exception as exc:
        return {"error": str(exc)}


def copy_tree(source_root, target_root, directories, root_files):
    copied = []
    for name in root_files:
        source = source_root / name
        if source.is_file():
            target = target_root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied.append((source, target))
    for name in directories:
        source_dir = source_root / name
        if not source_dir.exists():
            continue
        for source in source_dir.rglob("*"):
            if not source.is_file() or source.suffix.lower() in WEIGHT_SUFFIXES:
                continue
            target = target_root / source.relative_to(source_root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied.append((source, target))
    return copied


def main():
    tvi_target = DEST / "TVI-LFM"
    pmt_target = DEST / "PMT-SYSU"
    tvi_files = copy_tree(ROOT / "TVI-LFM", tvi_target, TVI_DIRS, ["main.py", "LICENSE", "environment-server.yml", "requirements.txt", "requirements-server.txt", "requirements-test.txt", "requirements-generators.txt"])
    pmt_files = copy_tree(ROOT / "PMT-SYSU", pmt_target, PMT_DIRS, ["train.py", "test.py", "README.md", "requirements-pmt.txt"])
    records = []
    for source, target in tvi_files + pmt_files:
        records.append({"source": source.as_posix(), "target": target.relative_to(PROJECT).as_posix(), "bytes": source.stat().st_size, "sha256": digest(source)})
    RUNTIME.mkdir(parents=True, exist_ok=True)
    manifest = {"purpose": "code-only legacy snapshot for reproduction", "weights_copied": False, "datasets_copied": False, "git_state": git_state(), "files": records}
    path = RUNTIME / "code_snapshot_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"files": len(records), "bytes": sum(r["bytes"] for r in records), "manifest": path.as_posix(), "weights_copied": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
