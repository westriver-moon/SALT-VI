#!/usr/bin/env python3
"""Migrate reproducibility assets referenced by the SALT-VI unified registry."""

import csv
import hashlib
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path


ROOT = Path("/home/cgv841/ybj")
PROJECT = ROOT / "SALT-VI"
REGISTRY = PROJECT / "reports/experiment_registry/experiment_registry.csv"
CONFIG_DEST = PROJECT / "runtime/reproduction_assets/configs"
CHECKPOINT_DEST = PROJECT / "checkpoints/reproduction_assets"
PRETRAINED_DEST = PROJECT / "pretrained/reproduction_assets"
RUNTIME_DEST = PROJECT / "runtime"
CONFIG_SCAN_ROOTS = [
    ROOT / "TVI-LFM" / "config",
    ROOT / "TVI-LFM" / "reports" / "experiment_registry" / "archived_configs",
    ROOT / "PMT-SYSU" / "pmt_sysu" / "config",
    ROOT / "PMT-SYSU" / "outputs",
    ROOT / "experiments",
]


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve(value, field):
    value = (value or "").strip()
    if not value:
        return None
    candidates = []
    if value.startswith("/"):
        candidates.append(Path(value))
    else:
        candidates.extend([
            ROOT / "TVI-LFM" / value,
            ROOT / "PMT-SYSU" / value,
            ROOT / value,
        ])
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()
    return None


def relative_archive(path):
    try:
        return path.relative_to(ROOT)
    except ValueError:
        return Path(path.name)


def copy_config(source):
    relative = relative_archive(source)
    target = CONFIG_DEST / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copy2(source, target)
    return target


def main():
    for directory in [CONFIG_DEST, CHECKPOINT_DEST, PRETRAINED_DEST, RUNTIME_DEST]:
        directory.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(REGISTRY.open("r", encoding="utf-8", newline="")))
    path_fields = {
        "config": "config_path",
        "environment": "environment_file",
    }
    assets = {}
    references = []
    inode_targets = {}
    config_catalog = []
    for existing_dir in [CHECKPOINT_DEST, PRETRAINED_DEST]:
        for existing in existing_dir.glob("*"):
            if existing.is_file():
                real = existing.resolve()
                inode_targets[(real.stat().st_dev, real.stat().st_ino)] = existing

    for row in rows:
        for kind, field in path_fields.items():
            value = (row.get(field) or "").strip()
            if not value:
                continue
            source = resolve(value, field)
            key = (kind, str(source) if source else value)
            if key not in assets:
                entry = {
                    "asset_key": f"{kind}:{len(assets) + 1:05d}",
                    "kind": kind,
                    "source_value": value,
                    "resolved_path": source.as_posix() if source else "",
                    "migrated_path": "",
                    "status": "missing",
                    "bytes": 0,
                    "sha256": "",
                    "sha256_status": "not_computed",
                    "record_ids": [],
                }
                if source:
                    entry["bytes"] = source.stat().st_size
                    if kind in {"config", "environment"}:
                        target = copy_config(source)
                        entry["migrated_path"] = target.relative_to(PROJECT).as_posix()
                        entry["sha256"] = sha256_file(source)
                        entry["sha256_status"] = "computed"
                        entry["status"] = "copied"
                    elif kind in {"checkpoint", "init_checkpoint", "pretrained"}:
                        real = source.resolve()
                        inode = (real.stat().st_dev, real.stat().st_ino)
                        if inode not in inode_targets:
                            target_dir = PRETRAINED_DEST if kind == "pretrained" else CHECKPOINT_DEST
                            target = target_dir / f"asset_{len(inode_targets) + 1:04d}_{real.name}"
                            os.link(real, target)
                            inode_targets[inode] = target
                        target = inode_targets[inode]
                        entry["migrated_path"] = target.relative_to(PROJECT).as_posix()
                        entry["status"] = "hardlinked"
                        entry["sha256_status"] = "not_computed_large_file"
                assets[key] = entry
            assets[key]["record_ids"].append(row.get("record_id", ""))
            references.append({
                "record_id": row.get("record_id", ""),
                "field": field,
                "asset_key": assets[key]["asset_key"],
            })

    for scan_root in CONFIG_SCAN_ROOTS:
        if not scan_root.exists():
            continue
        for source in sorted(scan_root.rglob("*")):
            if source.is_file() and source.suffix.lower() in {".yaml", ".yml"}:
                target = copy_config(source)
                config_catalog.append({
                    "source_path": source.as_posix(),
                    "migrated_path": target.relative_to(PROJECT).as_posix(),
                    "bytes": source.stat().st_size,
                    "sha256": sha256_file(source),
                })

    external_data = sorted({
        value.strip()
        for row in rows
        for field in ["data_root", "derived_data_root"]
        for value in [row.get(field, "")]
        if value and value.strip()
    })
    manifest = {
        "registry": REGISTRY.relative_to(PROJECT).as_posix(),
        "source_rows": len(rows),
        "assets": list(assets.values()),
        "references": references,
        "config_catalog": config_catalog,
        "external_data_roots": external_data,
        "weight_migration": "deferred_by_user; checkpoint and pretrained paths remain external references",
        "code_migration": "not_performed; legacy code roots and commits remain recorded in the registry",
    }
    manifest_path = RUNTIME_DEST / "reproduction_assets_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = defaultdict(int)
    for entry in assets.values():
        summary[f"{entry['kind']}__{entry['status']}"] += 1
    summary_path = RUNTIME_DEST / "reproduction_assets_summary.json"
    summary_path.write_text(json.dumps({"source_rows": len(rows), "unique_assets": len(assets), "config_files": len(config_catalog), "summary": dict(summary)}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"source_rows": len(rows), "unique_assets": len(assets), "config_files": len(config_catalog), "summary": dict(summary), "manifest": manifest_path.as_posix()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
