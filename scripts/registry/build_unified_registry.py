#!/usr/bin/env python3
"""Copy legacy experiment tables and build one lossless normalized registry."""

import csv
import hashlib
import json
import os
import re
import shutil
from pathlib import Path


PROJECT_ROOT = Path("/home/cgv841/ybj/SALT-VI")
LEGACY_ROOT = Path("/home/cgv841/ybj")
DEST_ROOT = PROJECT_ROOT / "reports" / "experiment_registry"
SOURCE_DEST = DEST_ROOT / "source_tables" / "legacy"
ROOTS = [
    LEGACY_ROOT / "TVI-LFM" / "reports",
    LEGACY_ROOT / "experiments",
    LEGACY_ROOT / "PMT-SYSU" / "outputs",
    LEGACY_ROOT / "TVI-LFM" / "train_outputs",
    LEGACY_ROOT / "docs",
]

FIELDS = [
    "record_id", "source_table", "source_original_path", "source_row_number",
    "record_type", "experiment_id", "stage", "experiment_group", "description",
    "dataset", "evaluation_protocol", "modalities", "seed", "code_root",
    "code_commit", "config_path", "init_checkpoint", "pretrained_path",
    "data_root", "derived_data_root", "environment_file", "launch_command",
    "run_dir", "status", "lifecycle", "best_epoch", "rank1", "mAP", "mINP",
    "rank5", "rank10", "checkpoint_path", "checkpoint_sha256",
    "checkpoint_status", "metrics_source", "log_source", "selection_rule",
    "extra_metrics_json", "notes", "source_sha256",
]


def norm_key(value):
    return re.sub(r"[^a-z0-9]", "", (value or "").lower())


def get(row, *names):
    lookup = {norm_key(k): (v or "").strip() for k, v in row.items()}
    for name in names:
        value = lookup.get(norm_key(name), "")
        if value:
            return value
    return ""


def record_type(path):
    name = path.name.lower()
    if "curve" in name:
        return "curve"
    if "inventory" in name or "manifest" in name:
        return "inventory"
    if "summary" in name or "result" in name or "best" in name:
        return "summary_or_result"
    if "metric" in name or "score" in name or "eval" in name:
        return "metrics"
    return "table"


def experiment_id(row, source):
    value = get(row, "experiment_id", "experiment", "exp", "run_id", "run", "job_id", "method", "name")
    if value:
        return value
    return source.parent.name + "/" + source.stem


def source_files():
    found = []
    for root in ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".csv", ".tsv"}:
                found.append(path)
    return sorted(set(found))


def main():
    DEST_ROOT.mkdir(parents=True, exist_ok=True)
    SOURCE_DEST.mkdir(parents=True, exist_ok=True)
    rows = []
    copied = 0
    for source in source_files():
        relative = source.relative_to(LEGACY_ROOT)
        copied_path = SOURCE_DEST / relative
        copied_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, copied_path)
        copied += 1
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        delimiter = "\t" if source.suffix.lower() == ".tsv" else ","
        try:
            with source.open("r", encoding="utf-8-sig", newline="", errors="replace") as handle:
                reader = csv.DictReader(handle, delimiter=delimiter)
                for number, raw in enumerate(reader, start=2):
                    raw = {str(k): (v or "").strip() for k, v in raw.items() if k is not None}
                    if not any(raw.values()):
                        continue
                    canonical = {
                        "record_id": f"{relative.as_posix()}#row{number}",
                        "source_table": (Path("reports/experiment_registry/source_tables/legacy") / relative).as_posix(),
                        "source_original_path": source.as_posix(),
                        "source_row_number": str(number),
                        "record_type": record_type(source),
                        "experiment_id": experiment_id(raw, source),
                        "stage": get(raw, "stage", "phase"),
                        "experiment_group": get(raw, "group", "experiment_group"),
                        "description": get(raw, "description", "design_summary"),
                        "dataset": get(raw, "dataset"),
                        "evaluation_protocol": get(raw, "protocol", "mode", "test_mode"),
                        "modalities": get(raw, "modalities", "training_mode", "test_modality"),
                        "seed": get(raw, "seed"),
                        "code_root": get(raw, "code_root", "worktree"),
                        "code_commit": get(raw, "code_commit", "git_commit_sha", "git commit sha"),
                        "config_path": get(raw, "yaml_path", "config_path", "runtime_config"),
                        "init_checkpoint": get(raw, "init_checkpoint", "training_weight_init", "initialization"),
                        "pretrained_path": get(raw, "pretrained_path", "pretrained"),
                        "data_root": get(raw, "data_root", "dataset_root"),
                        "derived_data_root": get(raw, "derived_data_root", "sr_root"),
                        "environment_file": get(raw, "environment_file", "environment", "environment_path"),
                        "launch_command": get(raw, "launch_command", "command"),
                        "run_dir": get(raw, "run_dir", "output_path", "output", "log_path"),
                        "status": get(raw, "status", "source_status"),
                        "lifecycle": get(raw, "config_lifecycle", "lifecycle"),
                        "best_epoch": get(raw, "best_epoch", "best_rank1_epoch", "checkpoint_epoch", "epoch"),
                        "rank1": get(raw, "rank1", "rank-1", "best_rank1", "final_rank1"),
                        "mAP": get(raw, "mAP", "map", "best_map", "final_map"),
                        "mINP": get(raw, "mINP", "minp", "best_minp", "final_minp"),
                        "rank5": get(raw, "rank5"),
                        "rank10": get(raw, "rank10"),
                        "checkpoint_path": get(raw, "checkpoint", "checkpoint_path", "best_model_path", "source_checkpoint", "archive_checkpoint", "checkpoints"),
                        "checkpoint_sha256": get(raw, "checkpoint_sha256"),
                        "checkpoint_status": get(raw, "checkpoint_status", "checkpoint_retained", "checkpoint_exists"),
                        "metrics_source": (Path("reports/experiment_registry/source_tables/legacy") / relative).as_posix(),
                        "log_source": get(raw, "log_source", "log_path", "train_log", "launcher_log"),
                        "selection_rule": get(raw, "selection_rule", "selection"),
                        "extra_metrics_json": json.dumps(raw, ensure_ascii=False, sort_keys=True),
                        "notes": get(raw, "notes", "note", "validity", "error", "design_summary"),
                        "source_sha256": digest,
                    }
                    if not canonical["checkpoint_status"]:
                        canonical["checkpoint_status"] = "recorded" if canonical["checkpoint_path"] else "not_recorded"
                    rows.append(canonical)
        except (UnicodeError, csv.Error):
            continue
    output = DEST_ROOT / "experiment_registry.csv"
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"source_tables": copied, "records": len(rows), "output": output.as_posix()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
