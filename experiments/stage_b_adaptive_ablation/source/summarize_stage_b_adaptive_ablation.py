#!/usr/bin/env python
import csv
import json
import re
from datetime import datetime
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = [
    "configs/stage_b/adaptive_no_sff/b1_scalar_alpha.yaml",
    "configs/stage_b/adaptive_no_sff/b2_sample_gate.yaml",
    "configs/stage_b/adaptive_no_sff/b3_channel_gate.yaml",
    "configs/stage_b/adaptive_no_sff/b4_residual_gate.yaml",
    "configs/stage_b/adaptive_no_sff/b5_norm_residual_gate.yaml",
]
OUT_ROOT = REPO_ROOT / "train_outputs" / "stage_b_adaptive_ablation"
STATUS_PATH = OUT_ROOT / "status.json"
SUMMARY_CSV = OUT_ROOT / "summary.csv"
SUMMARY_MD = OUT_ROOT / "SUMMARY.md"
EPOCH_RE = re.compile(r"Time:\s*([0-9:-]+);\s*Epoch:\s*(\d+);")
BEST_RE = re.compile(
    r"Best Fusion_RGB mINP:\s*([0-9.]+), Best mAP:\s*([0-9.]+), Best Rank1:\s*([0-9.]+)"
)


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.load(f, Loader=yaml.FullLoader) or {}


def read_status():
    if STATUS_PATH.exists():
        return json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    return {}


def parse_best_metrics(log_path):
    if not log_path.is_file():
        return {
            "best_epoch": None,
            "rank1": None,
            "map": None,
            "minp": None,
        }

    best_epoch = None
    best_rank1 = None
    best_map = None
    best_minp = None
    current_epoch = None
    pending_new_best = False

    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        epoch_match = EPOCH_RE.search(line)
        if epoch_match:
            current_epoch = int(epoch_match.group(2))
            continue

        if "New Best" in line:
            pending_new_best = True
            continue

        best_match = BEST_RE.search(line)
        if best_match and (pending_new_best or best_epoch is None):
            best_minp = float(best_match.group(1))
            best_map = float(best_match.group(2))
            best_rank1 = float(best_match.group(3))
            best_epoch = current_epoch
            pending_new_best = False

    return {
        "best_epoch": best_epoch,
        "rank1": best_rank1,
        "map": best_map,
        "minp": best_minp,
    }


def format_metric(value):
    if value is None:
        return ""
    return f"{float(value):.5f}"


def sort_key(row):
    if row["Rank-1"] is None:
        return (1, 0.0)
    return (0, -float(row["Rank-1"]))


def resolve_train_log(output_root):
    direct_log = output_root / "logs" / "log.log"
    if direct_log.is_file():
        return direct_log

    candidates = sorted(output_root.glob("**/logs/log.log"))
    if candidates:
        return candidates[0]

    return direct_log


def build_rows():
    status = read_status()
    rows = []
    for config_path in CONFIGS:
        config_file = REPO_ROOT / config_path
        config = load_yaml(config_file)
        experiment = Path(config_path).stem
        launcher_log = OUT_ROOT / experiment / "launcher.log"
        output_root = Path(config.get("output_path", ""))
        if not output_root.is_absolute():
            output_root = REPO_ROOT / output_root
        train_log = resolve_train_log(output_root)
        metrics = parse_best_metrics(train_log)
        if metrics["rank1"] is None:
            metrics = parse_best_metrics(launcher_log)
        state = status.get(experiment, {})
        rows.append(
            {
                "experiment": experiment,
                "adaptive_fusion_type": config.get("adaptive_fusion_type", ""),
                "config_path": config_path,
                "best_epoch": metrics["best_epoch"],
                "Rank-1": metrics["rank1"],
                "mAP": metrics["map"],
                "mINP": metrics["minp"],
                "status": state.get("status", "pending"),
                "launcher_log": str(launcher_log),
                "train_log": str(train_log),
            }
        )
    rows.sort(key=sort_key)
    return rows


def write_csv(rows):
    fieldnames = [
        "experiment",
        "adaptive_fusion_type",
        "config_path",
        "best_epoch",
        "Rank-1",
        "mAP",
        "mINP",
        "status",
        "launcher_log",
        "train_log",
    ]
    with SUMMARY_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_md(rows):
    lines = [
        "# Stage B Adaptive No-SFF Ablation Summary",
        "",
        f"Generated at: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "| experiment | adaptive_fusion_type | Rank-1 | mAP | mINP | best_epoch | status | config_path |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {experiment} | {adaptive_fusion_type} | {rank1} | {mapv} | {minp} | {best_epoch} | {status} | {config_path} |".format(
                experiment=row["experiment"],
                adaptive_fusion_type=row["adaptive_fusion_type"],
                rank1=format_metric(row["Rank-1"]),
                mapv=format_metric(row["mAP"]),
                minp=format_metric(row["mINP"]),
                best_epoch=row["best_epoch"] if row["best_epoch"] is not None else "",
                status=row["status"],
                config_path=row["config_path"],
            )
        )
    lines.extend(
        [
            "",
            "Sorted by Rank-1 descending.",
        ]
    )
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    write_csv(rows)
    write_md(rows)
    print(str(SUMMARY_CSV))
    print(str(SUMMARY_MD))


if __name__ == "__main__":
    main()
