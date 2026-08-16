#!/usr/bin/env python
import csv
import json
import re
from pathlib import Path

import yaml


CONFIG_DIR = Path("configs/stage_a/sampling_mining_ablation")
LOG_ROOT = Path("logs/sampling_mining_ablation")
OUT_ROOT = Path("train_outputs/sampling_mining_ablation")
STATUS_PATH = OUT_ROOT / "status.json"


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.load(f, Loader=yaml.FullLoader) or {}


def first_float(text):
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    return float(match.group(0)) if match else None


def parse_log(log_path):
    text = log_path.read_text(encoding="utf-8", errors="replace")
    current_epoch = None
    best = {"best_epoch": "", "Rank-1": "", "mAP": "", "mINP": ""}
    final = {"final_Rank-1": "", "final_mAP": "", "final_mINP": "", "final_epoch": ""}
    for line in text.splitlines():
        epoch_match = re.search(r"Epoch:\s*(\d+)", line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
            final["final_epoch"] = current_epoch
        best_match = re.search(r"Best .*?mINP:\s*([^,]+),\s*Best mAP:\s*([^,]+),\s*Best Rank1:\s*([^,\s]+)", line)
        if best_match:
            best["best_epoch"] = current_epoch if current_epoch is not None else ""
            best["mINP"] = first_float(best_match.group(1))
            best["mAP"] = first_float(best_match.group(2))
            best["Rank-1"] = first_float(best_match.group(3))

    metric_blocks = re.finditer(
        r"mINP:\s*([^\n]+)\s*\nmAP:\s*([^\n]+)\s*\n\s*Rank:\s*\[?([^\]\n\s,]+)",
        text,
        flags=re.MULTILINE,
    )
    for match in metric_blocks:
        final["final_mINP"] = first_float(match.group(1))
        final["final_mAP"] = first_float(match.group(2))
        final["final_Rank-1"] = first_float(match.group(3))
    return {**best, **final}


def find_log(exp):
    root = LOG_ROOT / exp
    candidates = list(root.rglob("log.log")) + list(root.rglob("train.log"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    status = {}
    if STATUS_PATH.exists():
        status = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    rows = []
    for cfg_path in sorted(CONFIG_DIR.glob("*.yaml")):
        cfg = load_yaml(cfg_path)
        exp = cfg_path.stem
        log_path = find_log(exp)
        metrics = {}
        if log_path:
            metrics = parse_log(log_path)
        run_status = status.get(exp, {}).get("status", "unknown")
        if not log_path:
            run_status = "missing_log" if run_status == "unknown" else run_status
        rows.append({
            "exp": exp,
            "sampler_type": cfg.get("sampler_type", ""),
            "triplet_mining": cfg.get("triplet_mining", ""),
            "batch_size": cfg.get("batch_size", ""),
            "num_pos": cfg.get("num_pos", ""),
            "best_epoch": metrics.get("best_epoch", ""),
            "Rank-1": metrics.get("Rank-1", ""),
            "mAP": metrics.get("mAP", ""),
            "mINP": metrics.get("mINP", ""),
            "final_Rank-1": metrics.get("final_Rank-1", ""),
            "final_mAP": metrics.get("final_mAP", ""),
            "final_mINP": metrics.get("final_mINP", ""),
            "status": run_status,
        })

    fields = ["exp", "sampler_type", "triplet_mining", "batch_size", "num_pos", "best_epoch", "Rank-1", "mAP", "mINP", "final_Rank-1", "final_mAP", "final_mINP", "status"]
    with (OUT_ROOT / "results.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    lines = ["| " + " | ".join(fields) + " |", "|" + "|".join(["---"] * len(fields)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    (OUT_ROOT / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_ROOT / 'RESULTS.md'} and {OUT_ROOT / 'results.csv'}")


if __name__ == "__main__":
    main()
