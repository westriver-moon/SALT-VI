#!/usr/bin/env python
"""Create unified summaries and the honest FINAL_REPORT for metric boost."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import (
    BEST_RE,
    EPOCH_RE,
    EXPECTED_E4,
    REPORT_ROOT,
    atomic_write_json,
    atomic_write_text,
    load_yaml,
    read_json,
)
from run_eval_sweep import build_eval_plan
from run_imta_experiments import DEFAULT_PLAN as DEFAULT_IMTA_PLAN, load_plan as load_imta_plan
from run_pairwise_improvements import DEFAULT_PLAN as DEFAULT_PAIRWISE_PLAN, load_plan as load_pairwise_plan
from run_fgap_experiments import DEFAULT_PLAN as DEFAULT_FGAP_PLAN, load_plan as load_fgap_plan
from run_train_sweep import build_train_plan


SUMMARY_FIELDS = [
    "Experiment", "Phase", "Training", "MER", "TTA", "Re-ranking", "Ensemble",
    "Rank-1", "mAP", "mINP", "delta_R1_pp", "delta_mAP_pp", "delta_mINP_pp",
    "Validity", "Status", "Git commit SHA", "Checkpoint", "Best epoch", "GPU",
    "Start time", "End time", "Runtime config", "Command", "Log path", "Error",
    "Manifest path", "Baseline experiment", "Design summary", "Config change summary",
    "Code change summary", "Worktree dirty", "Reproducibility status",
]


def _parse_training_log(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    current_epoch = None
    best = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        epoch_match = EPOCH_RE.search(line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
        match = BEST_RE.search(line)
        if match and current_epoch is not None:
            candidate = {
                "best_epoch": current_epoch,
                "mINP": float(match.group(1)),
                "mAP": float(match.group(2)),
                "Rank-1": float(match.group(3)),
            }
            if best is None or (candidate["Rank-1"], candidate["mAP"], candidate["mINP"]) > (
                best["Rank-1"], best["mAP"], best["mINP"]
            ):
                best = candidate
    return best or {}


def _find_training_checkpoint(status: Mapping[str, Any], best_epoch: Optional[int]) -> Optional[str]:
    if best_epoch is None:
        return status.get("checkpoint")
    runtime_path = status.get("runtime_config")
    if not runtime_path or not Path(runtime_path).is_file():
        return status.get("checkpoint")
    config = load_yaml(Path(runtime_path))
    output_root = Path(str(config.get("output_path", "")))
    if output_root.is_dir():
        matches = sorted(output_root.rglob(f"model_Fusion_{best_epoch}.pth"))
        if matches:
            return str(matches[-1].resolve())
    return status.get("checkpoint")


def _bool_text(value: bool) -> str:
    return "yes" if bool(value) else "no"


def _is_training_stage(stage: Any) -> bool:
    return str(stage).startswith(("TRAIN-", "PAIRWISE-", "IMTA-", "FGAP-"))


def _runtime_traits(status: Mapping[str, Any]) -> Dict[str, Any]:
    path = status.get("runtime_config")
    config = load_yaml(Path(path)) if path and Path(path).is_file() else {}
    scales = config.get("test_multi_scale", [[config.get("img_h", 288), config.get("img_w", 144)]])
    tta = bool(config.get("test_flip_tta", False)) or len(scales) > 1
    ensemble = str(config.get("ensemble_mode", "none"))
    return {
        "Training": _is_training_stage(status.get("phase", "")),
        "MER": bool(config.get("CAT_EVAL", False)),
        "TTA": tta,
        "Re-ranking": bool(config.get("rerank", False)),
        "Ensemble": ensemble,
    }


def _row(spec: Mapping[str, Any], status: Mapping[str, Any]) -> Dict[str, Any]:
    status = dict(status)
    if _is_training_stage(spec["stage"]) and status.get("status") == "completed_pending_summary":
        parsed = _parse_training_log(Path(str(status.get("log_path", ""))))
        if parsed:
            status.update(parsed)
            status["checkpoint"] = _find_training_checkpoint(status, parsed["best_epoch"])
            status["status"] = "succeeded"
    rank1 = status.get("Rank-1")
    map_value = status.get("mAP")
    minp = status.get("mINP")
    traits = _runtime_traits(status)
    command = status.get("command_shell") or status.get("command")
    if isinstance(command, list):
        command = " ".join(str(item) for item in command)
    runtime_path = status.get("runtime_config")
    manifest_path = Path(runtime_path).parent / "manifest.json" if runtime_path else None
    manifest = read_json(manifest_path, {}) if manifest_path and manifest_path.is_file() else {}
    return {
        "Experiment": spec["id"],
        "Phase": spec["stage"],
        "Training": _bool_text(traits["Training"]),
        "MER": _bool_text(traits["MER"]),
        "TTA": _bool_text(traits["TTA"]),
        "Re-ranking": _bool_text(traits["Re-ranking"]),
        "Ensemble": traits["Ensemble"],
        "Rank-1": rank1,
        "mAP": map_value,
        "mINP": minp,
        "delta_R1_pp": None if rank1 is None else (float(rank1) - EXPECTED_E4["Rank-1"]) * 100.0,
        "delta_mAP_pp": None if map_value is None else (float(map_value) - EXPECTED_E4["mAP"]) * 100.0,
        "delta_mINP_pp": None if minp is None else (float(minp) - EXPECTED_E4["mINP"]) * 100.0,
        "Validity": status.get("validity", spec.get("validity", "unknown")),
        "Status": status.get("status", "not-prepared"),
        "Git commit SHA": status.get("git_commit_sha"),
        "Checkpoint": status.get("checkpoint"),
        "Best epoch": status.get("best_epoch"),
        "GPU": status.get("gpu"),
        "Start time": status.get("start_time"),
        "End time": status.get("end_time"),
        "Runtime config": status.get("runtime_config"),
        "Command": command,
        "Log path": status.get("log_path"),
        "Error": status.get("error"),
        "Manifest path": str(manifest_path.resolve()) if manifest_path and manifest_path.is_file() else None,
        "Baseline experiment": manifest.get("baseline_experiment_id"),
        "Design summary": manifest.get("design_summary"),
        "Config change summary": manifest.get("config_change_summary"),
        "Code change summary": manifest.get("code_change_summary"),
        "Worktree dirty": manifest.get("worktree_dirty"),
        "Reproducibility status": manifest.get("reproducibility_status", "legacy-insufficient-evidence"),
    }


def build_all_plans() -> List[Dict[str, Any]]:
    """Return every experiment family that contributes to the unified report."""
    return (
        build_eval_plan()
        + build_train_plan()
        + load_pairwise_plan(DEFAULT_PAIRWISE_PLAN)
        + load_imta_plan(DEFAULT_IMTA_PLAN)
        + load_fgap_plan(DEFAULT_FGAP_PLAN)
    )


def collect_rows(report_root: Path = REPORT_ROOT) -> List[Dict[str, Any]]:
    rows = []
    for spec in build_all_plans():
        status_path = Path(report_root) / "runs" / spec["id"] / "status.json"
        status = read_json(status_path, {})
        if _is_training_stage(spec["stage"]) and status.get("status") == "completed_pending_summary":
            parsed = _parse_training_log(Path(str(status.get("log_path", ""))))
            if parsed:
                status.update(parsed)
                status["checkpoint"] = _find_training_checkpoint(status, parsed["best_epoch"])
                status["status"] = "succeeded"
                atomic_write_json(status_path, status)
        rows.append(_row(spec, status))
    return rows


def train4_seed_statistics(rows: Sequence[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    seeds = [
        row for row in rows
        if row["Status"] == "succeeded"
        and str(row["Experiment"]).startswith("TRAIN-4-seed-")
        and row["Rank-1"] is not None
    ]
    if not seeds:
        return None
    result = {"count": len(seeds), "experiments": [row["Experiment"] for row in seeds]}
    for key in ("Rank-1", "mAP", "mINP"):
        values = np.asarray([float(row[key]) for row in seeds], dtype=np.float64)
        result[key] = {"mean": float(values.mean()), "std": float(values.std()), "best": float(values.max())}
    return result


def _format_metric(value: Any) -> str:
    return "—" if value is None else f"{float(value):.5f}"


def _format_delta(value: Any) -> str:
    return "—" if value is None else f"{float(value):+.3f}"


def render_summary_markdown(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# Metric Boost Summary",
        "",
        "| Experiment | Training | MER | TTA | Re-ranking | Ensemble | Rank-1 | mAP | mINP | ΔR1 | ΔmAP | ΔmINP | Validity | Status | Reproducibility |",
        "|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {Experiment} | {Training} | {MER} | {TTA} | {Re-ranking} | {Ensemble} | {rank1} | {mapv} | {minp} | {dr1} | {dmap} | {dminp} | {Validity} | {Status} | {Reproducibility status} |".format(
                **row,
                rank1=_format_metric(row["Rank-1"]),
                mapv=_format_metric(row["mAP"]),
                minp=_format_metric(row["mINP"]),
                dr1=_format_delta(row["delta_R1_pp"]),
                dmap=_format_delta(row["delta_mAP_pp"]),
                dminp=_format_delta(row["delta_mINP_pp"]),
            )
        )
    return "\n".join(lines) + "\n"


def _best(rows: Iterable[Mapping[str, Any]], predicate) -> Optional[Mapping[str, Any]]:
    candidates = [row for row in rows if row["Status"] == "succeeded" and row["Rank-1"] is not None and predicate(row)]
    return max(candidates, key=lambda row: (row["Rank-1"], row["mAP"], row["mINP"])) if candidates else None


def _best_line(label: str, row: Optional[Mapping[str, Any]]) -> str:
    if row is None:
        return f"- {label}: not available (no qualifying experiment has run)."
    return f"- {label}: `{row['Experiment']}` — Rank-1 `{row['Rank-1']:.5f}`, mAP `{row['mAP']:.5f}`, mINP `{row['mINP']:.5f}`."


def render_final_report(rows: Sequence[Mapping[str, Any]]) -> str:
    eval0 = next(row for row in rows if row["Experiment"] == "EVAL-0")
    succeeded_rows = [row for row in rows if row["Status"] == "succeeded"]
    configured_only = [row for row in rows if row["Status"] in {"pending", "blocked", "not-prepared"}]
    failed = [row for row in rows if row["Status"] == "failed"]
    best_single = _best(rows, lambda row: row["Training"] == "yes" and row["TTA"] == "no" and row["Re-ranking"] == "no" and row["Ensemble"] == "none")
    best_ensemble = _best(rows, lambda row: row["Ensemble"] in {"feature", "score"} and row["Re-ranking"] == "no")
    best_any = _best(rows, lambda row: True)
    seed_stats = train4_seed_statistics(rows)
    table = render_summary_markdown(rows).split("\n", 2)[2]
    lines = [
        "# FINAL REPORT — SYSU-MM01 Metric Boost",
        "",
        "## Execution status",
        "",
        f"- E4 reproduced in this run: **{'yes' if eval0['Status'] == 'succeeded' else 'no; E4 has not been run (EVAL-0 pending)'}**.",
        f"- Experiments actually succeeded: **{len(succeeded_rows)}**.",
        f"- Experiments configured but not run/terminal: **{len(configured_only)}**.",
        f"- Experiments failed: **{len(failed)}**.",
        "- Reference only (not a new run): Rank-1 `0.81620`, mAP `0.78663`, mINP `0.67711`.",
        "- No test identity label is authorized for training, tuning, re-ranking selection, or model selection.",
        "",
        "## Unified metrics",
        "",
        table,
        "## Required best-result categories",
        "",
        _best_line("A. Best standard single model, no TTA/re-ranking", best_single),
        _best_line("B. Best standard ensemble, optional TTA, no re-ranking", best_ensemble),
        _best_line("C. Best metric-only result with optional MER/TTA/ensemble/re-ranking", best_any),
        "",
        "## Training and technique audit",
        "",
        "- Retraining: no prepared TRAIN experiment is counted as run until its status is `succeeded`.",
        "- TTA, MER, re-ranking, and ensemble are separate columns and are never merged into an unlabeled result.",
        "- Weighted MER and re-ranking searches are explicitly `exploratory test-set-tuned` without an independent validation set.",
        "- Checkpoint ensemble uses only existing, loadable checkpoints and is capped at five.",
        "- Best retainable training configuration: not available until TRAIN experiments run.",
        f"- TRAIN-4 seed statistics: `{json.dumps(seed_stats, ensure_ascii=False) if seed_stats else 'not available'}`.",
        "",
        "## Failed directions",
        "",
        "- None can be scientifically declared failed before the corresponding experiment runs.",
        "",
        "## Next experiments (maximum three)",
        "",
        "1. EVAL-0 — reproduce E4 under the official 10-trial all-search single-shot protocol.",
        "2. EVAL-1 — measure legacy equal-weight MER only if EVAL-0 passes.",
        "3. EVAL-2 — run the bounded 25-point weighted MER grid and label it exploratory.",
        "",
        "Every path, command, checkpoint, log, Git SHA, time, GPU, and status is retained in `summary.csv` and `summary.json`.",
        "",
    ]
    return "\n".join(lines)


def write_outputs(report_root: Path = REPORT_ROOT) -> List[Dict[str, Any]]:
    report_root = Path(report_root)
    report_root.mkdir(parents=True, exist_ok=True)
    rows = collect_rows(report_root)
    csv_path = report_root / "summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    atomic_write_json(
        report_root / "summary.json",
        {"reference_e4": EXPECTED_E4, "train4_seed_statistics": train4_seed_statistics(rows), "rows": rows},
    )
    atomic_write_text(report_root / "summary.md", render_summary_markdown(rows))
    atomic_write_text(report_root / "FINAL_REPORT.md", render_final_report(rows))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-root", type=Path, default=REPORT_ROOT)
    args = parser.parse_args()
    rows = write_outputs(args.report_root)
    print(json.dumps({"row_count": len(rows), "report_root": str(args.report_root)}, sort_keys=True))


if __name__ == "__main__":
    main()
