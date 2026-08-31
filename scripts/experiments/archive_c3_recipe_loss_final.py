#!/usr/bin/env python3
"""Finalize the C3 recipe/loss experiment archive and its Git-readable index."""

from __future__ import annotations

import argparse
import csv
import filecmp
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


EXPERIMENT_ID = "c3-recipe-loss-research-20260829"
GROUP_ID = "stage_a_c3_recipe_loss_research_20260829"
FINAL_VARIANTS = (
    "m2_cm_margin010",
    "m3_cm_margin020",
    "s0_selective_control",
    "s1_relation005",
    "s2_relation010",
)
COMPLETED_VARIANTS = {
    "r0_current",
    "r1_gray4_coupled",
    "r2_gray4_decoupled_ramp3",
    "l0_no_aux",
    "l1_msel_only",
    "l2_dcl_only",
    "l4_msel050_dcl010",
    "m1_cm_margin005",
    "m2_cm_margin010",
    "m3_cm_margin020",
    "s0_selective_control",
}
USER_STOPPED_VARIANTS = {"s1_relation005", "s2_relation010"}
LABELS = {
    "r0_current": "R0 current",
    "r1_gray4_coupled": "R1 gray4 coupled",
    "r2_gray4_decoupled_ramp3": "R2 gray4 decoupled ramp3",
    "l0_no_aux": "L0 no auxiliary",
    "l1_msel_only": "L1 MSEL only",
    "l2_dcl_only": "L2 DCL only",
    "l4_msel050_dcl010": "L4 MSEL0.50 DCL0.10",
    "m1_cm_margin005": "M1 cross-modal margin0.05",
    "m2_cm_margin010": "M2 cross-modal margin0.10",
    "m3_cm_margin020": "M3 cross-modal margin0.20",
    "s0_selective_control": "S0 selective control",
    "s1_relation005": "S1 relation0.05",
    "s2_relation010": "S2 relation0.10",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def copy_file(source: Path, destination: Path, pairs: list[tuple[Path, Path]]) -> None:
    if not source.is_file():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    pairs.append((source, destination))


def copy_tree(source: Path, destination: Path, pairs: list[tuple[Path, Path]]) -> None:
    if not source.is_dir():
        raise FileNotFoundError(source)
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            target = os.readlink(path)
            target_path = destination / path.relative_to(source)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.symlink_to(target)
        elif path.is_file():
            copy_file(path, destination / path.relative_to(source), pairs)


def assert_terminal(state: dict) -> None:
    if state.get("stage") != "finished":
        raise RuntimeError(f"scheduler is not terminal: {state.get('stage')!r}")
    jobs = state.get("jobs") or {}
    expected = COMPLETED_VARIANTS | USER_STOPPED_VARIANTS
    missing = sorted(expected - jobs.keys())
    if missing:
        raise RuntimeError(f"scheduler state is missing jobs: {missing}")
    wrong = []
    for name in sorted(expected):
        actual = jobs[name].get("status")
        wanted = "completed" if name in COMPLETED_VARIANTS else "failed"
        if actual != wanted:
            wrong.append((name, actual, wanted))
    for name in USER_STOPPED_VARIANTS:
        if jobs[name].get("exit_code") != -15:
            wrong.append((name, jobs[name].get("exit_code"), -15))
    if wrong:
        raise RuntimeError(f"unexpected terminal job states: {wrong}")


def compare_pairs(pairs: list[tuple[Path, Path]]) -> dict:
    failed = []
    total_bytes = 0
    for source, destination in pairs:
        total_bytes += source.stat().st_size
        if source.stat().st_size != destination.stat().st_size:
            failed.append(str(destination))
            continue
        if not filecmp.cmp(source, destination, shallow=False):
            failed.append(str(destination))
    if failed:
        raise RuntimeError(f"bytewise verification failed for {failed[:10]}")
    return {
        "method": "bytewise-file-compare",
        "new_checksum_calculated": False,
        "original_retained": True,
        "verified": True,
        "verified_file_count": len(pairs),
        "verified_bytes": total_bytes,
        "failed_files": [],
        "verified_at": utc_now(),
    }


def archive_final_batch(source_root: Path, archive_root: Path, batch1_root: Path) -> dict:
    state = load_json(source_root / "scheduler" / "state.json")
    results = load_json(source_root / "results.json")
    assert_terminal(state)
    if results.get("source") != "recollected-from-evaluation-events":
        raise RuntimeError("results.json was not rebuilt from evaluation events")

    if archive_root.exists():
        manifest = load_json(archive_root / "provenance" / "archive_manifest.json")
        verification = load_json(archive_root / "provenance" / "verification.json")
        if not verification.get("verified"):
            raise RuntimeError(f"existing archive is not verified: {archive_root}")
        return {"manifest": manifest, "verification": verification, "reused": True}

    staging = archive_root.with_name(f"{archive_root.name}.partial-{os.getpid()}")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    pairs: list[tuple[Path, Path]] = []

    for relative in (
        "plan.json",
        "results.json",
        "results.partial.json",
        "results.best_epoch.partial.json",
        "scheduler/state.json",
        "scheduler/collect-best-epoch-final.log",
        "scheduler/collect-final-archive.log",
    ):
        copy_file(source_root / relative, staging / relative, pairs)
    copy_tree(source_root / "diagnostics", staging / "diagnostics", pairs)

    for name in FINAL_VARIANTS:
        copy_file(
            source_root / "events" / f"{name}.jsonl",
            staging / "events" / f"{name}.jsonl",
            pairs,
        )
        copy_file(
            source_root / "launcher_logs" / f"{name}.log",
            staging / "launcher_logs" / f"{name}.log",
            pairs,
        )
        copy_tree(source_root / "runs" / name, staging / "runs" / name, pairs)

    for name in (
        "environment.json",
        "external_assets.json",
        "pip-freeze.txt",
        "repo_state.json",
        "worktree.patch",
        "index.patch",
    ):
        copy_file(
            batch1_root / "provenance" / name,
            staging / "provenance" / "batch1_source_snapshot" / name,
            pairs,
        )

    verification = compare_pairs(pairs)
    inventory = [
        {
            "archive_path": str(destination.relative_to(staging)),
            "source_path": str(source),
            "size_bytes": source.stat().st_size,
            "verification": "bytewise_equal",
        }
        for source, destination in pairs
    ]
    manifest = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "archive_kind": "final-batch2",
        "created_at": utc_now(),
        "source_root": str(source_root),
        "archive_root": str(archive_root),
        "batch1_archive_root": str(batch1_root),
        "archived_variants": list(FINAL_VARIANTS),
        "completed_variants": sorted(COMPLETED_VARIANTS & set(FINAL_VARIANTS)),
        "user_stopped_failed_variants": sorted(USER_STOPPED_VARIANTS),
        "scheduler_terminal_status": state.get("status"),
        "scheduler_stage": state.get("stage"),
        "archive_method": "copy-bytewise-verify-preserve-original-no-new-checksum",
        "original_retained": True,
        "scientific_payload_verification": "bytewise_equal",
        "new_checksum_calculated": False,
        "git_readable_index": "reports/experiment_archives/c3_recipe_loss_research_20260829",
        "notes": (
            "Batch 2 closes the five variants excluded from batch 1. S1 and S2 "
            "are retained as user-stopped failures and are not promoted to completed results."
        ),
    }
    write_json(staging / "provenance" / "final_metrics.json", results)
    write_json(staging / "provenance" / "archive_manifest.json", manifest)
    write_json(staging / "provenance" / "verification.json", verification)
    write_json(
        staging / "provenance" / "file_inventory.json",
        {
            "schema_version": 1,
            "generated_at": utc_now(),
            "scientific_payload": inventory,
            "metadata_files_excluded_from_self_verification": [
                "provenance/final_metrics.json",
                "provenance/archive_manifest.json",
                "provenance/verification.json",
                "provenance/file_inventory.json",
            ],
        },
    )
    os.replace(staging, archive_root)
    return {"manifest": manifest, "verification": verification, "reused": False}


def fmt_pct(value) -> str:
    return "—" if value is None else f"{100.0 * float(value):.4f}%"


def metrics_rows(results: dict, plan: dict) -> list[dict]:
    rows = []
    for run in results["training_runs"]:
        name = run["variant"]
        metrics = run.get("metrics") or {}
        rows.append(
            {
                "variant": name,
                "label": LABELS.get(name, name),
                "phase": run.get("phase"),
                "purpose": plan["variants"][name].get("purpose"),
                "archival_status": (
                    "completed" if name in COMPLETED_VARIANTS else "failed_user_stopped"
                ),
                "scheduler_status": run.get("status"),
                "exit_code": run.get("exit_code"),
                "best_epoch": metrics.get("epoch"),
                "rank1": metrics.get("Rank-1"),
                "mAP": metrics.get("mAP"),
                "mINP": metrics.get("mINP"),
                "evaluated_epochs": metrics.get("evaluated_epochs", []),
                "protocol": metrics.get("protocol"),
                "metrics_scope": (
                    "official_completed_run"
                    if name in COMPLETED_VARIANTS
                    else ("partial_interrupted_run" if metrics else "none")
                ),
            }
        )
    return rows


def write_metrics_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "variant",
        "label",
        "phase",
        "purpose",
        "archival_status",
        "scheduler_status",
        "exit_code",
        "best_epoch",
        "rank1",
        "mAP",
        "mINP",
        "evaluated_epochs",
        "metrics_scope",
        "protocol",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            item = dict(row)
            item["evaluated_epochs"] = json.dumps(item["evaluated_epochs"], separators=(",", ":"))
            writer.writerow(item)


def render_history(rows: list[dict], source_root: Path, archive_root: Path, batch1_root: Path) -> str:
    table = []
    for row in rows:
        note = ""
        if row["variant"] == "s1_relation005":
            note = "（中止前部分指标）"
        elif row["variant"] == "s2_relation010":
            note = "（首轮中止，无验证）"
        table.append(
            f"| {row['label']} | {row['archival_status']}{note} | "
            f"{row['best_epoch'] if row['best_epoch'] is not None else '—'} | "
            f"{fmt_pct(row['rank1'])} | {fmt_pct(row['mAP'])} | {fmt_pct(row['mINP'])} |"
        )
    return f"""# Stage-A C3 recipe and loss research: final archive

Date: 2026-08-31 (Asia/Shanghai)
Experiment group: `{GROUP_ID}`
Dataset and protocol: SYSU-MM01, all-search, single-shot, infrared-to-visible, official aggregate over gallery trials 0-9
Training budget: 24 epochs, seed 0

## Final status

| Variant | Archival status | Selected epoch | Rank-1 | mAP | mINP |
|---|---|---:|---:|---:|---:|
{chr(10).join(table)}

Eleven variants completed the full schedule. S1 and S2 were explicitly stopped by the
user and are archived as failures (`exit_code=-15`), not as completed experiments.
S1's epoch-21 values are retained only as partial interrupted-run evidence; S2 has no
evaluation result.

## Conclusions

- R0 remains the group best: Rank-1 70.0447%, mAP 68.4326%, mINP 56.7630%.
- M1 is the strongest modified mining/loss recipe: Rank-1 69.2848%. M2 and M3 do not
  improve on M1 or R0, so increasing the cross-modal margin is not supported.
- L4 reaches Rank-1 68.1593%; its reduced DCL recipe is viable but does not beat R0.
- S0 is substantially below the main C3 line. S1 is essentially level with S0 before
  interruption, and S2 was stopped before evaluation. These runs do not support further
  selective Relation Loss hyperparameter search.
- D0 and D1 diagnostics are retained as supporting evidence and are not counted as
  training-result rows.

## Archive layout

- Retained original run root: `{source_root}`
- Completed batch 1 (R0/R1/R2/L0/L1/L2/L4/M1): `{batch1_root}`
- Final batch 2 (M2/M3/S0/S1/S2): `{archive_root}`
- Web-readable Git index: `reports/experiment_archives/c3_recipe_loss_research_20260829/`

Both archive batches preserve the original run tree. Scientific payload copies were
verified byte-for-byte; no new checksum was calculated. The Git index contains the
full final metrics table, terminal scheduler state, experiment plan, diagnostics, and
all compact evaluation event streams. Large model/optimizer files remain in the server
archives and are intentionally not committed to Git.
"""


def launch_command(plan_variant: dict, name: str) -> str:
    fixed = plan_variant.get("base_config") or plan_variant.get("fixed_config")
    overrides = dict(plan_variant.get("overrides") or {})
    overrides.update(plan_variant.get("runtime_overrides") or {})
    parts = ["python", "scripts/train.py", "--config_select", str(fixed)]
    for key, value in sorted(overrides.items()):
        parts.extend(["--set", f"{key}={json.dumps(value, separators=(',', ':'))}"])
    return " ".join(parts)


def find_checkpoint(archive_root: Path, name: str, epoch) -> Path | None:
    if epoch is None:
        return None
    matches = list((archive_root / "runs" / name).rglob(f"model_IR_{epoch}.pth"))
    return matches[0] if matches else None


def update_registry(
    repository: Path,
    rows: list[dict],
    plan: dict,
    archive_root: Path,
    batch1_root: Path,
) -> list[dict]:
    path = repository / "reports" / "experiment_registry" / "experiment_registry.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise RuntimeError("experiment registry has no header")
        registry = list(reader)
    by_id = {row.get("record_id"): row for row in registry}
    base_commit = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    metric_by_name = {row["variant"]: row for row in rows}

    for name in FINAL_VARIANTS:
        metric = metric_by_name[name]
        resolved = plan["variants"][name]
        metrics = {
            "variant": name,
            "phase": metric["phase"],
            "overrides": resolved.get("overrides") or {},
            "runtime_overrides": resolved.get("runtime_overrides") or {},
            "evaluated_epochs": metric["evaluated_epochs"],
            "protocol": metric["protocol"],
            "metrics_scope": metric["metrics_scope"],
            "archive_integrity": "bytewise_copy_verified_no_new_checksum",
        }
        checkpoint = find_checkpoint(archive_root, name, metric["best_epoch"])
        stopped = name in USER_STOPPED_VARIANTS
        record = {key: "" for key in fieldnames}
        record.update(
            {
                "record_id": f"stage_a_c3_recipe_loss_research_20260829#{name}",
                "source_table": "docs/history/stage_a/stage_a_c3_recipe_loss_research_20260831.md",
                "source_original_path": str(archive_root / "provenance" / "archive_manifest.json"),
                "record_type": "result",
                "experiment_id": f"C3-RECIPE-{name.replace('_', '-').upper()}",
                "stage": "stage_a",
                "experiment_group": GROUP_ID,
                "description": resolved.get("purpose", ""),
                "dataset": "SYSU-MM01",
                "evaluation_protocol": "all-search single-shot 10 gallery trials; infrared query to visible gallery; official 10-trial aggregate",
                "modalities": "RGB + IR",
                "seed": "0",
                "code_root": str(repository),
                "code_commit": base_commit,
                "config_path": str(resolved.get("base_config") or resolved.get("fixed_config") or ""),
                "pretrained_path": "/home/cgv841/ybj/SALT-VI/pretrained/pmt_sysu/jx_vit_base_p16_224-80ecf9dd.pth",
                "data_root": "/home/cgv841/datasets/SYSU-MM01/",
                "derived_data_root": "/home/cgv841/datasets/derived/SYSU-MM01-swinir-x2-pmt256-v1",
                "environment_file": str(archive_root / "provenance" / "batch1_source_snapshot" / "environment.json"),
                "launch_command": launch_command(resolved, name),
                "run_dir": str(archive_root / "runs" / name),
                "status": "failed_user_stopped" if stopped else "succeeded",
                "lifecycle": "archived",
                "best_epoch": "" if metric["best_epoch"] is None else str(metric["best_epoch"]),
                "rank1": "" if metric["rank1"] is None else str(metric["rank1"]),
                "mAP": "" if metric["mAP"] is None else str(metric["mAP"]),
                "mINP": "" if metric["mINP"] is None else str(metric["mINP"]),
                "checkpoint_path": str(checkpoint) if checkpoint else "",
                "checkpoint_status": (
                    "retained_interrupted_best" if stopped and checkpoint else
                    "absent_user_stopped_before_evaluation" if stopped else
                    "retained_best"
                ),
                "metrics_source": str(archive_root / "provenance" / "final_metrics.json"),
                "log_source": str(archive_root / "events" / f"{name}.jsonl"),
                "selection_rule": "best evaluated epoch by Rank-1; official 10-trial aggregate",
                "extra_metrics_json": json.dumps(metrics, ensure_ascii=False, separators=(",", ":")),
                "notes": (
                    "User-stopped failure; partial metrics are retained for provenance and must not be treated as a completed result."
                    if stopped and metric["best_epoch"] is not None else
                    "User-stopped failure before the first evaluation; no result metric exists."
                    if stopped else
                    "Single seed. Resolved config, run manifest, best model, optimizer state, events, and launcher log retained."
                ),
                "migration_batch": "20260831_c3_recipe_loss_final_batch2",
                "migration_status": "archive_verified",
                "checkpoint_path_status": "exists" if checkpoint else "not_applicable",
                "checkpoint_path_check": "bytewise_copy_verified" if checkpoint else "user_stopped_no_evaluation",
                "log_source_status": "present",
            }
        )
        by_id[record["record_id"]] = record

    updated = []
    seen = set()
    for record in registry:
        record_id = record.get("record_id")
        if record_id in by_id:
            record = by_id[record_id]
        if record_id and record_id.startswith("stage_a_c3_recipe_loss_research_20260829#"):
            record["source_table"] = "docs/history/stage_a/stage_a_c3_recipe_loss_research_20260831.md"
            if record_id.split("#", 1)[1] not in FINAL_VARIANTS:
                record["notes"] = (
                    "Archived in completed batch 1. The final Git index covers all 13 variants; "
                    "large scientific payload remains in the server archive."
                )
        updated.append(record)
        seen.add(record_id)
    for name in FINAL_VARIANTS:
        record_id = f"stage_a_c3_recipe_loss_research_20260829#{name}"
        if record_id not in seen:
            updated.append(by_id[record_id])

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(updated)
    return [
        row for row in updated
        if row.get("record_id", "").startswith("stage_a_c3_recipe_loss_research_20260829#")
    ]


def write_registry_subset(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_git_index(
    repository: Path,
    source_root: Path,
    archive_root: Path,
    batch1_root: Path,
    archive_result: dict,
) -> None:
    results = load_json(source_root / "results.json")
    plan = load_json(source_root / "plan.json")
    state = load_json(source_root / "scheduler" / "state.json")
    rows = metrics_rows(results, plan)
    report = repository / "reports" / "experiment_archives" / "c3_recipe_loss_research_20260829"
    if report.exists():
        shutil.rmtree(report)
    report.mkdir(parents=True)

    write_json(report / "final_metrics.json", results)
    write_json(report / "scheduler_state.json", state)
    write_json(report / "plan.json", plan)
    write_json(report / "archive_manifest.json", archive_result["manifest"])
    write_json(report / "verification.json", archive_result["verification"])
    write_metrics_csv(report / "final_metrics.csv", rows)
    copy_tree(source_root / "events", report / "events", [])
    copy_tree(source_root / "diagnostics", report / "diagnostics", [])

    history = render_history(rows, source_root, archive_root, batch1_root)
    history_path = repository / "docs" / "history" / "stage_a" / "stage_a_c3_recipe_loss_research_20260831.md"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(history, encoding="utf-8")
    (report / "README.md").write_text(history, encoding="utf-8")

    registry_rows = update_registry(repository, rows, plan, archive_root, batch1_root)
    registry_path = repository / "reports" / "experiment_registry" / "experiment_registry.csv"
    with registry_path.open(newline="", encoding="utf-8") as handle:
        fieldnames = csv.DictReader(handle).fieldnames
    write_registry_subset(report / "registry_rows.csv", fieldnames, registry_rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--batch1-root", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    args = parser.parse_args()
    archive_result = archive_final_batch(
        args.source_root.resolve(),
        args.archive_root.resolve(),
        args.batch1_root.resolve(),
    )
    build_git_index(
        args.repository.resolve(),
        args.source_root.resolve(),
        args.archive_root.resolve(),
        args.batch1_root.resolve(),
        archive_result,
    )
    print(
        json.dumps(
            {
                "archive_root": str(args.archive_root.resolve()),
                "archive_reused": archive_result["reused"],
                "verified_file_count": archive_result["verification"]["verified_file_count"],
                "verified_bytes": archive_result["verification"]["verified_bytes"],
                "git_index": str(
                    args.repository.resolve()
                    / "reports"
                    / "experiment_archives"
                    / "c3_recipe_loss_research_20260829"
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
