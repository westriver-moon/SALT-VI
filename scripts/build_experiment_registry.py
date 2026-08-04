#!/usr/bin/env python3
"""Build a lightweight experiment registry for TVI-LFM consolidation.

The registry keeps human-readable and machine-readable records of available
experiment results without committing bulky train logs or checkpoints.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "reports" / "experiment_registry"
ARCHIVED_RESULTS = OUT / "archived_results.csv"
PRUNED_STAGE_A_GROUP_RUNS = {
    "A0_RN50_ORI",
    "A1_PMT_VIT",
}
PRUNED_STAGE_A_RECIPE_RUNS = {
    "pmt_recipe_288_768_no_projection",
    "pmt_recipe_288_2048_projection",
    "pmt_recipe_256_2048_projection",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def sha1_file(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def metric(value: object) -> str:
    if value in ("", None):
        return ""
    try:
        return f"{float(value):.5f}"
    except (TypeError, ValueError):
        return str(value)


def percent_to_ratio(value: object) -> str:
    if value in ("", None):
        return ""
    try:
        return f"{float(value) / 100.0:.5f}"
    except (TypeError, ValueError):
        return str(value)


def config_lifecycle(yaml_path: object) -> str:
    value = str(yaml_path or "")
    if not value:
        return "retired"
    path = REPO / value
    if not path.is_file():
        return "retired"
    if "archived_configs" in path.parts:
        return "archived"
    return "active"


def git(*args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if check and result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def yaml_inventory() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    full_paths = sorted(
        [
            *(REPO / "config" / "stage_a").rglob("*.yaml"),
            *(REPO / "config" / "stage_b").rglob("*.yaml"),
            *(OUT / "archived_configs").rglob("*.yaml"),
        ]
    )
    for full in full_paths:
        path = full.relative_to(REPO)
        try:
            data = yaml.safe_load(full.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError:
            # Some preserved legacy launch snapshots serialize EasyDict values
            # with Python-specific YAML tags.  Keep those snapshots indexed for
            # provenance, while leaving optional normalized fields blank.
            data = {}
        if path.parts[:2] == ("config", "stage_a"):
            stage = "stage_a"
        elif path.parts[:2] == ("config", "stage_b"):
            stage = "stage_b"
        else:
            stage = "archive"
        rows.append(
            {
                "yaml_path": path.as_posix(),
                "sha1": sha1_file(full),
                "stage": stage,
                "lifecycle": "archived" if stage == "archive" else "active",
                "output_path": data.get("output_path", ""),
                "training_mode": data.get("training_mode", ""),
                "test_modality": data.get("test_modality", ""),
                "fusion_way": data.get("fusion_way", ""),
                "pa": data.get("pa", ""),
                "total_train_epoch": data.get("total_train_epoch", ""),
            }
        )
    return rows


def archived_result_rows() -> list[dict[str, object]]:
    """Load finalized results whose raw runs live outside the Git worktree."""
    rows = []
    for row in read_csv(ARCHIVED_RESULTS):
        rows.append(
            {
                "stage": row.get("stage", "stage_b"),
                "group": row.get("group", "archived"),
                "experiment": row.get("experiment", ""),
                "yaml_path": row.get("yaml_path", ""),
                "status": row.get("status", "succeeded"),
                "best_epoch": row.get("best_epoch", ""),
                "rank1": metric(row.get("rank1")),
                "mAP": metric(row.get("mAP")),
                "mINP": metric(row.get("mINP")),
                "checkpoint": row.get("checkpoint", ""),
                "source": row.get("source", ""),
                "notes": row.get("notes", ""),
            }
        )
    return rows


def stage_a_group_rows() -> list[dict[str, object]]:
    mapping = {
        "A0_RN50_ORI": {
            "yaml": "config/stage_a/rn50_ori_stage_a_control.yaml",
            "checkpoint": "logs/stage_a_rn50_ori_control/sysu/Base/Baseline_train[RGB_IR]_wrt,id/models/model_IR_107.pth",
        },
        "A1_PMT_VIT": {
            "yaml": "config/stage_a/pmt_vit_stage_a.yaml",
            "checkpoint": "logs/stage_a_pmt_vit/sysu/Base/Baseline_train[RGB_IR]_wrt,id/models/model_IR_31.pth",
        },
    }
    rows = []
    for row in read_csv(REPO / "train_outputs" / "stage_a_group_current" / "stage_a_a0_a1_summary_metrics.csv"):
        run = row.get("run", "")
        best_epoch = row.get("best_epoch", "")
        mapped = mapping.get(run, {})
        checkpoint = mapped.get("checkpoint", "")
        notes = "A0 was intentionally stopped before 120 epochs" if run == "A0_RN50_ORI" else ""
        if run in PRUNED_STAGE_A_GROUP_RUNS:
            checkpoint = ""
            notes = f"{notes + '; ' if notes else ''}checkpoint pruned, metrics-only record"
        rows.append(
            {
                "stage": "stage_a",
                "group": "stage_a_group_current",
                "experiment": run,
                "yaml_path": mapped.get("yaml", ""),
                "status": "done" if run == "A1_PMT_VIT" else "stopped_by_user",
                "best_epoch": best_epoch,
                "rank1": percent_to_ratio(row.get("best_rank1_pct")),
                "mAP": percent_to_ratio(row.get("best_mAP_pct")),
                "mINP": percent_to_ratio(row.get("best_mINP_pct")),
                "checkpoint": checkpoint,
                "source": "train_outputs/stage_a_group_current/stage_a_a0_a1_summary_metrics.csv",
                "notes": notes,
            }
        )
    return rows


def stage_a_recipe_rows() -> list[dict[str, object]]:
    mapping = {
        "pmt_recipe_288_768_mbpatch_30": "config/stage_a/pmt_vit_stage_a_pmt_recipe_288x144_768_mbpatch.yaml",
        "pmt_recipe_288_768_no_projection": "config/stage_a/pmt_vit_stage_a_pmt_recipe_288x144_768.yaml",
        "pmt_recipe_288_2048_projection": "config/stage_a/pmt_vit_stage_a_pmt_recipe_288x144.yaml",
        "pmt_recipe_256_2048_projection": "config/stage_a/pmt_vit_stage_a_pmt_recipe_256x128.yaml",
    }
    rows = []
    for row in read_csv(REPO / "train_outputs" / "stage_a_pmt_recipe_mbpatch_30" / "mbpatch_summary_metrics.csv"):
        run = row.get("run_id", "")
        best_epoch = row.get("best_rank1_epoch", "")
        checkpoint = (row.get("checkpoints", "").strip("[]").replace("'", "").split(",")[0]).strip()
        notes = row.get("label", "")
        if run in PRUNED_STAGE_A_RECIPE_RUNS:
            checkpoint = ""
            notes = f"{notes}; checkpoint pruned, metrics-only record"
        rows.append(
            {
                "stage": "stage_a",
                "group": "pmt_recipe_size_projection_mbpatch",
                "experiment": run,
                "yaml_path": mapping.get(run, ""),
                "status": "done",
                "best_epoch": best_epoch,
                "rank1": metric(row.get("best_rank1")),
                "mAP": metric(row.get("best_rank1_mAP")),
                "mINP": metric(row.get("best_rank1_mINP")),
                "checkpoint": checkpoint,
                "source": "train_outputs/stage_a_pmt_recipe_mbpatch_30/mbpatch_summary_metrics.csv",
                "notes": notes,
            }
        )
    return rows


def branch_audit() -> list[dict[str, object]]:
    rows = []
    current_main = git("rev-parse", "origin/main")
    effective_ref = "origin/main"
    effective = git("rev-parse", effective_ref)
    local_lines = git(
        "for-each-ref",
        "--format=%(refname:short)|%(objectname:short)|%(subject)",
        "refs/heads",
    ).splitlines()
    worktree_map = {}
    current_path = ""
    for line in git("worktree", "list", "--porcelain").splitlines():
        if line.startswith("worktree "):
            current_path = line.split(" ", 1)[1]
        elif line.startswith("branch "):
            worktree_map[line.split("refs/heads/", 1)[-1]] = current_path

    for line in local_lines:
        branch, sha, subject = line.split("|", 2)
        full_ref = f"refs/heads/{branch}"
        is_ancestor_main = subprocess.run(
            ["git", "merge-base", "--is-ancestor", full_ref, "origin/main"],
            cwd=REPO,
        ).returncode == 0
        is_ancestor_effective = subprocess.run(
            ["git", "merge-base", "--is-ancestor", full_ref, effective_ref],
            cwd=REPO,
        ).returncode == 0
        if branch == "main":
            action = "keep as default branch"
        elif branch.startswith("autoresearch/"):
            action = "archive tag, then delete local worktree/branch"
        elif branch == "codex/tvilfm-vit-stageb-push":
            action = "local Stage B working branch; archive tag, then delete after main contains effective r6 work"
        elif branch.startswith("backup/"):
            action = "keep as backup tag/branch unless user wants full pruning"
        else:
            action = "manual review"
        rows.append(
            {
                "branch": branch,
                "sha": sha,
                "subject": subject,
                "worktree": worktree_map.get(branch, ""),
                "ancestor_of_origin_main": str(is_ancestor_main).lower(),
                "ancestor_of_effective_branch": str(is_ancestor_effective).lower(),
                "recommended_action": action,
                "origin_main": current_main[:12],
                "effective_branch": effective[:12],
            }
        )
    return rows


def write_report(result_rows: list[dict[str, object]], yaml_rows: list[dict[str, object]], branch_rows: list[dict[str, object]]) -> None:
    sorted_results = sorted(result_rows, key=lambda row: (row["stage"], row["group"], str(row["experiment"])))
    best_stage_a = max(
        [row for row in sorted_results if row["stage"] == "stage_a" and row["rank1"]],
        key=lambda row: float(row["rank1"]),
        default=None,
    )
    best_stage_b = max(
        [row for row in sorted_results if row["stage"] == "stage_b" and row["rank1"]],
        key=lambda row: float(row["rank1"]),
        default=None,
    )
    lines = [
        "# TVI-LFM Experiment Registry",
        "",
        f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "- Purpose: preserve the legacy Stage-A/early Stage-B subset and selected later experiment records with retained checkpoints.",
        "- Scope: this table does **not** include the full metric-boost, A3-E4 HPT Stage-2/3, multistage-text, or RegDB leaderboards. The archived Qwen grid/pair-equal and SALT ablation entries are included as verified SYSU records; their source experiment directories remain authoritative.",
        "- Bulky logs/checkpoints are intentionally not copied into Git. A non-empty checkpoint field means that the server-side file is currently retained; an empty field is a metrics-only record whose checkpoint was pruned.",
        "",
        "## Highlights",
        "",
    ]
    if best_stage_a:
        best_stage_a_epoch = best_stage_a["best_epoch"] or "not recorded"
        lines.append(
            f"- Best Rank-1 within this legacy Stage-A subset: `{best_stage_a['experiment']}` epoch `{best_stage_a_epoch}`, "
            f"Rank-1 `{best_stage_a['rank1']}`, mAP `{best_stage_a['mAP']}`, mINP `{best_stage_a['mINP']}`."
        )
    if best_stage_b:
        lines.append(
            f"- Best Rank-1 among recorded Stage-B rows: `{best_stage_b['experiment']}` epoch `{best_stage_b['best_epoch']}`, "
            f"Rank-1 `{best_stage_b['rank1']}`, mAP `{best_stage_b['mAP']}`, mINP `{best_stage_b['mINP']}`."
        )
    lines.append(
        "- The archived Qwen and SALT results are evidence only and are not promoted as the active default training recipe; "
        "the Stage-3 `PAIR-EQUAL` provenance record remains the active mainline reference."
    )
    lines.extend(
        [
            f"- YAML configs retained in this branch: `{len(yaml_rows)}`.",
            f"- Local branches audited: `{len(branch_rows)}`.",
            "",
            "## Result Table",
            "",
            "| stage | group | experiment | YAML | lifecycle | status | best_epoch | Rank-1 | mAP | mINP |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in sorted_results:
        lines.append(
            f"| {row['stage']} | {row['group']} | {row['experiment']} | `{row['yaml_path']}` | "
            f"{row['config_lifecycle']} | "
            f"{row['status']} | {row['best_epoch']} | {row['rank1']} | {row['mAP']} | {row['mINP']} |"
        )
    lines.extend(
        [
            "",
            "## Branch Consolidation Notes",
            "",
            "| branch | sha | worktree | recommended_action |",
            "| --- | --- | --- | --- |",
        ]
    )
    for row in branch_rows:
        lines.append(
            f"| `{row['branch']}` | `{row['sha']}` | `{row['worktree']}` | {row['recommended_action']} |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `experiment_results.csv`: generated normalized view of this registry subset; it is not the global project leaderboard.",
            "- `yaml_inventory.csv`: existing active/archived YAML configs with lifecycle, SHA-1, and key fields.",
            "- `archived_results.csv`: source/input rows imported from completed runs outside the main worktree. Its rows intentionally reappear in the generated `experiment_results.csv`; the validator enforces equality to prevent drift.",
            "- `archived_configs/`: YAML snapshots for completed external runs.",
            "- `branch_audit.csv`: local branch/worktree audit used for consolidation.",
        ]
    )
    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    yaml_rows = yaml_inventory()
    result_rows = [
        *stage_a_group_rows(),
        *stage_a_recipe_rows(),
        *archived_result_rows(),
    ]
    for row in result_rows:
        row["config_lifecycle"] = config_lifecycle(row.get("yaml_path"))
    branch_rows = branch_audit()
    write_csv(
        OUT / "experiment_results.csv",
        result_rows,
        [
            "stage",
            "group",
            "experiment",
            "yaml_path",
            "config_lifecycle",
            "status",
            "best_epoch",
            "rank1",
            "mAP",
            "mINP",
            "checkpoint",
            "source",
            "notes",
        ],
    )
    write_csv(
        OUT / "yaml_inventory.csv",
        yaml_rows,
        [
            "yaml_path",
            "sha1",
            "stage",
            "lifecycle",
            "output_path",
            "training_mode",
            "test_modality",
            "fusion_way",
            "pa",
            "total_train_epoch",
        ],
    )
    write_csv(
        OUT / "branch_audit.csv",
        branch_rows,
        [
            "branch",
            "sha",
            "subject",
            "worktree",
            "ancestor_of_origin_main",
            "ancestor_of_effective_branch",
            "recommended_action",
            "origin_main",
            "effective_branch",
        ],
    )
    write_report(result_rows, yaml_rows, branch_rows)
    print(OUT / "README.md")
    print(OUT / "experiment_results.csv")
    print(OUT / "yaml_inventory.csv")
    print(OUT / "branch_audit.csv")


if __name__ == "__main__":
    main()
