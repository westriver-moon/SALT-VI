#!/usr/bin/env python3
"""Validate experiment archives by discovery and schema, not fixed run counts."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def read_json(path: Path, errors: List[str], label: str) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        errors.append(f"{label}: missing {path.name}")
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - parser wording varies by runtime.
        errors.append(f"{label}: invalid {path.name}: {exc}")
        return None
    if not isinstance(payload, dict):
        errors.append(f"{label}: {path.name} must contain a JSON object")
        return None
    return payload


def read_csv(path: Path, errors: List[str], label: str) -> List[Dict[str, str]]:
    if not path.is_file():
        errors.append(f"{label}: missing {path.name}")
        return []
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    except Exception as exc:  # pragma: no cover - parser wording varies by runtime.
        errors.append(f"{label}: invalid {path.name}: {exc}")
        return []


def _archive_stage_roots(reports: Path) -> List[Path]:
    roots = []
    if not reports.is_dir():
        return roots
    for root in reports.iterdir():
        runs = root / "runs"
        if not runs.is_dir():
            continue
        if (
            (root / "resolved_plan.yaml").exists()
            or (root / "shared_input_fingerprint.json").exists()
        ):
            roots.append(root)
    return sorted(roots)


def _validate_stage(root: Path, errors: List[str]) -> None:
    label = root.name
    for name in ("environment.json", "shared_input_fingerprint.json", "resolved_plan.yaml"):
        if not (root / name).is_file():
            errors.append(f"{label}: missing shared {name}")
    plan_path = root / "resolved_plan.yaml"
    if plan_path.is_file():
        try:
            yaml.safe_load(plan_path.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - parser wording varies by runtime.
            errors.append(f"{label}: invalid resolved_plan.yaml: {exc}")

    for run in sorted((root / "runs").iterdir()):
        if not run.is_dir():
            continue
        run_label = f"{label}/{run.name}"
        for name in ("environment.json", "dataset_fingerprint.json", "code.patch"):
            if (run / name).exists():
                errors.append(f"{run_label}: redundant {name}")
        if list(run.glob("model_output/**/configs.yaml")):
            errors.append(f"{run_label}: redundant deep configs.yaml")
        for name in ("runtime_config.yaml", "config_diff.yaml", "artifact_hashes.json"):
            if not (run / name).is_file():
                errors.append(f"{run_label}: missing {name}")
        manifest = read_json(run / "manifest.json", errors, run_label)
        if manifest is None:
            continue
        if manifest.get("environment_ref") != "../../environment.json":
            errors.append(f"{run_label}: invalid environment_ref")
        if manifest.get("dataset_fingerprint_ref") != "../../shared_input_fingerprint.json":
            errors.append(f"{run_label}: invalid dataset_fingerprint_ref")

    non_run_manifests = [
        path
        for path in root.rglob("*manifest.json")
        if "runs" not in path.relative_to(root).parts
    ]
    if not non_run_manifests:
        errors.append(f"{label}: missing stage archive manifest")


def _validate_partition_manifest(path: Path, payload: Dict[str, Any], errors: List[str]) -> None:
    keys = ("completed_candidates", "reused_candidates", "excluded_runs")
    if not any(key in payload for key in keys):
        return
    partitions = []
    for key in keys:
        value = payload.get(key, [])
        if not isinstance(value, list):
            errors.append(f"{path}: {key} must be a list")
            value = []
        partitions.append(value)
    id_sets = [{item.get("experiment_id") for item in items if isinstance(item, dict)} for items in partitions]
    if any(id_sets[left] & id_sets[right] for left in range(3) for right in range(left + 1, 3)):
        errors.append(f"{path}: candidate partitions overlap")
    if any(item.get("terminal_status") != "succeeded" for item in partitions[0] if isinstance(item, dict)):
        errors.append(f"{path}: completed candidate is not succeeded")
    if any(item.get("terminal_status") != "stopped_by_user" for item in partitions[2] if isinstance(item, dict)):
        errors.append(f"{path}: excluded run lacks stopped_by_user status")


def _validate_reproduction_manifest(path: Path, payload: Dict[str, Any], errors: List[str]) -> None:
    reproduction = payload.get("metric_reproduction")
    if reproduction is None:
        return
    if not isinstance(reproduction, dict):
        errors.append(f"{path}: metric_reproduction must be an object")
        return
    if reproduction.get("metrics_equal") is not True:
        errors.append(f"{path}: metric equality is not established")
    if reproduction.get("checkpoint_binary_equal") is not False:
        errors.append(f"{path}: checkpoint binary inequality is not explicit")
    left = reproduction.get("stage2_checkpoint_sha256")
    right = reproduction.get("stage3_checkpoint_sha256")
    if left == right:
        errors.append(f"{path}: unequal checkpoint binaries have equal hashes")


def _validate_pruned_checkpoints(reports: Path, errors: List[str]) -> None:
    for status_path in sorted(reports.rglob("status.json")):
        status = read_json(status_path, errors, str(status_path.parent.relative_to(reports)))
        if not status or status.get("checkpoint_retained") is not False:
            continue
        label = str(status_path.parent.relative_to(reports))
        if status.get("checkpoint") is not None:
            errors.append(f"{label}: pruned checkpoint path is still active")
        if not status.get("checkpoint_original_path"):
            errors.append(f"{label}: missing checkpoint_original_path")
        if not SHA256_RE.fullmatch(str(status.get("checkpoint_sha256", ""))):
            errors.append(f"{label}: invalid checkpoint_sha256")
        disposition = status.get("checkpoint_disposition")
        if not status.get("checkpoint_archive_blob") and disposition != "deleted_without_archive":
            errors.append(f"{label}: missing checkpoint_archive_blob")
        if disposition == "deleted_without_archive" and not status.get("checkpoint_prune_ledger"):
            errors.append(f"{label}: deletion without archive lacks a prune ledger")


def _validate_feature_archives(reports: Path, errors: List[str]) -> None:
    feature_root = reports / "feature_domain_gap"
    if not feature_root.is_dir():
        return
    runs = sorted(
        run
        for run in feature_root.iterdir()
        if run.is_dir()
        and any((run / name).exists() for name in ("summary.json", "raw_artifacts.json", "result_tables.md"))
    )
    for run in runs:
        label = f"feature_domain_gap/{run.name}"
        for raw in ("metrics.json", "metrics.csv", "figures"):
            if (run / raw).exists():
                errors.append(f"{label}: raw artifact remains in Git archive: {raw}")
        for name in ("summary.json", "result_tables.md", "selected_figures"):
            if not (run / name).exists():
                errors.append(f"{label}: missing compact artifact {name}")
        raw_manifest = read_json(run / "raw_artifacts.json", errors, label)
        if raw_manifest is None:
            continue
        if raw_manifest.get("schema_version") != 2:
            errors.append(f"{label}: raw artifact manifest must use schema_version 2")
        if not raw_manifest.get("archive_id") or raw_manifest.get("relative_path") != run.name:
            errors.append(f"{label}: portable archive identity is incomplete")
        artifacts = raw_manifest.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            errors.append(f"{label}: external artifact list is incomplete")
            continue
        for artifact in artifacts:
            if not isinstance(artifact, dict) or not SHA256_RE.fullmatch(str(artifact.get("sha256", ""))):
                errors.append(f"{label}: invalid external artifact SHA-256")


def _validate_registry(reports: Path, errors: List[str], check_server_paths: bool) -> None:
    registry = reports / "experiment_registry"
    if not registry.is_dir():
        return
    main_rows = read_csv(registry / "experiment_results.csv", errors, "experiment_registry")
    archived_rows = read_csv(registry / "archived_results.csv", errors, "experiment_registry")

    def key(row: Dict[str, str]) -> Tuple[str, str, str]:
        return row.get("stage", ""), row.get("group", ""), row.get("experiment", "")

    main_by_key: Dict[Tuple[str, str, str], Dict[str, str]] = {}
    for row in main_rows:
        row_key = key(row)
        if row_key in main_by_key:
            errors.append(f"experiment_registry: duplicate result key {row_key}")
        main_by_key[row_key] = row

    shared_fields = (
        "stage",
        "group",
        "experiment",
        "yaml_path",
        "status",
        "best_epoch",
        "rank1",
        "mAP",
        "mINP",
        "checkpoint",
        "source",
        "notes",
    )
    for archived in archived_rows:
        row_key = key(archived)
        generated = main_by_key.get(row_key)
        if generated is None:
            errors.append(f"experiment_registry: archived source row missing from generated results: {row_key}")
            continue
        drift = [field for field in shared_fields if archived.get(field, "") != generated.get(field, "")]
        if drift:
            errors.append(f"experiment_registry: archived/generated drift for {row_key}: {', '.join(drift)}")

    if check_server_paths:
        for row in main_rows:
            checkpoint = row.get("checkpoint", "").strip()
            if checkpoint and not Path(checkpoint).is_file():
                errors.append(
                    "experiment_registry: retained checkpoint is missing for "
                    f"{row.get('experiment', '')}: {checkpoint}"
                )


def validate_archive_contract(
    repo_root: Path = REPO_ROOT,
    check_server_paths: Optional[bool] = None,
) -> List[str]:
    errors: List[str] = []
    repo_root = Path(repo_root)
    reports = repo_root / "reports"
    if not reports.is_dir():
        return ["reports: directory is missing"]
    if check_server_paths is None:
        check_server_paths = Path("/home/cgv841/ybj").is_dir()

    for root in _archive_stage_roots(reports):
        _validate_stage(root, errors)

    for manifest_path in sorted(reports.rglob("*manifest.json")):
        if manifest_path.name == "manifest.json" and "runs" in manifest_path.parts:
            continue
        payload = read_json(manifest_path, errors, str(manifest_path.relative_to(reports)))
        if payload is None:
            continue
        _validate_partition_manifest(manifest_path, payload, errors)
        _validate_reproduction_manifest(manifest_path, payload, errors)

    _validate_pruned_checkpoints(reports, errors)
    _validate_feature_archives(reports, errors)
    _validate_registry(reports, errors, check_server_paths)

    source_path = repo_root / "scripts/analysis/domain_gap_report.py"
    if not source_path.is_file():
        errors.append("domain_gap_report.py: missing")
    elif "mmd_ratio" in source_path.read_text(encoding="utf-8"):
        errors.append("domain_gap_report.py: unstable MMD ratio is still generated")
    return errors


def main() -> None:
    server_checkpoint_paths_checked = Path("/home/cgv841/ybj").is_dir()
    errors = validate_archive_contract(check_server_paths=server_checkpoint_paths_checked)
    if errors:
        print(json.dumps({"status": "failed", "errors": errors}, ensure_ascii=False, indent=2))
        raise SystemExit(1)
    print(
        json.dumps(
            {
                "status": "ok",
                "validation": "discovery-schema-registry-consistency",
                "server_checkpoint_paths_checked": server_checkpoint_paths_checked,
            }
        )
    )


if __name__ == "__main__":
    main()
