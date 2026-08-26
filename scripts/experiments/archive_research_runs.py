#!/usr/bin/env python3
"""Reproducible, non-destructive archiving for the 2026-08-21/22 SALT-VI runs.

The tool snapshots the exact dirty worktree, copies completed experiment trees,
adds provenance, hashes every archived file, and leaves the original run intact.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import time
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPEC = PROJECT_ROOT / "configs/experiments/archive_20260822.json"
CHUNK_SIZE = 16 * 1024 * 1024


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def run(command: list[str], *, cwd: Path | None = None, check: bool = True) -> str:
    result = subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        check=check,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout


def git(repo: Path, *args: str, check: bool = True) -> str:
    return run(["git", "-C", str(repo), *args], check=check).strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_regular_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.is_symlink():
            yield path


def write_inventory(root: Path, destination: Path) -> dict[str, Any]:
    destination_abs = destination.resolve(strict=False)
    records: list[str] = []
    total_bytes = 0
    for path in iter_regular_files(root):
        if path.resolve(strict=False) == destination_abs:
            continue
        relative = path.relative_to(root).as_posix()
        records.append(f"{sha256_file(path)}  {relative}")
        total_bytes += path.stat().st_size
    payload = "\n".join(records) + ("\n" if records else "")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(payload, encoding="utf-8")
    return {
        "file_count": len(records),
        "total_bytes_excluding_inventory": total_bytes,
        "inventory_sha256": sha256_text(payload),
    }


def verify_inventory(root: Path, inventory: Path) -> dict[str, Any]:
    failures: list[dict[str, str]] = []
    checked = 0
    for line in inventory.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split("  ", 1)
        path = root / relative
        if not path.is_file():
            failures.append({"path": relative, "error": "missing"})
            continue
        actual = sha256_file(path)
        checked += 1
        if actual != expected:
            failures.append(
                {"path": relative, "expected": expected, "actual": actual}
            )
    return {"ok": not failures, "checked": checked, "failures": failures}


def source_paths(repo: Path) -> list[str]:
    output = subprocess.check_output(
        [
            "git",
            "-C",
            str(repo),
            "ls-files",
            "-c",
            "-o",
            "--exclude-standard",
            "--deduplicate",
            "-z",
        ]
    )
    return sorted(item.decode("utf-8") for item in output.split(b"\0") if item)


def copy_path(source: Path, destination: Path) -> None:
    if source.is_dir():
        shutil.copytree(source, destination, symlinks=True)
    elif source.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination, follow_symlinks=False)
    else:
        raise FileNotFoundError(source)


def directory_stats(path: Path) -> tuple[int, int]:
    count = 0
    size = 0
    for item in iter_regular_files(path):
        count += 1
        size += item.stat().st_size
    return count, size


def tree_digest(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    count = 0
    size = 0
    for item in iter_regular_files(path):
        relative = item.relative_to(path).as_posix()
        if ".git" in item.relative_to(path).parts:
            continue
        item_hash = sha256_file(item)
        digest.update(f"{item_hash}  {relative}\n".encode("utf-8"))
        count += 1
        size += item.stat().st_size
    return {"tree_sha256": digest.hexdigest(), "file_count": count, "bytes": size}


def asset_record(spec: dict[str, Any]) -> dict[str, Any]:
    path = Path(spec["path"])
    record: dict[str, Any] = {
        "id": spec["id"],
        "path": str(path),
        "mode": spec.get("mode", "metadata"),
        "required": bool(spec.get("required", True)),
        "exists": path.exists(),
    }
    if not path.exists():
        if record["required"]:
            raise FileNotFoundError(f"required external asset is missing: {path}")
        return record
    stat = path.stat()
    record.update({"mtime_ns": stat.st_mtime_ns, "is_file": path.is_file()})
    mode = record["mode"]
    if path.is_file():
        record["bytes"] = stat.st_size
        if mode in {"file_sha256", "tree_sha256"}:
            record["sha256"] = sha256_file(path)
    elif mode == "tree_sha256":
        record.update(tree_digest(path))
    else:
        count, size = directory_stats(path)
        record.update({"file_count": count, "bytes": size})
    if mode == "git":
        head = git(path, "rev-parse", "HEAD", check=False)
        status = git(path, "status", "--porcelain=v1", "-uall", check=False)
        if head:
            record["git_head"] = head
            record["git_status_porcelain"] = status
        else:
            record["git_probe_error"] = (
                "directory metadata was captured, but git rev-parse was unavailable "
                "to the archiving account"
            )
    return record


def environment_record(repo: Path) -> dict[str, Any]:
    torch_probe = (
        "import json; "
        "\ntry:\n import torch; "
        "print(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,"
        "'cuda_available':torch.cuda.is_available()}))"
        "\nexcept Exception as e:\n print(json.dumps({'error':repr(e)}))"
    )
    probe = run([sys.executable, "-c", torch_probe], check=False).strip()
    try:
        torch_info = json.loads(probe) if probe else {"error": "empty probe"}
    except json.JSONDecodeError:
        torch_info = {"raw": probe}
    return {
        "captured_at": utc_now(),
        "host": platform.node(),
        "platform": platform.platform(),
        "python_executable": sys.executable,
        "python": sys.version,
        "torch": torch_info,
        "git_version": run(["git", "--version"]).strip(),
        "nvidia_smi": run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            check=False,
        ).strip(),
        "repo": str(repo),
    }


def create_snapshot(spec: dict[str, Any], spec_path: Path) -> Path:
    repo = Path(spec["repo_root"]).resolve(strict=True)
    target = Path(spec["source_snapshot_root"]).resolve(strict=False)
    manifest_path = target / "snapshot_manifest.json"
    if manifest_path.is_file():
        existing = load_json(manifest_path)
        print(json.dumps({"snapshot": str(target), "status": "exists", "digest": existing["source_digest"]}))
        return target
    if target.exists():
        raise RuntimeError(f"snapshot target exists without manifest: {target}")

    stage = target.with_name(f"{target.name}.partial-{os.getpid()}")
    if stage.exists():
        raise RuntimeError(f"staging path already exists: {stage}")
    stage.mkdir(parents=True)
    try:
        tree_root = stage / "source_tree" / repo.name
        tree_root.mkdir(parents=True)
        paths = source_paths(repo)
        source_records: list[str] = []
        for relative in paths:
            source = repo / relative
            destination = tree_root / relative
            copy_path(source, destination)
            if source.is_file():
                source_records.append(
                    f"{sha256_file(source)}  {relative}"
                )
        source_inventory = "\n".join(source_records) + "\n"
        (stage / "source_inventory.sha256").write_text(source_inventory, encoding="utf-8")

        with tarfile.open(stage / f"{repo.name}-source.tar.gz", "w:gz") as archive:
            archive.add(tree_root, arcname=repo.name, recursive=True)

        run(
            ["git", "-C", str(repo), "bundle", "create", str(stage / "repository.bundle"), "--all"]
        )
        (stage / "worktree.patch").write_bytes(
            subprocess.check_output(["git", "-C", str(repo), "diff", "--binary", "HEAD"])
        )
        (stage / "index.patch").write_bytes(
            subprocess.check_output(["git", "-C", str(repo), "diff", "--cached", "--binary", "HEAD"])
        )
        repo_state = {
            "captured_at": utc_now(),
            "branch": git(repo, "branch", "--show-current"),
            "head": git(repo, "rev-parse", "HEAD"),
            "upstream": git(repo, "rev-parse", "@{upstream}", check=False),
            "ahead_behind": git(repo, "rev-list", "--left-right", "--count", "@{upstream}...HEAD", check=False),
            "status_porcelain_v2": git(repo, "status", "--porcelain=v2", "-uall"),
            "tracked_and_untracked_source_files": len(paths),
            "tracked_modified": len(git(repo, "diff", "--name-only").splitlines()),
            "staged": len(git(repo, "diff", "--cached", "--name-only").splitlines()),
            "untracked": len(git(repo, "ls-files", "--others", "--exclude-standard").splitlines()),
        }
        atomic_json(stage / "repo_state.json", repo_state)
        atomic_json(stage / "environment.json", environment_record(repo))
        (stage / "pip-freeze.txt").write_text(
            run([sys.executable, "-m", "pip", "freeze"], check=False), encoding="utf-8"
        )
        asset_records = [asset_record(item) for item in spec.get("external_assets", [])]
        atomic_json(stage / "external_assets.json", asset_records)
        shutil.copy2(spec_path, stage / "archive_spec.json")

        snapshot_manifest = {
            "schema_version": 1,
            "created_at": utc_now(),
            "repo_root": str(repo),
            "snapshot_root": str(target),
            "source_digest": sha256_text(source_inventory),
            "source_file_count": len(source_records),
            "repo_state": repo_state,
            "contents": {
                "source_tree": f"source_tree/{repo.name}",
                "portable_source_tar": f"{repo.name}-source.tar.gz",
                "git_bundle": "repository.bundle",
                "dirty_patch": "worktree.patch",
                "staged_patch": "index.patch",
                "source_inventory": "source_inventory.sha256",
                "external_assets": "external_assets.json",
            },
        }
        atomic_json(stage / "snapshot_manifest.json", snapshot_manifest)
        inventory_meta = write_inventory(stage, stage / "inventory.sha256")
        snapshot_manifest["snapshot_inventory"] = inventory_meta
        atomic_json(stage / "snapshot_manifest.json", snapshot_manifest)
        # Refresh once because snapshot_manifest changed after the first inventory pass.
        inventory_meta = write_inventory(stage, stage / "inventory.sha256")
        snapshot_manifest["snapshot_inventory"] = inventory_meta
        atomic_json(stage / "snapshot_manifest.json", snapshot_manifest)
        write_inventory(stage, stage / "inventory.sha256")

        target.parent.mkdir(parents=True, exist_ok=True)
        stage.replace(target)
        verification = verify_inventory(target, target / "inventory.sha256")
        if not verification["ok"]:
            raise RuntimeError(f"snapshot verification failed: {verification['failures'][:3]}")
        print(json.dumps({"snapshot": str(target), "status": "created", "verification": verification}))
        return target
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        raise


def create_asset_registry(spec: dict[str, Any]) -> Path:
    target = Path(spec["strong_asset_registry_root"]).resolve(strict=False)
    manifest_path = target / "asset_registry_manifest.json"
    if manifest_path.is_file():
        verification = verify_inventory(target, target / "inventory.sha256")
        if not verification["ok"]:
            raise RuntimeError(
                f"existing strong asset registry failed verification: {verification['failures'][:3]}"
            )
        print(
            json.dumps(
                {
                    "asset_registry": str(target),
                    "status": "exists",
                    "verification": verification,
                }
            )
        )
        return target
    if target.exists():
        raise RuntimeError(f"strong asset registry target exists without manifest: {target}")

    stage = target.with_name(f"{target.name}.partial-{os.getpid()}")
    if stage.exists():
        raise RuntimeError(f"asset registry staging path already exists: {stage}")
    stage.mkdir(parents=True)
    try:
        records = [asset_record(item) for item in spec.get("strong_external_assets", [])]
        records_path = stage / "strong_external_assets.json"
        atomic_json(records_path, records)
        missing = [item["id"] for item in records if item.get("required") and not item.get("exists")]
        if missing:
            raise RuntimeError(f"required strong assets are missing: {missing}")
        registry_manifest = {
            "schema_version": 1,
            "created_at": utc_now(),
            "registry_root": str(target),
            "asset_count": len(records),
            "asset_records_sha256": sha256_file(records_path),
            "method": "file SHA-256 or path-sorted tree SHA-256; assets remain in place",
        }
        atomic_json(stage / "asset_registry_manifest.json", registry_manifest)
        write_inventory(stage, stage / "inventory.sha256")
        target.parent.mkdir(parents=True, exist_ok=True)
        stage.replace(target)
        verification = verify_inventory(target, target / "inventory.sha256")
        if not verification["ok"]:
            raise RuntimeError(f"asset registry verification failed: {verification['failures'][:3]}")
        print(
            json.dumps(
                {
                    "asset_registry": str(target),
                    "status": "created",
                    "verification": verification,
                }
            )
        )
        return target
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        raise


def copy_asset_registry(spec: dict[str, Any], destination: Path) -> dict[str, Any] | None:
    value = spec.get("strong_asset_registry_root")
    if not value:
        return None
    registry = Path(value).resolve(strict=False)
    manifest_path = registry / "asset_registry_manifest.json"
    inventory_path = registry / "inventory.sha256"
    records_path = registry / "strong_external_assets.json"
    if not (manifest_path.is_file() and inventory_path.is_file() and records_path.is_file()):
        return None
    verification = verify_inventory(registry, inventory_path)
    if not verification["ok"]:
        raise RuntimeError(f"strong asset registry verification failed: {verification['failures'][:3]}")
    destination.mkdir(parents=True, exist_ok=True)
    sources = [manifest_path, inventory_path, records_path]
    for optional_name in ["owner_asset_fingerprint.json", "owner_asset_probe.py"]:
        optional = registry / optional_name
        if optional.is_file():
            sources.append(optional)
    for source in sources:
        target = destination / source.name
        if not target.exists() or sha256_file(target) != sha256_file(source):
            shutil.copy2(source, target)
    manifest = load_json(manifest_path)
    return {
        "registry_root": str(registry),
        "asset_records_sha256": manifest["asset_records_sha256"],
        "registry_manifest_sha256": sha256_file(manifest_path),
        "copied_manifest": "provenance/shared_snapshot/strong_asset_registry/asset_registry_manifest.json",
        "verification": verification,
    }


def parse_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            events.append(value)
    return events


def event_summary(run_root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for path in sorted(run_root.rglob("*.jsonl")):
        events = parse_events(path)
        evals = [item for item in events if item.get("event_type") == "eval_epoch"]
        trains = [item for item in events if item.get("event_type") == "train_epoch"]
        if not evals and not trains:
            continue
        entry: dict[str, Any] = {
            "event_count": len(events),
            "evaluation_count": len(evals),
            "train_count": len(trains),
        }
        if evals:
            entry["final_evaluation"] = evals[-1]
            entry["best_rank1_evaluation"] = max(
                evals, key=lambda item: float(item.get("metrics", {}).get("Rank-1", float("-inf")))
            )
        if trains:
            entry["final_train"] = trains[-1]
        result[path.relative_to(run_root).as_posix()] = entry
    return result


def latest_tree_mtime(root: Path) -> float:
    return max((path.stat().st_mtime for path in root.rglob("*") if path.is_file()), default=0.0)


def completion_status(experiment: dict[str, Any]) -> dict[str, Any]:
    run_root = Path(experiment["run_root"])
    status: dict[str, Any] = {"ready": True, "reasons": [], "evidence": {}}
    if not run_root.is_dir():
        return {"ready": False, "reasons": [f"run root missing: {run_root}"], "evidence": {}}
    completion = experiment.get("completion", {})
    for pattern in completion.get("required_patterns", []):
        matches = sorted(run_root.glob(pattern))
        status["evidence"].setdefault("required_patterns", {})[pattern] = [
            item.relative_to(run_root).as_posix() for item in matches
        ]
        if not matches:
            status["ready"] = False
            status["reasons"].append(f"required pattern has no match: {pattern}")

    scheduler_path = completion.get("scheduler_state")
    if scheduler_path:
        path = run_root / scheduler_path
        if not path.is_file():
            status["ready"] = False
            status["reasons"].append(f"scheduler state missing: {scheduler_path}")
        else:
            scheduler = load_json(path)
            status["evidence"]["scheduler_state"] = scheduler
            if scheduler.get("status") != "completed":
                status["ready"] = False
                status["reasons"].append(f"scheduler status is {scheduler.get('status')!r}")
            if scheduler.get("failed"):
                status["ready"] = False
                status["reasons"].append("scheduler records failed variants")
            if scheduler.get("running"):
                status["ready"] = False
                status["reasons"].append("scheduler still records running variants")

    event_gate = completion.get("event_gate")
    if event_gate:
        path = run_root / event_gate["path"]
        if not path.is_file():
            status["ready"] = False
            status["reasons"].append(f"event file missing: {event_gate['path']}")
        else:
            events = parse_events(path)
            matching = [item for item in events if item.get("event_type") == event_gate.get("event_type", "eval_epoch")]
            latest_epoch = max((int(item.get("epoch", -1)) for item in matching), default=-1)
            status["evidence"]["event_gate"] = {
                "path": event_gate["path"],
                "event_type": event_gate.get("event_type", "eval_epoch"),
                "latest_epoch": latest_epoch,
                "required_epoch": int(event_gate["minimum_epoch"]),
            }
            if latest_epoch < int(event_gate["minimum_epoch"]):
                status["ready"] = False
                status["reasons"].append(
                    f"latest {event_gate.get('event_type', 'eval_epoch')} epoch {latest_epoch} < {event_gate['minimum_epoch']}"
                )

    quiet_seconds = int(completion.get("quiet_seconds", 0))
    if quiet_seconds:
        age = time.time() - latest_tree_mtime(run_root)
        status["evidence"]["tree_quiet_seconds"] = age
        if age < quiet_seconds:
            status["ready"] = False
            status["reasons"].append(f"run tree quiet for only {age:.1f}s; requires {quiet_seconds}s")
    return status


def evidence_files(root: Path) -> list[dict[str, Any]]:
    names = {"configs.yaml", "run_manifest.json", "summary.json", "metrics.json", "selection.json"}
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name in names:
            records.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return records


def snapshot_source_root(snapshot: Path, repo_name: str) -> Path:
    source = snapshot / "source_tree" / repo_name
    if not source.is_dir():
        raise FileNotFoundError(f"snapshot source tree missing: {source}")
    return source


def archive_experiment(spec: dict[str, Any], experiment_id: str) -> Path:
    experiment = spec["experiments"][experiment_id]
    run_root = Path(experiment["run_root"]).resolve(strict=True)
    target = Path(experiment["archive_root"]).resolve(strict=False)
    if target.exists():
        inventory = target / "provenance/inventory.sha256"
        if not inventory.is_file():
            raise RuntimeError(f"archive exists without inventory: {target}")
        verification = verify_inventory(target, inventory)
        if not verification["ok"]:
            raise RuntimeError(f"existing archive verification failed: {verification['failures'][:3]}")
        print(json.dumps({"experiment": experiment_id, "archive": str(target), "status": "exists", "verification": verification}))
        return target

    completion = completion_status(experiment)
    if not completion["ready"]:
        raise RuntimeError(f"experiment is not archive-ready: {experiment_id}: {completion['reasons']}")
    snapshot = Path(spec["source_snapshot_root"]).resolve(strict=True)
    snapshot_manifest = load_json(snapshot / "snapshot_manifest.json")
    source_root = snapshot_source_root(snapshot, Path(spec["repo_root"]).name)

    target.parent.mkdir(parents=True, exist_ok=True)
    stage = target.with_name(f"{target.name}.partial-{os.getpid()}")
    if stage.exists():
        raise RuntimeError(f"archive staging path already exists: {stage}")
    try:
        shutil.copytree(run_root, stage, symlinks=True, copy_function=shutil.copy2)
        provenance = stage / "provenance"
        provenance.mkdir(parents=True, exist_ok=True)
        selected_root = provenance / "source_files"
        for relative in experiment.get("source_files", []):
            copy_path(source_root / relative, selected_root / relative)
        shared = provenance / "shared_snapshot"
        shared.mkdir(parents=True)
        for name in ["snapshot_manifest.json", "repo_state.json", "environment.json", "pip-freeze.txt", "external_assets.json", "archive_spec.json"]:
            shutil.copy2(snapshot / name, shared / name)
        runtime_spec = Path(spec.get("_spec_path", ""))
        if runtime_spec.is_file():
            shutil.copy2(runtime_spec, shared / "archive_spec_runtime.json")
        shutil.copy2(Path(__file__).resolve(), provenance / "archiver_runtime.py")
        strong_asset_registry = copy_asset_registry(
            spec, shared / "strong_asset_registry"
        )
        for relative in experiment.get("diagnostic_logs", []):
            source = run_root / relative
            if source.is_file():
                copy_path(source, provenance / "launch_diagnostics" / relative)

        metrics = event_summary(run_root)
        atomic_json(provenance / "final_metrics.json", metrics)
        manifest = {
            "schema_version": 3,
            "experiment_id": experiment_id,
            "kind": experiment["kind"],
            "archived_at": utc_now(),
            "archive_method": "copy-verify-preserve-original",
            "original_run_root": str(run_root),
            "archive_root": str(target),
            "original_retained": True,
            "launch_commands": experiment.get("launch_commands", []),
            "completion_evidence": completion,
            "source_snapshot": {
                "path": str(snapshot),
                "source_digest": snapshot_manifest["source_digest"],
                "repo_head": snapshot_manifest["repo_state"]["head"],
                "repo_branch": snapshot_manifest["repo_state"]["branch"],
            },
            "source_files": experiment.get("source_files", []),
            "resolved_config_and_result_files": evidence_files(run_root),
            "metrics_file": "provenance/final_metrics.json",
            "external_assets_file": "provenance/shared_snapshot/external_assets.json",
            "archiver_runtime": {
                "path": "provenance/archiver_runtime.py",
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "runtime_archive_spec": {
                "path": "provenance/shared_snapshot/archive_spec_runtime.json",
                "sha256": sha256_file(runtime_spec) if runtime_spec.is_file() else None,
            },
            "strong_external_assets": strong_asset_registry,
        }
        atomic_json(provenance / "archive_manifest.json", manifest)
        inventory_meta = write_inventory(stage, provenance / "inventory.sha256")
        manifest["inventory"] = inventory_meta
        atomic_json(provenance / "archive_manifest.json", manifest)
        # The manifest changed after inventory creation; regenerate the inventory.
        inventory_meta = write_inventory(stage, provenance / "inventory.sha256")
        manifest["inventory"] = inventory_meta
        atomic_json(provenance / "archive_manifest.json", manifest)
        write_inventory(stage, provenance / "inventory.sha256")

        stage.replace(target)
        verification = verify_inventory(target, target / "provenance/inventory.sha256")
        if not verification["ok"]:
            raise RuntimeError(f"archive verification failed: {verification['failures'][:3]}")
        atomic_json(target / "provenance/verification.json", {"verified_at": utc_now(), **verification})
        # verification.json is intentionally outside inventory; it reports verification of the sealed payload.
        print(json.dumps({"experiment": experiment_id, "archive": str(target), "status": "created", "verification": verification}))
        return target
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        raise


def augment_archive_assets(spec: dict[str, Any], experiment_id: str) -> Path:
    experiment = spec["experiments"][experiment_id]
    root = Path(experiment["archive_root"]).resolve(strict=True)
    provenance = root / "provenance"
    inventory = provenance / "inventory.sha256"
    manifest_path = provenance / "archive_manifest.json"
    if not inventory.is_file() or not manifest_path.is_file():
        raise RuntimeError(f"archive provenance is incomplete: {root}")
    before = verify_inventory(root, inventory)
    if not before["ok"]:
        raise RuntimeError(f"archive failed pre-augmentation verification: {before['failures'][:3]}")

    destination = provenance / "shared_snapshot" / "strong_asset_registry"
    registry_info = copy_asset_registry(spec, destination)
    if registry_info is None:
        raise RuntimeError("strong asset registry has not been created")
    manifest = load_json(manifest_path)
    already_attached = manifest.get("strong_external_assets", {}).get(
        "registry_manifest_sha256"
    ) == registry_info["registry_manifest_sha256"]
    if already_attached:
        print(
            json.dumps(
                {
                    "experiment": experiment_id,
                    "archive": str(root),
                    "status": "strong-assets-exist",
                    "verification": before,
                }
            )
        )
        return root

    verification_path = provenance / "verification.json"
    if verification_path.exists():
        verification_path.unlink()
    shutil.copy2(Path(__file__).resolve(), provenance / "asset_registry_attacher.py")
    manifest["strong_external_assets"] = registry_info
    manifest["strong_assets_attached_at"] = utc_now()
    manifest["asset_registry_attacher"] = {
        "path": "provenance/asset_registry_attacher.py",
        "sha256": sha256_file(Path(__file__).resolve()),
    }
    manifest["inventory"] = {
        "path": "provenance/inventory.sha256",
        "scope": "all sealed payload files except inventory.sha256 and post-check verification.json",
    }
    atomic_json(manifest_path, manifest)
    write_inventory(root, inventory)
    verification = verify_inventory(root, inventory)
    if not verification["ok"]:
        raise RuntimeError(f"archive failed post-augmentation verification: {verification['failures'][:3]}")
    atomic_json(
        verification_path,
        {
            "verified_at": utc_now(),
            "sealed_payload_note": "verification.json is the detached report and is not listed in inventory.sha256",
            **verification,
        },
    )
    print(
        json.dumps(
            {
                "experiment": experiment_id,
                "archive": str(root),
                "status": "strong-assets-attached",
                "verification": verification,
            }
        )
    )
    return root


def inventory_records(path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, relative = line.split("  ", 1)
            records[relative] = digest
    return records


def path_fingerprint(path: Path) -> dict[str, Any]:
    if path.is_file():
        return {"kind": "file", "bytes": path.stat().st_size, "sha256": sha256_file(path)}
    if path.is_dir():
        return {"kind": "directory", **tree_digest(path)}
    return {"kind": "missing"}


def audit_experiment(spec: dict[str, Any], experiment_id: str) -> dict[str, Any]:
    experiment = spec["experiments"][experiment_id]
    original = Path(experiment["run_root"])
    archive = Path(experiment["archive_root"])
    checks: dict[str, Any] = {}
    failures: list[str] = []
    if not original.is_dir():
        failures.append(f"original run root missing: {original}")
    if not archive.is_dir():
        failures.append(f"archive root missing: {archive}")
        return {"ok": False, "failures": failures, "checks": checks}

    provenance = archive / "provenance"
    inventory = provenance / "inventory.sha256"
    if inventory.is_file():
        checks["archive_inventory"] = verify_inventory(archive, inventory)
        if not checks["archive_inventory"]["ok"]:
            failures.append("archive inventory verification failed")
    else:
        failures.append("archive inventory is missing")

    manifest_path = provenance / "archive_manifest.json"
    if not manifest_path.is_file():
        failures.append("archive manifest is missing")
        return {"ok": False, "failures": failures, "checks": checks}
    manifest = load_json(manifest_path)
    checks["manifest_identity"] = {
        "experiment_id": manifest.get("experiment_id"),
        "original_run_root": manifest.get("original_run_root"),
        "archive_root": manifest.get("archive_root"),
        "original_retained": manifest.get("original_retained"),
    }
    if manifest.get("experiment_id") != experiment_id:
        failures.append("archive manifest experiment_id mismatch")
    if manifest.get("original_run_root") != str(original.resolve(strict=False)):
        failures.append("archive manifest original_run_root mismatch")
    if manifest.get("original_retained") is not True:
        failures.append("archive manifest does not assert original retention")
    if not manifest.get("launch_commands"):
        failures.append("launch commands are missing")

    if original.is_dir() and inventory.is_file():
        sealed = inventory_records(inventory)
        original_files = {
            path.relative_to(original).as_posix(): path
            for path in iter_regular_files(original)
        }
        archived_payload = {
            path.relative_to(archive).as_posix(): path
            for path in iter_regular_files(archive)
            if path.relative_to(archive).parts[0] != "provenance"
        }
        missing_from_archive = sorted(set(original_files) - set(archived_payload))
        extra_in_archive = sorted(set(archived_payload) - set(original_files))
        payload_hash_failures: list[str] = []
        for relative, path in original_files.items():
            expected = sealed.get(relative)
            if expected is None or sha256_file(path) != expected:
                payload_hash_failures.append(relative)
        checks["original_payload_copy"] = {
            "original_file_count": len(original_files),
            "archived_payload_file_count": len(archived_payload),
            "missing_from_archive": missing_from_archive,
            "extra_in_archive": extra_in_archive,
            "hash_failures": payload_hash_failures,
        }
        if missing_from_archive or extra_in_archive or payload_hash_failures:
            failures.append("original and archived scientific payload differ")

    archived_experiment = dict(experiment)
    archived_experiment["run_root"] = str(archive)
    archived_completion = dict(experiment.get("completion", {}))
    archived_completion["quiet_seconds"] = 0
    archived_experiment["completion"] = archived_completion
    checks["completion_gate_on_archive"] = completion_status(archived_experiment)
    if not checks["completion_gate_on_archive"]["ready"]:
        failures.append("completion gate fails against archived payload")

    snapshot_value = manifest.get("source_snapshot", {}).get("path")
    if not snapshot_value:
        failures.append("source snapshot reference is missing")
    else:
        snapshot = Path(snapshot_value)
        snapshot_inventory = snapshot / "inventory.sha256"
        if not snapshot_inventory.is_file():
            failures.append("shared source snapshot inventory is missing")
        source_root = snapshot / "source_tree" / Path(spec["repo_root"]).name
        source_checks: dict[str, Any] = {}
        for relative in manifest.get("source_files", []):
            archived_source = provenance / "source_files" / relative
            shared_source = source_root / relative
            archived_fp = path_fingerprint(archived_source)
            shared_fp = path_fingerprint(shared_source)
            source_checks[relative] = {
                "archived": archived_fp,
                "shared_snapshot": shared_fp,
                "match": archived_fp == shared_fp and archived_fp.get("kind") != "missing",
            }
            if not source_checks[relative]["match"]:
                failures.append(f"archived source/config does not match snapshot: {relative}")
        checks["source_and_config_copies"] = source_checks

    evidence_failures: list[str] = []
    for item in manifest.get("resolved_config_and_result_files", []):
        path = archive / item["path"]
        if not path.is_file() or sha256_file(path) != item.get("sha256"):
            evidence_failures.append(item["path"])
    checks["resolved_config_and_result_files"] = {
        "count": len(manifest.get("resolved_config_and_result_files", [])),
        "hash_failures": evidence_failures,
    }
    if evidence_failures:
        failures.append("resolved config/result evidence hash mismatch")
    if not (provenance / "final_metrics.json").is_file():
        failures.append("final_metrics.json is missing")
    if not (provenance / "shared_snapshot/external_assets.json").is_file():
        failures.append("external_assets.json is missing")

    strong = provenance / "shared_snapshot/strong_asset_registry"
    strong_inventory = strong / "inventory.sha256"
    if strong_inventory.is_file():
        checks["strong_asset_registry_copy"] = verify_inventory(strong, strong_inventory)
        if not checks["strong_asset_registry_copy"]["ok"]:
            failures.append("strong asset registry copy verification failed")
    else:
        failures.append("strong asset registry copy is missing")

    if experiment_id == "stage_a_h3_qct_040_e35_20260822":
        diagnostics = provenance / "launch_diagnostics"
        required_logs = ["train.log", "train_e35.log", "train_e35_retry.log"]
        missing_logs = [name for name in required_logs if not (diagnostics / name).is_file()]
        checks["launch_diagnostics"] = {"required": required_logs, "missing": missing_logs}
        if missing_logs:
            failures.append("H3-35 launch diagnostics are incomplete")

    return {"ok": not failures, "failures": failures, "checks": checks}


def audit_all(spec: dict[str, Any], ids: list[str], output: Path | None) -> int:
    snapshot = Path(spec["source_snapshot_root"])
    asset_registry = Path(spec["strong_asset_registry_root"])
    result: dict[str, Any] = {
        "schema_version": 1,
        "audited_at": utc_now(),
        "experiments": {},
        "shared_source_snapshot": verify_inventory(
            snapshot, snapshot / "inventory.sha256"
        ) if (snapshot / "inventory.sha256").is_file() else {"ok": False, "error": "missing"},
        "strong_asset_registry": verify_inventory(
            asset_registry, asset_registry / "inventory.sha256"
        ) if (asset_registry / "inventory.sha256").is_file() else {"ok": False, "error": "missing"},
    }
    for experiment_id in ids:
        result["experiments"][experiment_id] = audit_experiment(spec, experiment_id)
    result["ok"] = (
        result["shared_source_snapshot"].get("ok", False)
        and result["strong_asset_registry"].get("ok", False)
        and all(item["ok"] for item in result["experiments"].values())
    )
    if output:
        atomic_json(output.resolve(strict=False), result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["ok"] else 1


def selected_ids(spec: dict[str, Any], values: list[str] | None) -> list[str]:
    if not values or values == ["all"]:
        return list(spec["experiments"])
    unknown = sorted(set(values) - set(spec["experiments"]))
    if unknown:
        raise KeyError(f"unknown experiment ids: {unknown}")
    return values


def status_command(spec: dict[str, Any], ids: list[str]) -> int:
    result: dict[str, Any] = {}
    for experiment_id in ids:
        experiment = spec["experiments"][experiment_id]
        target = Path(experiment["archive_root"])
        if target.is_dir():
            result[experiment_id] = {"archived": True, "archive_root": str(target)}
        else:
            result[experiment_id] = {"archived": False, **completion_status(experiment)}
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def watch(spec: dict[str, Any], ids: list[str], poll_seconds: int, max_hours: float) -> int:
    deadline = time.monotonic() + max_hours * 3600
    pending = set(ids)
    while pending:
        for experiment_id in list(pending):
            experiment = spec["experiments"][experiment_id]
            if Path(experiment["archive_root"]).is_dir():
                pending.remove(experiment_id)
                continue
            status = completion_status(experiment)
            print(json.dumps({"checked_at": utc_now(), "experiment": experiment_id, **status}), flush=True)
            if status["ready"]:
                archive_experiment(spec, experiment_id)
                pending.remove(experiment_id)
        if not pending:
            break
        if time.monotonic() >= deadline:
            print(json.dumps({"status": "timeout", "pending": sorted(pending)}), flush=True)
            return 2
        time.sleep(poll_seconds)
    print(json.dumps({"status": "complete", "experiments": ids}), flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("snapshot")
    subparsers.add_parser("assets")
    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--id", action="append", dest="ids")
    archive_parser = subparsers.add_parser("archive")
    archive_parser.add_argument("--id", action="append", dest="ids", required=True)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--id", action="append", dest="ids")
    augment_parser = subparsers.add_parser("augment-assets")
    augment_parser.add_argument("--id", action="append", dest="ids")
    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("--id", action="append", dest="ids")
    audit_parser.add_argument("--output", type=Path)
    watch_parser = subparsers.add_parser("watch")
    watch_parser.add_argument("--id", action="append", dest="ids")
    watch_parser.add_argument("--poll-seconds", type=int, default=300)
    watch_parser.add_argument("--max-hours", type=float, default=36.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    spec_path = args.spec.resolve(strict=True)
    spec = load_json(spec_path)
    spec["_spec_path"] = str(spec_path)
    if args.command == "snapshot":
        create_snapshot(spec, spec_path)
        return 0
    if args.command == "assets":
        create_asset_registry(spec)
        return 0
    ids = selected_ids(spec, getattr(args, "ids", None))
    if args.command == "status":
        return status_command(spec, ids)
    if args.command == "archive":
        for experiment_id in ids:
            archive_experiment(spec, experiment_id)
        return 0
    if args.command == "verify":
        failed = False
        for experiment_id in ids:
            root = Path(spec["experiments"][experiment_id]["archive_root"])
            inventory = root / "provenance/inventory.sha256"
            result = verify_inventory(root, inventory) if inventory.is_file() else {"ok": False, "error": "inventory missing"}
            print(json.dumps({"experiment": experiment_id, **result}))
            failed = failed or not result.get("ok", False)
        return 1 if failed else 0
    if args.command == "augment-assets":
        for experiment_id in ids:
            augment_archive_assets(spec, experiment_id)
        return 0
    if args.command == "audit":
        return audit_all(spec, ids, args.output)
    if args.command == "watch":
        return watch(spec, ids, args.poll_seconds, args.max_hours)
    parser.error(f"unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
