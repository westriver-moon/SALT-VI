"""Inventory model-only checkpoints that legacy resume paths could still load.

The scan is read-only. It records each model_*.pth file together with whether
a full-state checkpoint_latest.pth exists for the same run, so the retirement
decision can be made from evidence instead of deleting blindly.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SEARCH_ROOTS = (
    "checkpoints",
    "logs/raw/experiments",
    "train_outputs",
)


def _has_full_state_nearby(path, repo_root):
    for parent in path.parents:
        candidate = parent / "checkpoint" / "checkpoint_latest.pth"
        if candidate.is_file():
            return True
        if (parent / "checkpoint_latest.pth").is_file():
            return True
        if parent == repo_root:
            break
    return False


def _collect_files(root):
    if not root.is_dir():
        return []
    return [
        path
        for path in root.rglob("model_*.pth")
        if path.is_file() and not any(part == ".git" for part in path.parts)
    ]


def _mode_from_name(path):
    name = path.name
    for mode in ("IR", "Fusion", "Text"):
        if f"_{mode}_" in name:
            return mode
    return "unknown"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)
    repo_root = args.repo_root.resolve()
    files = []
    for root_name in SEARCH_ROOTS:
        files.extend(_collect_files(repo_root / root_name))
    files = sorted(set(files))
    output = args.output or (
        repo_root
        / "reports"
        / "checkpoint_inventory"
        / f"model_only_checkpoints_{datetime.now(timezone.utc):%Y%m%d}.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "checkpoint_path",
                "mode",
                "size_bytes",
                "mtime_utc",
                "has_full_state_nearby",
            ]
        )
        for path in files:
            stat = path.stat()
            writer.writerow(
                [
                    str(path),
                    _mode_from_name(path),
                    stat.st_size,
                    datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(
                        timespec="seconds"
                    ),
                    _has_full_state_nearby(path, repo_root),
                ]
            )
    print(f"Wrote {len(files)} inventory rows to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
