"""Source-integrity check used by the canonical SYSU SwinIR builder."""

from pathlib import Path
import subprocess


ALGORITHM_SOURCE_PATHS = (
    "src/salt_vi/entrypoints/train.py",
    "src/salt_vi/engine",
    "configs",
    "src/salt_vi/data",
    "src/salt_vi/models",
    "src/salt_vi/optim",
    "scripts/vision_text/super_resolution",
    "src/salt_vi/utils/super_resolution",
)


def _git(repo_root, *args):
    return subprocess.check_output(
        ["git", "-C", str(repo_root), *args], text=True
    ).strip()


def assert_clean_algorithm_source(repo_root):
    """Reject tracked, staged, or untracked algorithm changes before a build."""
    repo_root = Path(repo_root).resolve()
    checks = (
        ("diff", "--name-only", "--", *ALGORITHM_SOURCE_PATHS),
        ("diff", "--cached", "--name-only", "--", *ALGORITHM_SOURCE_PATHS),
        ("ls-files", "--others", "--exclude-standard", "--", *ALGORITHM_SOURCE_PATHS),
    )
    labels = ("tracked worktree", "staged", "untracked")
    dirty = {
        label: paths
        for label, args in zip(labels, checks)
        if (paths := _git(repo_root, *args).splitlines())
    }
    if dirty:
        details = "; ".join(f"{label}: {paths}" for label, paths in dirty.items())
        raise RuntimeError(f"Algorithm source must be clean: {details}")
    return _git(repo_root, "rev-parse", "HEAD")
