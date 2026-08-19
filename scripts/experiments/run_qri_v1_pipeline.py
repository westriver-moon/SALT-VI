#!/usr/bin/env python3
"""Launch the preregistered, isolated QRI-v1 Stage-A variants."""

from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from qri_pipeline_runner import (  # noqa: E402
    run_registry,
    validate_base_completion,
    validate_manifest,
)


REGISTRY = Path(__file__).resolve().parents[2] / "configs/pipelines/sysu_qri_v1.yaml"
__all__ = ["validate_base_completion", "validate_manifest"]


if __name__ == "__main__":
    raise SystemExit(run_registry(REGISTRY))
