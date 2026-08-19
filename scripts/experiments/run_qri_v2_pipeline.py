#!/usr/bin/env python3
"""Launch the preregistered, isolated QRI-v2 imagination-first Stage-A variants."""

from pathlib import Path

from qri_pipeline_runner import run_registry


REGISTRY = Path(__file__).resolve().parents[2] / "configs/pipelines/sysu_qri_v2.yaml"


if __name__ == "__main__":
    raise SystemExit(run_registry(REGISTRY))
