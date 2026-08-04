#!/usr/bin/env python
"""Prepare or run the three pairwise E4 improvement experiments in parallel."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import resolve_e4_checkpoint
from run_train_sweep import prepare, run_one


DEFAULT_PLAN = REPO_ROOT / "configs/metric_boost/pairwise_improvements.yaml"


def load_plan(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("experiments"), list):
        raise ValueError(f"Invalid pairwise experiment plan: {path}")
    stage = str(payload.get("stage", "PAIRWISE-1"))
    common_overrides = dict(payload.get("common_overrides", {}))
    result: List[Dict[str, Any]] = []
    seen = set()
    for row in payload["experiments"]:
        experiment_id = str(row["id"])
        if experiment_id in seen:
            raise ValueError(f"Duplicate experiment id: {experiment_id}")
        seen.add(experiment_id)
        overrides = dict(common_overrides)
        overrides.update(dict(row.get("overrides", {})))
        required = {"loss_names", "llm_aug", "llm_aug_prob", "id_loss_weight"}
        missing = sorted(required - set(overrides))
        if missing:
            raise ValueError(f"{experiment_id} is missing overrides: {missing}")
        result.append(
            {
                "id": experiment_id,
                "stage": stage,
                "overrides": overrides,
                "validity": "training experiment; pairwise interaction test",
                "dependency": None,
                "description": str(row.get("description", "")),
            }
        )
    if len(result) != 3:
        raise ValueError(f"Expected exactly three pairwise experiments, got {len(result)}")
    return result


def _run_worker(experiment: Dict[str, Any], checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    # gpu_id_override=None deliberately delegates allocation to the existing
    # hardware-idle check plus advisory per-GPU lease.
    return run_one(experiment, checkpoint, gpu_id_override=None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument("--only", action="append", default=[])
    args = parser.parse_args()
    if args.prepare_only == args.run:
        parser.error("Choose exactly one of --prepare-only or --run")
    if not 1 <= args.max_workers <= 3:
        parser.error("--max-workers must be between 1 and 3")

    experiments = load_plan(args.plan.resolve())
    if args.only:
        requested = set(args.only)
        experiments = [item for item in experiments if item["id"] in requested]
        missing = requested - {item["id"] for item in experiments}
        if missing:
            parser.error(f"Unknown experiment ids: {sorted(missing)}")
    checkpoint = resolve_e4_checkpoint()

    if args.prepare_only:
        outputs = [prepare(item, checkpoint) for item in experiments]
    else:
        outputs = []
        failures = []
        with ProcessPoolExecutor(max_workers=min(args.max_workers, len(experiments))) as pool:
            futures = {pool.submit(_run_worker, item, checkpoint): item["id"] for item in experiments}
            for future in as_completed(futures):
                experiment_id = futures[future]
                try:
                    outputs.append(future.result())
                except Exception as exc:
                    failures.append({"experiment": experiment_id, "error": repr(exc)})
        if failures:
            print(json.dumps({"ok": False, "failures": failures}, indent=2, sort_keys=True))
            raise SystemExit(1)

    print(
        json.dumps(
            {
                "ok": True,
                "mode": "run" if args.run else "prepare-only",
                "experiments": [item["id"] for item in experiments],
                "checkpoint": checkpoint["checkpoint"],
                "statuses": [item.get("status") for item in outputs],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
