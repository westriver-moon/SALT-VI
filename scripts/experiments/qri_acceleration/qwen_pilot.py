#!/usr/bin/env python3
"""Benchmark one-call QRI planning against the three-round proposer."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
PLUGIN_ROOT = PROJECT_ROOT / "plugins" / "qwen_imagination"
for path in (PROJECT_ROOT, PROJECT_ROOT / "src", PLUGIN_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fast_components import OneShotQwenPlanner  # noqa: E402
from qwen_imagination.regional.qwen_v2 import (  # noqa: E402
    ImaginativeQwenReasoner,
    V2_NON_EDIT_STATES,
)
from qwen_imagination.regional.schema import Region  # noqa: E402


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def load_region(run_root: Path, row: dict) -> Region:
    mask = np.asarray(Image.open(run_root / row["mask"]).convert("L")) > 0
    return Region(
        row["region_id"],
        row["category"],
        tuple(int(value) for value in row["bbox_xyxy"]),
        mask,
        side=row.get("side"),
    )


def coverage(proposals: dict[str, list[dict]]) -> dict[str, float]:
    values = list(proposals.values())
    return {
        "region_count": len(values),
        "positive_region_fraction": sum(
            any(item["state"] not in V2_NON_EDIT_STATES for item in candidates)
            for candidates in values
        )
        / max(1, len(values)),
        "unresolved_region_fraction": sum(
            any(item["state"] == "unresolved" for item in candidates)
            for candidates in values
        )
        / max(1, len(values)),
        "mean_candidate_count": sum(len(candidates) for candidates in values)
        / max(1, len(values)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--reference-limit", type=int, default=2)
    parser.add_argument("--selected-regions", type=int, default=2)
    args = parser.parse_args()
    run_root = args.run_root.expanduser().resolve()
    base = json.loads((run_root / "metrics" / "prepare_pilot.json").read_text())
    planner = OneShotQwenPlanner(args.endpoint, args.model_id)
    reference = ImaginativeQwenReasoner(
        endpoint=args.endpoint,
        model_id=args.model_id,
        timeout_seconds=300,
        enable_thinking=True,
        reasoning_effort="high",
        proposal_rounds=3,
        roi_board_size_px=512,
    )
    output = []
    path = run_root / "metrics" / "qwen_pilot.json"
    for index, row in enumerate(base["records"][: args.limit]):
        lr = Image.open(run_root / row["lr"]).convert("RGB")
        swin = Image.open(run_root / row["swin"]).convert("RGB")
        selected_rows = sorted(
            row["regions"], key=lambda item: (-item["u_blur"], item["region_id"])
        )[: args.selected_regions]
        regions = [load_region(run_root, item) for item in selected_rows]
        tick = time.perf_counter()
        plan = planner.plan(lr, swin, regions, max_worlds=3, seed=index)
        one_seconds = time.perf_counter() - tick
        result = {
            "source_key": row["source_key"],
            "selected_region_ids": [region.region_id for region in regions],
            "one_shot_seconds": one_seconds,
            "one_shot": plan,
            "one_shot_coverage": coverage(plan["regions"]),
        }
        if index < args.reference_limit:
            tick = time.perf_counter()
            proposals = reference.propose(lr, swin, regions)
            reference_seconds = time.perf_counter() - tick
            proposal_payload = {
                region_id: [
                    {
                        "state": item.state,
                        "value": item.value,
                        "evidence": item.evidence,
                        "evidence_source": item.evidence_source,
                    }
                    for item in candidates
                ]
                for region_id, candidates in proposals.items()
            }
            result.update(
                {
                    "reference_seconds": reference_seconds,
                    "reference_proposals": proposal_payload,
                    "reference_coverage": coverage(proposal_payload),
                }
            )
        output.append(result)
        atomic_json(
            path,
            {
                "schema_version": 1,
                "endpoint": args.endpoint,
                "sample_count": len(output),
                "complete": False,
                "records": output,
            },
        )
        print(
            json.dumps(
                {
                    "index": index + 1,
                    "source": row["source_key"],
                    "one_shot_seconds": round(one_seconds, 2),
                    "worlds": len(plan["worlds"]),
                },
                separators=(",", ":"),
            ),
            flush=True,
        )
    payload = {
        "schema_version": 1,
        "endpoint": args.endpoint,
        "sample_count": len(output),
        "complete": True,
        "records": output,
    }
    atomic_json(path, payload)
    print(json.dumps({"metrics": str(path), "sample_count": len(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
