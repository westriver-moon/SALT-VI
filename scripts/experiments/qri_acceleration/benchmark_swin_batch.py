#!/usr/bin/env python3
"""Benchmark official SwinIR batches on deterministic QRI pilot inputs."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
for path in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from salt_vi.utils.super_resolution.build_sysu_swinir_x2 import (  # noqa: E402
    infer,
    load_swinir,
)


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--swinir-root", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-sizes", default="1,4,8,16,32")
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    base = json.loads((run_root / "metrics" / "prepare_pilot.json").read_text())
    samples = {}
    for modality in ("rgb", "ir"):
        rows = [item for item in base["records"] if item["modality"] == modality]
        samples[modality] = [
            np.asarray(Image.open(run_root / item["lr"]).convert("RGB"), dtype=np.uint8)
            for item in rows
        ]
    model, implementation = load_swinir(args.swinir_root, args.model, args.device)
    infer(model, np.stack(samples["rgb"][:1]), "rgb", args.device)

    records = []
    for modality in ("rgb", "ir"):
        source = samples[modality]
        for batch_size in [int(item) for item in args.batch_sizes.split(",")]:
            images = np.stack([source[index % len(source)] for index in range(batch_size)])
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            output = infer(model, images, modality, args.device)
            elapsed = time.perf_counter() - started
            record = {
                "modality": modality,
                "batch_size": batch_size,
                "seconds": elapsed,
                "seconds_per_image": elapsed / batch_size,
                "peak_allocated_mib": torch.cuda.max_memory_allocated() / (1024**2),
                "output_shape": list(output.shape),
            }
            records.append(record)
            print(json.dumps(record, separators=(",", ":")), flush=True)
    payload = {
        "schema_version": 1,
        "device": args.device,
        "implementation": implementation,
        "records": records,
    }
    path = run_root / "metrics" / "swin_batch.json"
    atomic_json(path, payload)
    print(json.dumps({"metrics": str(path), "record_count": len(records)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
