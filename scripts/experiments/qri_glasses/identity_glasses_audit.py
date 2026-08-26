#!/usr/bin/env python3
"""Audit ReID identity preservation for localized eyewear candidates."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
QRI_ACCELERATION = PROJECT_ROOT / "scripts" / "experiments" / "qri_acceleration"
for path in (PROJECT_ROOT, PROJECT_ROOT / "src", QRI_ACCELERATION):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from quality_audit import identity_feature, load_identity_model  # noqa: E402


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--identity-config", required=True, type=Path)
    parser.add_argument("--identity-checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    device = torch.device(args.device)
    model, config = load_identity_model(
        args.identity_config, args.identity_checkpoint, device
    )
    reference = Image.open(args.reference).convert("RGB")
    reference_feature = identity_feature(model, reference, "rgb", config, device)

    records = []
    for row in metrics["records"]:
        candidate = Image.open(row["composite"]).convert("RGB")
        feature = identity_feature(model, candidate, "rgb", config, device)
        cosine = float(
            torch.nn.functional.cosine_similarity(reference_feature, feature).item()
        )
        records.append(
            {
                "variant": row["variant"]["name"],
                "seed": row["seed"],
                "composite": row["composite"],
                "identity_cosine_to_swin": cosine,
            }
        )
        print(
            json.dumps(
                {
                    "variant": row["variant"]["name"],
                    "seed": row["seed"],
                    "identity_cosine": round(cosine, 6),
                },
                separators=(",", ":"),
            ),
            flush=True,
        )

    payload = {
        "schema_version": 1,
        "reference": str(args.reference),
        "identity_config": str(args.identity_config),
        "identity_checkpoint": str(args.identity_checkpoint),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, payload)
    print(json.dumps({"metrics": str(args.output), "record_count": len(records)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
