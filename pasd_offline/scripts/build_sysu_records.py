from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pasd_offline.sysu import build_sysu_records, select_pilot_records  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build canonical SYSU PASD records")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--rgb-candidates", required=True)
    parser.add_argument("--ir-candidates")
    parser.add_argument("--output", required=True)
    parser.add_argument("--pilot-output")
    parser.add_argument("--pilot-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20_260_808)
    parser.add_argument("--views-per-source", type=int, choices=(1, 5), default=1)
    args = parser.parse_args()
    records = build_sysu_records(
        args.dataset_root,
        {
            modality: path
            for modality, path in (("rgb", args.rgb_candidates), ("ir", args.ir_candidates))
            if path
        },
        args.output,
        seed=args.seed,
        views_per_source=args.views_per_source,
    )
    pilot = (
        select_pilot_records(records, args.pilot_output, args.pilot_size, args.seed)
        if args.pilot_output
        else []
    )
    print(json.dumps({
        "records": len(records),
        "views": len(records) * args.views_per_source,
        "views_per_source": args.views_per_source,
        "pilot_records": len(pilot),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
