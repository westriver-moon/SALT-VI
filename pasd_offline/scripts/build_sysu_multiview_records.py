from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pasd_offline.sysu import build_sysu_multiview_records, select_pilot_records  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build official SYSU five-view PASD records")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--rgb-candidates", required=True)
    parser.add_argument("--ir-candidates", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pilot-output")
    parser.add_argument("--pilot-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20_260_808)
    args = parser.parse_args()
    records = build_sysu_multiview_records(
        args.dataset_root,
        args.rgb_candidates,
        args.ir_candidates,
        args.output,
        seed=args.seed,
    )
    pilot = []
    if args.pilot_output:
        pilot = select_pilot_records(records, args.pilot_output, args.pilot_size, args.seed)
    print(
        json.dumps(
            {"records": len(records), "views": len(records) * 5, "pilot_records": len(pilot)},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
