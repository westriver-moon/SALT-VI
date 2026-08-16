from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pasd_offline.regdb import build_regdb_records  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build canonical RegDB PASD records")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--rgb-candidates", required=True)
    parser.add_argument("--ir-candidates", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20_260_808)
    parser.add_argument("--trial", type=int, default=1)
    parser.add_argument(
        "--exclude-all",
        action="store_true",
        help="Only include sources listed by the RegDB trial index.",
    )
    args = parser.parse_args()
    records = build_regdb_records(
        args.dataset_root,
        {
            "rgb": args.rgb_candidates,
            "ir": args.ir_candidates,
        },
        args.output,
        seed=args.seed,
        views_per_source=1,
        trial=args.trial,
        include_all=not args.exclude_all,
    )
    print(
        json.dumps(
            {
                "records": len(records),
                "views": len(records),
                "views_per_source": 1,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
