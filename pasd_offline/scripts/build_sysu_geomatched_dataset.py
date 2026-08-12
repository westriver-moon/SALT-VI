from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pasd_offline.geomatched import build_geomatched_dataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build geometry-matched SYSU RGB/IR data")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--rgb-root", required=True)
    parser.add_argument("--ir-root", required=True)
    parser.add_argument("--combined-root", required=True)
    args = parser.parse_args()
    result = build_geomatched_dataset(
        args.dataset_root,
        args.rgb_root,
        args.ir_root,
        args.combined_root,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
