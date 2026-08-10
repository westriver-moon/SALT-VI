#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys


FEATURE_ROOT = Path(__file__).resolve().parents[1]
if str(FEATURE_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(FEATURE_ROOT / "src"))

from salt_feature_analysis.generalization import render_generalization_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Render train-vs-test generalization summary")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", default=str(FEATURE_ROOT / "artifacts"))
    args = parser.parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    result = render_generalization_summary(
        output_root / "tables" / args.run_id / "comparison_summary.csv",
        output_root / "tables" / args.run_id / "generalization_summary.csv",
        output_root / "figures" / args.run_id / "generalization_summary.png",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
