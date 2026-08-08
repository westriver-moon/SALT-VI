from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pasd_offline.sysu import build_sysu_records  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build PASD input JSONL from SYSU caption assets")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--caption-dict", action="append", required=True)
    parser.add_argument("--identity-caption-map", action="append")
    parser.add_argument("--caption-scope", choices=("image", "identity"), default="image")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    count = build_sysu_records(
        dataset_root=args.dataset_root,
        caption_dicts=args.caption_dict,
        identity_caption_maps=args.identity_caption_map,
        caption_scope=args.caption_scope,
        output_path=args.output,
    )
    print(f"records={count} output={Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
