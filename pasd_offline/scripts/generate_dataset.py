from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pasd_offline.config import GenerationConfig  # noqa: E402
from pasd_offline.contracts import prepare_contracts  # noqa: E402
from pasd_offline.generate import generate_batch  # noqa: E402
from pasd_offline.tasks import load_tasks  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an offline PASD dataset from caption JSONL")
    parser.add_argument("--config", required=True)
    parser.add_argument("--records", required=True)
    parser.add_argument("--caption-mode", choices=("first", "random", "all"), default="first")
    parser.add_argument("--caption-pool")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    config = GenerationConfig.from_yaml(args.config)
    seed = config.seed if args.seed is None else args.seed
    tasks = load_tasks(args.records, args.caption_mode, seed, args.caption_pool)
    prepare_contracts(config, args.records, tasks)
    entries = generate_batch(config, tasks, args.records)
    print(f"generated={len(entries)} output_root={config.output_root}")


if __name__ == "__main__":
    main()
