from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pasd_offline.scheduler import run_dynamic_scheduler  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an offline PASD dataset from caption JSONL")
    parser.add_argument("--config", required=True)
    parser.add_argument("--records", required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--worker-max-sources", type=int)
    args = parser.parse_args()

    result = run_dynamic_scheduler(
        args.config,
        args.records,
        poll_seconds=args.poll_seconds,
        max_workers=args.workers,
        worker_max_sources=args.worker_max_sources,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
