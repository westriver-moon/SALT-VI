from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pasd_offline.scheduler import run_dynamic_scheduler  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Dynamically use only idle physical GPUs 1-3")
    parser.add_argument("--config", required=True)
    parser.add_argument("--records", required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--max-workers", type=int, default=3, choices=(1, 2, 3))
    parser.add_argument("--worker-max-sources", type=int)
    args = parser.parse_args()
    result = run_dynamic_scheduler(
        args.config,
        args.records,
        poll_seconds=args.poll_seconds,
        max_workers=args.max_workers,
        worker_max_sources=args.worker_max_sources,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
