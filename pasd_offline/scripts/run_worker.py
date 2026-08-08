from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pasd_offline.config import GenerationConfig  # noqa: E402
from pasd_offline.generate import generate_worker  # noqa: E402
from pasd_offline.scheduler import has_foreign_compute_process  # noqa: E402
from pasd_offline.tasks import load_tasks  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one resumable PASD worker on an allowed GPU")
    parser.add_argument("--config", required=True)
    parser.add_argument("--records", required=True)
    parser.add_argument("--physical-gpu", required=True, type=int, choices=(1, 2, 3))
    parser.add_argument("--batch-size", type=int, default=0, help="0 benchmarks 5/2/1")
    parser.add_argument("--max-sources", type=int)
    args = parser.parse_args()

    if os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip() != str(args.physical_gpu):
        raise RuntimeError("worker must be launched with exactly its physical GPU first in CUDA_VISIBLE_DEVICES")
    config = GenerationConfig.from_yaml(args.config)
    tasks = load_tasks(args.records, "all", config.seed)
    result = generate_worker(
        config,
        tasks,
        batch_size=args.batch_size,
        physical_gpu=args.physical_gpu,
        max_sources=args.max_sources,
        contention_check=lambda: has_foreign_compute_process(args.physical_gpu, os.getpid()),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 75 if result["status"] == "resource_contention" else 0


if __name__ == "__main__":
    raise SystemExit(main())
