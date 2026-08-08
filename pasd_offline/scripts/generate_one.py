from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pasd_offline.config import GenerationConfig  # noqa: E402
from pasd_offline.generate import generate_task  # noqa: E402
from pasd_offline.runtime import PASDGenerator  # noqa: E402
from pasd_offline.tasks import GenerationTask  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate one PASD image from an explicit caption")
    parser.add_argument("--config", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--caption", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    config = GenerationConfig.from_yaml(args.config)
    task = GenerationTask(
        image=Path(args.image).expanduser().resolve(),
        caption=args.caption,
        output=Path(args.output),
        seed=config.seed if args.seed is None else args.seed,
    )
    entry = generate_task(PASDGenerator(config), task, config.output_root)
    print(entry["output"])


if __name__ == "__main__":
    main()
