from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pasd_offline.contracts import load_build_contract  # noqa: E402
from pasd_offline.generate import consolidate_manifest, validate_source  # noqa: E402
from pasd_offline.config import GenerationConfig  # noqa: E402
from pasd_offline.tasks import group_tasks_by_source, load_tasks  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a fixed-size SYSU PASD multiview dataset")
    parser.add_argument("--config", required=True)
    parser.add_argument("--records", required=True)
    args = parser.parse_args()
    config = GenerationConfig.from_yaml(args.config)
    load_build_contract(config)
    output_root = config.output_root
    tasks = load_tasks(args.records, "all", seed=0)
    groups = group_tasks_by_source(tasks)
    errors = []
    view_count = 0
    for group in groups:
        source_key = group[0].source_key
        try:
            metadata = validate_source(group, output_root, config)
            view_count += len(metadata["views"])
        except (OSError, KeyError, ValueError, json.JSONDecodeError) as error:
            errors.append(f"invalid:{source_key}:{error}")
    expected_sources = len(groups)
    summary = {
        "expected_sources": expected_sources,
        "record_sources": len(groups),
        "valid_views": view_count,
        "expected_views": expected_sources * 5,
        "errors": errors[:1000],
        "error_count": len(errors),
        "complete": not errors and view_count == expected_sources * 5,
    }
    report = output_root / "validation-report.json"
    report.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["complete"]:
        raise SystemExit(1)
    consolidate_manifest(output_root, tasks, config)


if __name__ == "__main__":
    main()
