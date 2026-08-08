from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageStat


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pasd_offline.generate import consolidate_manifest, sha256, source_is_complete  # noqa: E402
from pasd_offline.config import GenerationConfig  # noqa: E402
from pasd_offline.tasks import group_tasks_by_source, load_tasks  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a fixed-size SYSU PASD multiview dataset")
    parser.add_argument("--config", required=True)
    parser.add_argument("--records", required=True)
    parser.add_argument("--skip-hashes", action="store_true")
    args = parser.parse_args()
    config = GenerationConfig.from_yaml(args.config)
    output_root = config.output_root
    tasks = load_tasks(args.records, "all", seed=0)
    groups = group_tasks_by_source(tasks)
    errors = []
    view_count = 0
    for group in groups:
        source_key = group[0].source_key
        if not source_is_complete(group, output_root, config):
            errors.append(f"incomplete:{source_key}")
            continue
        metadata_path = output_root / "metadata" / Path(source_key).parent / f"{Path(source_key).stem}.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if len(metadata.get("views", [])) != 5:
            errors.append(f"view_count:{source_key}")
            continue
        for view in metadata["views"]:
            path = output_root / view["output"]
            try:
                with Image.open(path) as image:
                    image.load()
                    if image.format != "PNG" or image.mode != "RGB" or image.size != (256, 512):
                        raise ValueError(f"contract={image.format}/{image.mode}/{image.size}")
                    if max(ImageStat.Stat(image).var) <= 0:
                        raise ValueError("constant")
                    if metadata["modality"] == "ir":
                        pixels = np.asarray(image)
                        if not (np.array_equal(pixels[..., 0], pixels[..., 1]) and np.array_equal(pixels[..., 1], pixels[..., 2])):
                            raise ValueError("IR channels differ")
                if not args.skip_hashes and sha256(path) != view["output_sha256"]:
                    raise ValueError("sha256 mismatch")
                view_count += 1
            except Exception as error:
                errors.append(f"invalid:{source_key}:{view.get('view_index')}:{error}")
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
