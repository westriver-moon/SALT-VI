#!/usr/bin/env python3
"""Merge and verify completed Qwen caption-augmentation shards."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=str(path.parent), delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--shard-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with Path(args.input).open("r", encoding="utf-8") as handle:
        source = json.load(handle)
    merged = {}
    for path in sorted(Path(args.shard_dir).glob("caption_qwen3_14b_awq_4x.shard-*.json")):
        with path.open("r", encoding="utf-8") as handle:
            shard = json.load(handle)
        overlap = set(merged).intersection(shard)
        if overlap:
            raise ValueError(f"duplicate keys across shards: {sorted(overlap)[:3]}")
        merged.update(shard)
    missing = set(source).difference(merged)
    extra = set(merged).difference(source)
    invalid = []
    for key, value in merged.items():
        paraphrases = value.get("paraphrases")
        valid = (
            isinstance(paraphrases, list)
            and len(paraphrases) == 4
            and all(isinstance(text, str) and text.strip() for text in paraphrases)
            and len({text.strip().casefold() for text in paraphrases}) == 4
            and all(len(text.split()) <= 45 for text in paraphrases)
            and value.get("description") == source[key].get("description")
        )
        if not valid:
            invalid.append(key)
    if missing or extra or invalid:
        raise ValueError(
            f"incomplete merge: expected={len(source)} actual={len(merged)} "
            f"missing={len(missing)} extra={len(extra)} invalid={len(invalid)}"
        )
    atomic_json(Path(args.output), {key: merged[key] for key in sorted(merged)})
    print(f"verified {len(merged)} records and {len(merged) * 4} paraphrases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
