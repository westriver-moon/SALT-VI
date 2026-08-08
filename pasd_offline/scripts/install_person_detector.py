from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

from ultralytics import YOLO


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Install the small COCO person detector")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    previous = Path.cwd()
    try:
        os.chdir(output.parent)
        YOLO("yolov8n.pt")
        downloaded = output.parent / "yolov8n.pt"
    finally:
        os.chdir(previous)
    if downloaded != output:
        os.replace(downloaded, output)
    print(f"path={output} size={output.stat().st_size} sha256={sha256(output)}")


if __name__ == "__main__":
    main()
