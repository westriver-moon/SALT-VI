from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Install the official PASD checkpoint archive")
    parser.add_argument("--archive", required=True)
    parser.add_argument("--destination", default=ROOT / "checkpoints" / "pasd")
    args = parser.parse_args()

    archive = Path(args.archive).resolve()
    destination = Path(args.destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with ZipFile(archive) as zip_file:
        zip_file.extractall(destination)

    hasher = hashlib.sha256()
    with archive.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    digest = hasher.hexdigest()
    (destination / "official-archive.sha256").write_text(
        f"{digest}  {archive.name}\n", encoding="utf-8"
    )
    print(destination / "checkpoint-100000")


if __name__ == "__main__":
    main()
