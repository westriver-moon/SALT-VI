from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


CAMERAS = {
    "rgb": ("cam1", "cam2", "cam4", "cam5"),
    "ir": ("cam3", "cam6"),
}


@dataclass(frozen=True)
class SYSUTrainSource:
    index: int
    source_key: str
    identity: str
    camera: str
    label: int


def read_train_identities(data_root: str | Path) -> list[str]:
    root = Path(data_root).expanduser().resolve()
    identities = []
    for name in ("train_id.txt", "val_id.txt"):
        text = (root / "exp" / name).read_text(encoding="utf-8")
        identities.extend(
            token.strip().zfill(4)
            for token in text.replace("\n", ",").split(",")
            if token.strip()
        )
    return sorted(set(identities))


def read_test_identities(data_root: str | Path) -> list[str]:
    root = Path(data_root).expanduser().resolve()
    text = (root / "exp" / "test_id.txt").read_text(encoding="utf-8")
    return sorted(
        {
            token.strip().zfill(4)
            for token in text.replace("\n", ",").split(",")
            if token.strip()
        }
    )


def collect_train_source_records(
    data_root: str | Path, modality: str
) -> list[SYSUTrainSource]:
    root = Path(data_root).expanduser().resolve()
    modality = str(modality).lower()
    if modality not in CAMERAS:
        raise ValueError(f"Unsupported SYSU modality: {modality}")
    identities = read_train_identities(root)
    label_by_identity = {identity: index for index, identity in enumerate(identities)}
    records = []
    for identity in identities:
        for camera in CAMERAS[modality]:
            directory = root / camera / identity
            if not directory.is_dir():
                continue
            for path in sorted(item for item in directory.iterdir() if item.is_file()):
                records.append(
                    SYSUTrainSource(
                        index=len(records),
                        source_key=path.relative_to(root).as_posix(),
                        identity=identity,
                        camera=camera,
                        label=label_by_identity[identity],
                    )
                )
    return records


def collect_test_source_records(
    data_root: str | Path, modality: str
) -> list[SYSUTrainSource]:
    """Collect every test-ID image, including all possible single-shot galleries."""
    root = Path(data_root).expanduser().resolve()
    modality = str(modality).lower()
    if modality not in CAMERAS:
        raise ValueError(f"Unsupported SYSU modality: {modality}")
    identities = read_test_identities(root)
    label_by_identity = {identity: index for index, identity in enumerate(identities)}
    records = []
    for identity in identities:
        for camera in CAMERAS[modality]:
            directory = root / camera / identity
            if not directory.is_dir():
                continue
            for path in sorted(item for item in directory.iterdir() if item.is_file()):
                records.append(
                    SYSUTrainSource(
                        index=len(records),
                        source_key=path.relative_to(root).as_posix(),
                        identity=identity,
                        camera=camera,
                        label=label_by_identity[identity],
                    )
                )
    return records


def source_manifest_path(data_root: str | Path, modality: str) -> Path:
    return (
        Path(data_root).expanduser().resolve()
        / f"train_{str(modality).lower()}_resized_sources.jsonl"
    )


def write_train_source_manifest(
    data_root: str | Path,
    modality: str,
    records: list[SYSUTrainSource],
) -> Path:
    path = source_manifest_path(data_root, modality)
    path.write_text(
        "".join(json.dumps(asdict(record), separators=(",", ":")) + "\n" for record in records),
        encoding="utf-8",
    )
    return path


def load_train_source_records(
    data_root: str | Path, modality: str
) -> list[SYSUTrainSource]:
    path = source_manifest_path(data_root, modality)
    if not path.is_file():
        return collect_train_source_records(data_root, modality)
    records = [
        SYSUTrainSource(**json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if [record.index for record in records] != list(range(len(records))):
        raise ValueError(f"SYSU source manifest has invalid indices: {path}")
    return records
