from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class GenerationTask:
    image: Path
    caption: str
    output: Path
    seed: int
    modality: str = ""
    identity: str = ""
    source_key: str = ""
    view_index: int = 0
    camera: int = -1
    split: str = ""


def _relative_path(value: str | Path, field: str) -> Path:
    normalized = str(value).replace("\\", "/")
    path = Path(normalized)
    if (
        not normalized
        or normalized.startswith("/")
        or re.match(r"^[A-Za-z]:/", normalized)
        or any(part == ".." for part in path.parts)
    ):
        raise ValueError(f"{field} must be a relative path without '..': {value!r}")
    return path


def normalize_source_key(value: str) -> str:
    return _relative_path(value, "source_key").as_posix()


def normalize_output_path(value: str | Path) -> Path:
    return _relative_path(value, "output")


def task_payload(task: GenerationTask) -> dict:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(task).items()
    }


def load_tasks(records_path: str | Path) -> list[GenerationTask]:
    """Load the canonical one-row-per-source record format."""

    records_path = Path(records_path).expanduser().resolve()
    tasks: list[GenerationTask] = []
    for line in records_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        source_key = normalize_source_key(record["source_key"])
        views = list(record.get("views", ()))
        if len(views) not in (1, 5):
            raise ValueError(f"{source_key} must contain one or five views")
        for view in views:
            tasks.append(
                GenerationTask(
                    image=Path(record["image"]).expanduser().resolve(),
                    caption=str(view["caption"]),
                    output=normalize_output_path(view["output"]),
                    seed=int(view["seed"]),
                    modality=str(record.get("modality", "")),
                    identity=str(record.get("identity", "")),
                    source_key=source_key,
                    view_index=int(view["view_index"]),
                    camera=int(record.get("camera", -1)),
                    split=str(record.get("split", "")),
                )
            )
    return tasks


def group_tasks_by_source(tasks: list[GenerationTask]) -> list[list[GenerationTask]]:
    grouped: dict[str, list[GenerationTask]] = {}
    order: list[str] = []
    for task in tasks:
        key = task.source_key or str(task.image)
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(task)
    result = []
    for key in order:
        values = sorted(grouped[key], key=lambda task: task.view_index)
        if len({task.image for task in values}) != 1:
            raise ValueError(f"source task group has multiple images: {key}")
        result.append(values)
    return result
