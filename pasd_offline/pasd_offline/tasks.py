from __future__ import annotations

import json
import math
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
    hypothesis_id: str = ""
    hypothesis_weight: float = 1.0
    imagination_contract_sha256: str = ""
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
        if not views:
            raise ValueError(f"{source_key} must contain at least one view")
        weights = [
            float(view.get("hypothesis_weight", 1.0 / len(views))) for view in views
        ]
        if any(not math.isfinite(weight) or weight <= 0 for weight in weights):
            raise ValueError(f"{source_key} hypothesis weights must be finite and positive")
        if not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(f"{source_key} hypothesis weights must sum to one")
        for view, weight in zip(views, weights):
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
                    hypothesis_id=str(
                        view.get("hypothesis_id", f"h{int(view['view_index']):02d}")
                    ),
                    hypothesis_weight=weight,
                    imagination_contract_sha256=str(
                        record.get("imagination_contract_sha256", "")
                    ),
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
