from __future__ import annotations

import json
import random
from dataclasses import dataclass
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


def _caption_output(path: Path, caption_index: int) -> Path:
    return path.with_name(f"{path.stem}__caption{caption_index:02d}{path.suffix or '.png'}")


def load_tasks(
    records_path: str | Path,
    mode: str,
    seed: int,
    caption_pool_path: str | Path | None = None,
) -> list[GenerationTask]:
    records_path = Path(records_path).resolve()
    caption_pool = (
        json.loads(Path(caption_pool_path).read_text(encoding="utf-8"))
        if caption_pool_path
        else {}
    )
    tasks: list[GenerationTask] = []
    for line_index, line in enumerate(records_path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        record = json.loads(line)
        if "views" in record:
            views = list(record["views"])
            if mode == "first":
                views = views[:1]
            elif mode == "random":
                views = [random.Random(seed + line_index).choice(views)]
            elif mode != "all":
                raise ValueError(f"unknown caption mode: {mode}")
            for view in views:
                tasks.append(
                    GenerationTask(
                        image=Path(record["image"]).expanduser().resolve(),
                        caption=str(view["caption"]),
                        output=Path(view["output"]),
                        seed=int(view["seed"]),
                        modality=str(record.get("modality", "")),
                        identity=str(record.get("identity", "")),
                        source_key=str(record.get("source_key", "")),
                        view_index=int(view["view_index"]),
                        camera=int(record.get("camera", -1)),
                        split=str(record.get("split", "")),
                    )
                )
            continue
        if "captions" in record:
            captions = record["captions"]
        elif "caption" in record:
            captions = [record["caption"]]
        else:
            captions = caption_pool[record["caption_pool_key"]]
        output = Path(record.get("output") or f"{Path(record['image']).stem}.png")
        if mode == "first":
            selected = [(0, captions[0])]
        elif mode == "random":
            index = random.Random(seed + line_index).randrange(len(captions))
            selected = [(index, captions[index])]
        elif mode == "all":
            selected = list(enumerate(captions))
        else:
            raise ValueError(f"unknown caption mode: {mode}")

        for caption_index, caption in selected:
            task_output = _caption_output(output, caption_index) if mode == "all" and len(captions) > 1 else output
            tasks.append(
                GenerationTask(
                    image=Path(record["image"]).expanduser().resolve(),
                    caption=caption,
                    output=task_output,
                    seed=int(record.get("seed", seed + line_index * 1000 + caption_index)),
                    modality=str(record.get("modality", "")),
                    identity=str(record.get("identity", "")),
                    source_key=str(record.get("source_key", "")),
                    view_index=int(caption_index),
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
