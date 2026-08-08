import json
from pathlib import Path

import pytest

from pasd_offline.tasks import load_tasks


def test_caption_modes(tmp_path: Path):
    records = tmp_path / "records.jsonl"
    records.write_text(
        json.dumps(
            {
                "image": str(tmp_path / "sample.png"),
                "captions": ["caption zero", "caption one"],
                "output": "rgb/sample.png",
            }
        ),
        encoding="utf-8",
    )

    first = load_tasks(records, "first", 7)
    random_a = load_tasks(records, "random", 7)
    random_b = load_tasks(records, "random", 7)
    all_tasks = load_tasks(records, "all", 7)

    assert first[0].caption == "caption zero"
    assert random_a == random_b
    assert [task.caption for task in all_tasks] == ["caption zero", "caption one"]
    assert all_tasks[0].output.name == "sample__caption00.png"
    assert all_tasks[1].output.name == "sample__caption01.png"
    assert all(task.task_kind == "generic" for task in all_tasks)


@pytest.mark.parametrize("source_key", ["/tmp/person.jpg", "../person.jpg", "C:/person.jpg"])
def test_five_view_source_key_must_be_safe_relative_path(
    tmp_path: Path, source_key: str
):
    records = tmp_path / "records.jsonl"
    records.write_text(
        json.dumps(
            {
                "image": str(tmp_path / "person.jpg"),
                "source_key": source_key,
                "views": [
                    {
                        "view_index": index,
                        "caption": f"caption {index}",
                        "seed": index,
                        "output": f"images/view_{index}.png",
                    }
                    for index in range(5)
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="source_key must be a relative path"):
        load_tasks(records, "all", 7)


def test_views_schema_creates_explicit_five_view_tasks(tmp_path: Path):
    records = tmp_path / "records.jsonl"
    records.write_text(
        json.dumps(
            {
                "image": str(tmp_path / "person.jpg"),
                "source_key": "cam1/0001/person.jpg",
                "views": [
                    {
                        "view_index": index,
                        "caption": f"caption {index}",
                        "seed": index,
                        "output": f"images/view_{index}.png",
                    }
                    for index in range(5)
                ],
            }
        ),
        encoding="utf-8",
    )
    tasks = load_tasks(records, "all", 7)
    assert all(task.task_kind == "five_view" for task in tasks)


@pytest.mark.parametrize("output", ["/tmp/view.png", "../view.png", "C:/view.png"])
def test_five_view_output_must_be_safe_relative_path(tmp_path: Path, output: str):
    records = tmp_path / "records.jsonl"
    records.write_text(
        json.dumps(
            {
                "image": str(tmp_path / "person.jpg"),
                "source_key": "cam1/0001/person.jpg",
                "views": [
                    {
                        "view_index": index,
                        "caption": f"caption {index}",
                        "seed": index,
                        "output": output if index == 0 else f"images/view_{index}.png",
                    }
                    for index in range(5)
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="output must be a relative path"):
        load_tasks(records, "all", 7)
