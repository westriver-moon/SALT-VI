import json
from pathlib import Path

import pytest

from pasd_offline.tasks import load_tasks


@pytest.mark.parametrize("views", [1, 2, 5])
def test_loads_canonical_source_records(tmp_path: Path, views: int):
    records = tmp_path / "records.jsonl"
    records.write_text(
        json.dumps(
            {
                "image": str(tmp_path / "sample.png"),
                "source_key": "cam1/0001/sample.png",
                "views": [
                    {
                        "view_index": index,
                        "hypothesis_weight": 1 / views,
                        "caption": f"caption {index}",
                        "seed": index,
                        "output": f"images/view_{index}.png",
                    }
                    for index in range(views)
                ],
            }
        ),
        encoding="utf-8",
    )
    tasks = load_tasks(records)
    assert [task.view_index for task in tasks] == list(range(views))
    assert [task.caption for task in tasks] == [f"caption {index}" for index in range(views)]
    assert [task.hypothesis_weight for task in tasks] == pytest.approx(
        [1 / views] * views
    )


def test_rejects_hypothesis_weights_that_do_not_sum_to_one(tmp_path: Path):
    records = tmp_path / "records.jsonl"
    records.write_text(
        json.dumps(
            {
                "image": str(tmp_path / "sample.png"),
                "source_key": "cam1/0001/sample.png",
                "views": [
                    {
                        "view_index": 0,
                        "hypothesis_weight": 0.8,
                        "caption": "caption",
                        "seed": 0,
                        "output": "images/view.png",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="sum to one"):
        load_tasks(records)


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
        load_tasks(records)


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
        load_tasks(records)
