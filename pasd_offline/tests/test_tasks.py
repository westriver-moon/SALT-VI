import json
from pathlib import Path

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
