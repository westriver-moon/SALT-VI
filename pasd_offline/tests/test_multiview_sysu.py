import json
from pathlib import Path

from PIL import Image

from pasd_offline.sysu import build_sysu_multiview_records, select_pilot_records
from pasd_offline.tasks import load_tasks


def candidate(identity: str, camera: int):
    return {
        "id": identity,
        "cam": str(camera),
        "img": "0001",
        "description": f"original {identity} camera {camera}",
        "paraphrases": [f"variant {index} {identity} camera {camera}" for index in range(4)],
    }


def test_builds_five_deterministic_protocol_views(tmp_path: Path):
    root = tmp_path / "SYSU-MM01"
    (root / "exp").mkdir(parents=True)
    (root / "exp" / "train_id.txt").write_text("1", encoding="utf-8")
    (root / "exp" / "val_id.txt").write_text("2", encoding="utf-8")
    (root / "exp" / "test_id.txt").write_text("3", encoding="utf-8")
    rgb = {"datasets/sysu/cam1/0001/0001.jpg": candidate("0001", 1)}
    # Existing RGB candidate text is immutable. One known official record has a
    # paraphrase equal to its original caption, while its four paraphrases are
    # still mutually distinct; the five-view contract must preserve it verbatim.
    rgb["datasets/sysu/cam1/0001/0001.jpg"]["paraphrases"][0] = rgb[
        "datasets/sysu/cam1/0001/0001.jpg"
    ]["description"]
    for relative in ("cam1/0001/0001.jpg",):
        path = root / relative
        path.parent.mkdir(parents=True)
        Image.new("RGB", (64, 128), "gray").save(path)
    rgb_path = tmp_path / "rgb.json"
    rgb_path.write_text(json.dumps(rgb), encoding="utf-8")
    output = tmp_path / "records.jsonl"
    records = build_sysu_multiview_records(
        root, {"rgb": rgb_path}, output, seed=17, enforce_official_counts=False
    )
    assert len(records) == 1
    assert all(len(record["views"]) == 5 for record in records)
    assert len({view["seed"] for record in records for view in record["views"]}) == 5
    tasks = load_tasks(output, "all", 0)
    assert len(tasks) == 5
    pilot = select_pilot_records(records, tmp_path / "pilot.jsonl", count=1)
    assert {record["modality"] for record in pilot} == {"rgb"}
