import json
from pathlib import Path

from PIL import Image

from pasd_offline.sysu import build_sysu_records, select_pilot_records
from pasd_offline.tasks import load_tasks


def candidate(identity: str, camera: int):
    return {
        "id": identity,
        "cam": str(camera),
        "img": "0001",
        "description": f"original {identity} camera {camera}",
        "paraphrases": [f"variant {index} {identity} camera {camera}" for index in range(4)],
    }


def test_builds_one_or_five_deterministic_protocol_views(tmp_path: Path):
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
    for views in (1, 5):
        records = build_sysu_records(
            root,
            {"rgb": rgb_path},
            output,
            seed=17,
            views_per_source=views,
            enforce_official_counts=False,
        )
        assert len(records) == 1
        assert len(records[0]["views"]) == views
        assert len({view["seed"] for view in records[0]["views"]}) == views
        assert len(load_tasks(output)) == views
        pilot = select_pilot_records(records, tmp_path / "pilot.jsonl", count=1)
        assert {record["modality"] for record in pilot} == {"rgb"}


def test_include_all_keeps_non_protocol_sources(tmp_path: Path):
    root = tmp_path / "SYSU-MM01"
    (root / "exp").mkdir(parents=True)
    (root / "exp" / "train_id.txt").write_text("1", encoding="utf-8")
    (root / "exp" / "val_id.txt").write_text("2", encoding="utf-8")
    (root / "exp" / "test_id.txt").write_text("3", encoding="utf-8")
    for identity in ("0001", "9999"):
        path = root / "cam6" / identity
        path.mkdir(parents=True)
        Image.new("RGB", (64, 128), "gray").save(path / "0001.jpg")
    captions = {
        "datasets/sysu/cam6/0001/0001.jpg": candidate("0001", 6),
        "datasets/sysu/cam6/9999/0001.jpg": candidate("9999", 6),
    }
    path = tmp_path / "ir.json"
    path.write_text(json.dumps(captions), encoding="utf-8")
    output = tmp_path / "records.jsonl"
    protocol = build_sysu_records(
        root,
        {"ir": path},
        output,
        seed=17,
        views_per_source=1,
        enforce_official_counts=False,
    )
    assert len(protocol) == 1
    assert protocol[0]["identity"] == "0001"
    full = build_sysu_records(
        root,
        {"ir": path},
        output,
        seed=17,
        views_per_source=1,
        enforce_official_counts=False,
        include_all=True,
    )
    assert len(full) == 2
    assert {record["split"] for record in full} == {"train", "all"}
