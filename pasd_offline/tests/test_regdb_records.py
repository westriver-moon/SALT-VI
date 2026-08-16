import json
from pathlib import Path

from PIL import Image

from pasd_offline.regdb import build_regdb_records, read_trial_splits
from pasd_offline.tasks import load_tasks


def _caption(value: str) -> dict:
    return {"description": value}


def _make_regdb(tmp_path: Path) -> Path:
    root = tmp_path / "RegDB"
    for identity in ("1", "2"):
        for directory, kind in (("Visible", "v"), ("Thermal", "t")):
            path = root / directory / identity
            path.mkdir(parents=True, exist_ok=True)
            for frame in ("00007", "00009"):
                Image.new("RGB", (64, 192), "gray").save(
                    path / f"male_front_{kind}_{frame}_{identity}.bmp"
                )
    (root / "idx").mkdir(parents=True)
    (root / "idx" / "train_visible_1.txt").write_text(
        "Visible/1/male_front_v_00007_1.bmp 0\nVisible/1/male_front_v_00009_1.bmp 0\n",
        encoding="utf-8",
    )
    (root / "idx" / "train_thermal_1.txt").write_text(
        "Thermal/1/male_front_t_00007_1.bmp 0\nThermal/1/male_front_t_00009_1.bmp 0\n",
        encoding="utf-8",
    )
    (root / "idx" / "test_visible_1.txt").write_text(
        "Visible/2/male_front_v_00007_2.bmp 0\nVisible/2/male_front_v_00009_2.bmp 0\n",
        encoding="utf-8",
    )
    (root / "idx" / "test_thermal_1.txt").write_text(
        "Thermal/2/male_front_t_00007_2.bmp 0\nThermal/2/male_front_t_00009_2.bmp 0\n",
        encoding="utf-8",
    )
    return root


def _captions(root: Path, tmp_path: Path) -> tuple[Path, Path]:
    rgb = {
        f"datasets/regdb/Visible/{identity}/male_front_v_{frame}_{identity}.bmp": _caption(
            f"rgb {identity} {frame}"
        )
        for identity in ("1", "2")
        for frame in ("00007", "00009")
    }
    ir = {
        f"datasets/regdb/Thermal/{identity}/male_front_t_{frame}_{identity}.bmp": _caption(
            f"ir {identity} {frame}"
        )
        for identity in ("1", "2")
        for frame in ("00007", "00009")
    }
    rgb_path = tmp_path / "rgb.json"
    ir_path = tmp_path / "ir.json"
    rgb_path.write_text(json.dumps(rgb), encoding="utf-8")
    ir_path.write_text(json.dumps(ir), encoding="utf-8")
    return rgb_path, ir_path


def test_reads_regdb_trial_splits(tmp_path: Path):
    root = _make_regdb(tmp_path)
    splits = read_trial_splits(root, trial=1)
    assert splits["Visible/1/male_front_v_00007_1.bmp"] == "train"
    assert splits["Visible/2/male_front_v_00007_2.bmp"] == "test"


def test_builds_all_regdb_rgb_and_ir_records(tmp_path: Path):
    root = _make_regdb(tmp_path)
    rgb_path, ir_path = _captions(root, tmp_path)
    output = tmp_path / "records.jsonl"
    records = build_regdb_records(
        root,
        {"rgb": rgb_path, "ir": ir_path},
        output,
        seed=17,
        views_per_source=1,
        enforce_official_counts=False,
    )
    assert len(records) == 8
    assert [record["modality"] for record in records].count("rgb") == 4
    assert [record["modality"] for record in records].count("ir") == 4
    assert {record["split"] for record in records} == {"train", "test"}
    assert sum(record["split"] == "train" for record in records) == 4
    assert sum(record["split"] == "test" for record in records) == 4
    assert {record["camera"] for record in records} == {0}
    tasks = load_tasks(output)
    assert len(tasks) == 8
    assert all(task.output.is_relative_to(Path("images")) for task in tasks)
    assert all(task.caption.startswith(("rgb ", "ir ")) for task in tasks)


def test_regdb_trial_mode_marks_train_and_test(tmp_path: Path):
    root = _make_regdb(tmp_path)
    rgb_path, ir_path = _captions(root, tmp_path)
    output = tmp_path / "records.jsonl"
    records = build_regdb_records(
        root,
        {"rgb": rgb_path, "ir": ir_path},
        output,
        seed=17,
        views_per_source=1,
        trial=1,
        include_all=False,
        enforce_official_counts=False,
    )
    assert len(records) == 8
    assert {record["split"] for record in records} == {"train", "test"}
    assert sum(record["split"] == "train" for record in records) == 4
    assert sum(record["split"] == "test" for record in records) == 4
