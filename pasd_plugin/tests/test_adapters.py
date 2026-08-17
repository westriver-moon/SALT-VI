from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from pasd_plugin.adapters import build_records
from pasd_plugin.config import PluginConfig


def _image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (40, 100), (20, 30, 40)).save(path)


def _config(tmp_path: Path, dataset: str, root: Path, rgb: Path, ir: Path) -> PluginConfig:
    return PluginConfig(
        dataset=dataset,
        dataset_root=root,
        captions={"rgb": rgb, "ir": ir},
        output_root=tmp_path / "derived",
        pretrained_model_path=tmp_path / "sd",
        pasd_model_path=tmp_path / "pasd",
    )


def _captions(path: Path, values: dict[str, str]) -> Path:
    path.write_text(json.dumps({key: {"description": value} for key, value in values.items()}), encoding="utf-8")
    return path


def test_sysu_adapter_reads_protocol_cameras_and_captions(tmp_path: Path) -> None:
    root = tmp_path / "SYSU-MM01"
    (root / "exp").mkdir(parents=True)
    for split, identities in (("train", "0001"), ("val", "0002"), ("test", "0003")):
        (root / "exp" / f"{split}_id.txt").write_text(identities, encoding="utf-8")
    rgb_key = "cam1/0001/a.jpg"
    ir_key = "cam3/0002/b.jpg"
    _image(root / rgb_key)
    _image(root / ir_key)
    rgb = _captions(tmp_path / "rgb.json", {f"datasets/sysu/{rgb_key}": "rgb"})
    ir = _captions(tmp_path / "ir.json", {f"datasets/sysu/{ir_key}": "ir"})

    records = build_records(_config(tmp_path, "sysu", root, rgb, ir))

    assert [(record.source_key, record.modality) for record in records] == [(rgb_key, "rgb"), (ir_key, "ir")]
    assert records[0].protocol == {"sysu": {"split": "train"}}
    assert records[1].protocol == {"sysu": {"split": "val"}}


def test_regdb_adapter_deduplicates_sources_and_keeps_all_trials(tmp_path: Path) -> None:
    root = tmp_path / "RegDB"
    (root / "idx").mkdir(parents=True)
    rgb_key = "Visible/0001/a.jpg"
    ir_key = "Thermal/0001/b.jpg"
    _image(root / rgb_key)
    _image(root / ir_key)
    for trial in range(1, 11):
        (root / "idx" / f"train_visible_{trial}.txt").write_text(f"{rgb_key} 0\n", encoding="utf-8")
        (root / "idx" / f"test_visible_{trial}.txt").write_text("", encoding="utf-8")
        (root / "idx" / f"train_thermal_{trial}.txt").write_text(f"{ir_key} 0\n", encoding="utf-8")
        (root / "idx" / f"test_thermal_{trial}.txt").write_text("", encoding="utf-8")
    rgb = _captions(tmp_path / "rgb.json", {f"datasets/regdb/{rgb_key}": "rgb"})
    ir = _captions(tmp_path / "ir.json", {f"datasets/regdb/{ir_key}": "ir"})

    records = build_records(_config(tmp_path, "regdb", root, rgb, ir))

    assert len(records) == 2
    assert records[0].protocol["regdb"]["trials"] == {str(index): "train" for index in range(1, 11)}


def test_llcm_adapter_preserves_index_labels_and_train_test_membership(tmp_path: Path) -> None:
    root = tmp_path / "LLCM"
    (root / "idx").mkdir(parents=True)
    sources = {
        "train_vis.txt": ("vis/0001/0001_c04_train_vis.jpg", 3, "rgb"),
        "train_nir.txt": ("nir/0001/0001_c06_train_nir.jpg", 3, "ir"),
        "test_vis.txt": ("vis/0002/0002_c01_test_vis.jpg", 7, "rgb"),
        "test_nir.txt": ("nir/0002/0002_c02_test_nir.jpg", 7, "ir"),
    }
    captions = {"rgb": {}, "ir": {}}
    for index, (key, label, modality) in sources.items():
        _image(root / key)
        (root / "idx" / index).write_text(f"{key} {label}\n", encoding="utf-8")
        captions[modality][f"datasets/llcm/{key}"] = modality
    rgb = _captions(tmp_path / "rgb.json", captions["rgb"])
    ir = _captions(tmp_path / "ir.json", captions["ir"])

    records = build_records(_config(tmp_path, "llcm", root, rgb, ir))

    assert len(records) == 4
    train = next(record for record in records if record.source_key.endswith("train_nir.jpg"))
    assert train.camera == 6
    assert train.source_label == 3
    assert train.protocol == {"llcm": {"split": "train", "label": 3}}


def test_adapter_rejects_unused_captions(tmp_path: Path) -> None:
    root = tmp_path / "LLCM"
    (root / "idx").mkdir(parents=True)
    for name in ("train_vis.txt", "train_nir.txt", "test_vis.txt", "test_nir.txt"):
        (root / "idx" / name).write_text("", encoding="utf-8")
    rgb = _captions(tmp_path / "rgb.json", {"vis/extra.jpg": "rgb"})
    ir = _captions(tmp_path / "ir.json", {"nir/extra.jpg": "ir"})
    with pytest.raises(ValueError, match="no official sources"):
        build_records(_config(tmp_path, "llcm", root, rgb, ir))


def test_adapter_rejects_missing_caption_and_conflicting_duplicate_source(tmp_path: Path) -> None:
    root = tmp_path / "SYSU-MM01"
    (root / "exp").mkdir(parents=True)
    for split, identity in (("train", "0001"), ("val", ""), ("test", "")):
        (root / "exp" / f"{split}_id.txt").write_text(identity, encoding="utf-8")
    key = "cam1/0001/a.jpg"
    _image(root / key)
    rgb = _captions(tmp_path / "rgb.json", {})
    ir = _captions(tmp_path / "ir.json", {})
    with pytest.raises(KeyError, match="missing rgb caption"):
        build_records(_config(tmp_path, "sysu", root, rgb, ir))

    llcm = tmp_path / "LLCM"
    (llcm / "idx").mkdir(parents=True)
    duplicate = "shared/0001/a.jpg"
    _image(llcm / duplicate)
    for index, line in {
        "train_vis.txt": f"{duplicate} 1\n",
        "train_nir.txt": f"{duplicate} 1\n",
        "test_vis.txt": "",
        "test_nir.txt": "",
    }.items():
        (llcm / "idx" / index).write_text(line, encoding="utf-8")
    rgb = _captions(tmp_path / "llcm-rgb.json", {f"datasets/llcm/{duplicate}": "rgb"})
    ir = _captions(tmp_path / "llcm-ir.json", {f"datasets/llcm/{duplicate}": "ir"})
    with pytest.raises(ValueError, match="conflicting index metadata"):
        build_records(_config(tmp_path, "llcm", llcm, rgb, ir))
