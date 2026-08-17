from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from PIL import Image, ImageDraw
import pytest

from pasd_plugin.config import PluginConfig
from pasd_plugin.contracts import SourceRecord, atomic_json, sha256_file, write_jsonl
from pasd_plugin.generation import consolidate_manifest, generate_records, prepare_build
from pasd_plugin.validation import validate_dataset, validate_protocol


class FakeGenerator:
    def __init__(self, config: PluginConfig):
        self.config = config

    def generate_views(self, image, captions, seeds, modality, batch_size):
        output = Image.new("RGB", (256, 512), "black")
        ImageDraw.Draw(output).rectangle((20, 20, 235, 491), fill=(30, 80, 120))
        return [output], {
            "mode": "person_fit_blurred_background",
            "source_size": [40, 100],
            "target_size": [256, 512],
            "scale": 5.12,
            "resized_size": [205, 512],
            "padding": [25, 0, 26, 0],
            "foreground_box": [25, 0, 230, 512],
            "background_blur_radius": 24.0,
        }


def _prepared_record(tmp_path: Path) -> tuple[PluginConfig, SourceRecord, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    source = tmp_path / "source.jpg"
    Image.new("RGB", (40, 100), "white").save(source)
    config = PluginConfig(
        dataset="llcm",
        dataset_root=tmp_path,
        captions={"rgb": tmp_path / "rgb.json", "ir": tmp_path / "ir.json"},
        output_root=tmp_path / "output",
        pretrained_model_path=tmp_path / "sd",
        pasd_model_path=tmp_path / "pasd",
    )
    record = SourceRecord(
        source_key="vis/0001/source.jpg",
        image=str(source),
        modality="rgb",
        identity="0001",
        camera=4,
        caption="a person",
        seed=7,
        output="images/vis/0001/source.png",
        protocol={"llcm": {"split": "train", "label": 0}},
        source_label=0,
    )
    records_path = tmp_path / "records.jsonl"
    records = write_jsonl(records_path, [record])
    prepare_build(config, records_path)
    generate_records(config, records, physical_gpu=2, generator_factory=FakeGenerator)
    consolidate_manifest(config, records)
    return config, record, source


def test_generation_manifest_and_no_stretch_validation(tmp_path: Path) -> None:
    config, record, _ = _prepared_record(tmp_path)
    records = [record]

    assert validate_dataset(config, records)["complete"]
    summary = json.loads((config.output_root / "manifest.json").read_text(encoding="utf-8"))
    assert summary["complete"]
    assert (config.output_root / "manifest.jsonl").is_file()
    row = json.loads((config.output_root / "manifest.jsonl").read_text(encoding="utf-8"))
    assert row["caption"] == record.caption
    assert row["seed"] == record.seed
    assert row["input_sha256"]
    assert row["build_sha256"] == config.build_sha256


def test_fake_generator_generates_one_atomic_view_for_each_modality(tmp_path: Path) -> None:
    source_rgb = tmp_path / "rgb.jpg"
    source_ir = tmp_path / "ir.jpg"
    Image.new("RGB", (40, 100), "white").save(source_rgb)
    Image.new("RGB", (40, 100), "gray").save(source_ir)
    config = PluginConfig(
        dataset="llcm",
        dataset_root=tmp_path,
        captions={"rgb": tmp_path / "rgb.json", "ir": tmp_path / "ir.json"},
        output_root=tmp_path / "output",
        pretrained_model_path=tmp_path / "sd",
        pasd_model_path=tmp_path / "pasd",
    )
    records = [
        SourceRecord("vis/0001/rgb.jpg", str(source_rgb), "rgb", "0001", 4, "rgb person", 11,
                     "images/vis/0001/rgb.png", {"llcm": {"split": "train", "label": 0}}, 0),
        SourceRecord("nir/0001/ir.jpg", str(source_ir), "ir", "0001", 6, "ir person", 12,
                     "images/nir/0001/ir.png", {"llcm": {"split": "train", "label": 0}}, 0),
    ]
    records_path = tmp_path / "records.jsonl"
    write_jsonl(records_path, records)
    prepare_build(config, records_path)
    result = generate_records(config, records, physical_gpu=2, generator_factory=FakeGenerator)
    summary = consolidate_manifest(config, records)

    assert result == {"completed": 2, "skipped": 0, "requested": 2}
    assert summary["complete"]
    assert [record.seed for record in records] == [11, 12]
    assert all((config.output_root / record.output).is_file() for record in records)
    assert validate_dataset(config, records)["complete"]


def test_validation_rejects_contract_and_artifact_failures(tmp_path: Path) -> None:
    config, record, source = _prepared_record(tmp_path / "caption")
    assert not validate_dataset(config, [replace(record, caption="different person")])["complete"]
    assert not validate_dataset(config, [replace(record, protocol={"llcm": {"split": "bad", "label": 0}})])["complete"]
    with pytest.raises(ValueError, match="all ten"):
        validate_protocol(replace(record, protocol={"regdb": {"trials": {"1": "train"}}}), "regdb")

    config, record, _ = _prepared_record(tmp_path / "checksum")
    output = config.output_root / record.output
    Image.new("RGB", (256, 512), "green").save(output)
    assert not validate_dataset(config, [record])["complete"]

    config, record, _ = _prepared_record(tmp_path / "size")
    output = config.output_root / record.output
    Image.new("RGB", (255, 512), "green").save(output)
    marker_path = config.output_root / "metadata" / "vis/0001/source.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["artifact"]["output_sha256"] = sha256_file(output)
    marker["artifact"]["output_bytes"] = output.stat().st_size
    atomic_json(marker_path, marker)
    assert not validate_dataset(config, [record])["complete"]

    config, record, source = _prepared_record(tmp_path / "missing")
    source.unlink()
    assert not consolidate_manifest(config, [record])["complete"]
    assert not validate_dataset(config, [record])["complete"]
