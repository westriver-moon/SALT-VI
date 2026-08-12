from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pasd_offline.config import GenerationConfig
from pasd_offline.generate import (
    consolidate_manifest,
    generate_source_group,
    invalidate_invalid_sources,
    prepare_build,
    source_is_generated,
    source_is_validated,
)
from pasd_offline.scheduler import generated_source_count
from pasd_offline.tasks import GenerationTask


class FakeGenerator:
    def __init__(self, config):
        self.config = config

    def generate_views(self, image_path, captions, seeds, modality, batch_size):
        pixels = np.zeros(
            (self.config.target_height, self.config.target_width, 3), dtype=np.uint8
        )
        pixels[:, :, 0] = np.arange(self.config.target_width, dtype=np.uint8)
        image = Image.fromarray(pixels, "RGB")
        return [image.copy() for _ in captions], {"mode": "test"}


def source_tasks(source: Path, views: int) -> list[GenerationTask]:
    return [
        GenerationTask(
            image=source,
            caption=f"caption {index}",
            output=Path("images") / source.stem / f"view_{index:02d}.png",
            seed=100 + index,
            modality="rgb",
            identity="0001",
            source_key=f"cam1/0001/{source.name}",
            view_index=index,
            hypothesis_id=f"h{index:02d}",
            hypothesis_weight=1 / views,
            imagination_contract_sha256="contract-test",
            camera=1,
            split="train",
        )
        for index in range(views)
    ]


def build_fixture(tmp_path: Path, views: int, configured: int | None = None):
    source = tmp_path / "source.jpg"
    Image.new("RGB", (16, 32), "gray").save(source)
    tasks = source_tasks(source, views)
    records = tmp_path / "records.jsonl"
    records.write_text(json.dumps({"views": views}) + "\n", encoding="utf-8")
    sd = tmp_path / "sd"
    pasd = tmp_path / "pasd"
    sd.mkdir()
    pasd.mkdir()
    config = GenerationConfig(
        pretrained_model_path=sd,
        pasd_model_path=pasd,
        output_root=tmp_path / "output",
        target_height=64,
        target_width=32,
        views_per_source=views if configured is None else configured,
    )
    prepare_build(config, records, tasks)
    return config, records, source, tasks


@pytest.mark.parametrize("views,configured", [(1, 1), (2, 0), (5, 5)])
def test_generation_and_manifest_support_fixed_or_dynamic_views(
    tmp_path: Path, views: int, configured: int
):
    config, records, _, tasks = build_fixture(tmp_path, views, configured)
    marker = generate_source_group(FakeGenerator(config), tasks, config.output_root, views)

    assert source_is_generated(tasks, config.output_root, config)
    assert source_is_validated(tasks, config.output_root, config)
    assert generated_source_count([tasks], config.output_root, config) == 1
    summary = consolidate_manifest(config.output_root, tasks, config, records)
    assert summary["complete"]
    assert summary["source_count"] == 1
    assert summary["view_count"] == views
    assert summary["views_per_source"] == configured
    assert marker["imagination_contract_sha256"] == "contract-test"


def test_changed_caption_invalidates_source_marker(tmp_path: Path):
    config, _, _, tasks = build_fixture(tmp_path, 1)
    generate_source_group(FakeGenerator(config), tasks, config.output_root, 1)
    changed = [GenerationTask(**{**tasks[0].__dict__, "caption": "changed"})]
    assert not source_is_generated(changed, config.output_root, config)


def test_corrupt_png_enters_repair_cycle(tmp_path: Path):
    config, records, _, tasks = build_fixture(tmp_path, 1)
    generate_source_group(FakeGenerator(config), tasks, config.output_root, 1)
    output = config.output_root / tasks[0].output
    payload = bytearray(output.read_bytes())
    payload[len(payload) // 2] ^= 1
    output.write_bytes(payload)

    assert source_is_generated(tasks, config.output_root, config)
    assert not source_is_validated(tasks, config.output_root, config)
    summary = consolidate_manifest(config.output_root, tasks, config, records)
    assert not summary["validated_complete"]
    assert invalidate_invalid_sources(config.output_root, summary) == [tasks[0].source_key]
    assert not source_is_generated(tasks, config.output_root, config)


def test_changed_records_cannot_publish_existing_outputs(tmp_path: Path):
    config, records, _, tasks = build_fixture(tmp_path, 1)
    generate_source_group(FakeGenerator(config), tasks, config.output_root, 1)
    records.write_text('{"changed":true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="config or records changed"):
        consolidate_manifest(config.output_root, tasks, config, records)
