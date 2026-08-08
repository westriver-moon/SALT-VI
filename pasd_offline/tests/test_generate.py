from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from pasd_offline.config import GenerationConfig
from pasd_offline.generate import (
    consolidate_manifest,
    generate_source_group,
    generate_task,
    source_is_complete,
)
from pasd_offline.scheduler import completed_source_count
from pasd_offline.tasks import GenerationTask


class FakeGenerator:
    config = SimpleNamespace(png_compress_level=4)

    def generate(
        self, image_path: Path, caption: str, seed: int, modality: str
    ) -> Image.Image:
        assert caption == "a person wearing red"
        assert seed == 13
        assert modality in ("rgb", "ir")
        if modality == "ir":
            return Image.new("L", (32, 64), 96).convert("RGB")
        return Image.new("RGB", (32, 64), (128, 32, 16))


def test_generate_task_writes_public_record(tmp_path: Path):
    source = tmp_path / "source.png"
    Image.new("RGB", (16, 32), (0, 0, 0)).save(source)
    task = GenerationTask(
        image=source,
        caption="a person wearing red",
        output=Path("images/rgb/source.png"),
        seed=13,
        modality="rgb",
        identity="1",
    )

    entry = generate_task(FakeGenerator(), task, tmp_path / "public-data")

    output = Path(entry["output"])
    assert output.is_file()
    assert entry["caption"] == "a person wearing red"
    assert entry["output_size"] == [32, 64]
    assert len(entry["input_sha256"]) == 64
    assert len(entry["output_sha256"]) == 64


def test_generate_task_propagates_ir_modality(tmp_path: Path):
    source = tmp_path / "source.png"
    Image.new("RGB", (16, 32), (0, 0, 0)).save(source)
    task = GenerationTask(
        image=source,
        caption="a person wearing red",
        output=Path("images/ir/source.png"),
        seed=13,
        modality="ir",
        identity="1",
    )
    entry = generate_task(FakeGenerator(), task, tmp_path / "public-data")
    with Image.open(entry["output"]) as image:
        pixels = np.asarray(image)
    assert np.array_equal(pixels[..., 0], pixels[..., 1])
    assert np.array_equal(pixels[..., 1], pixels[..., 2])


class FakeMultiviewGenerator:
    def __init__(self, config):
        self.config = config

    def generate_views(self, image_path, captions, seeds, modality, batch_size):
        pixels = np.zeros((self.config.target_height, self.config.target_width, 3), dtype=np.uint8)
        pixels[:, :, 0] = np.arange(self.config.target_width, dtype=np.uint8)
        image = Image.fromarray(pixels, "RGB")
        return [image.copy() for _ in captions], {"mode": "test"}


def five_view_tasks(source: Path):
    return [
        GenerationTask(
            image=source,
            caption=f"caption {index}",
            output=Path(f"images/cam1/0001/0001/view_{index:02d}.png"),
            seed=100 + index,
            modality="rgb",
            identity="0001",
            source_key="cam1/0001/0001.jpg",
            view_index=index,
            camera=1,
            split="train",
        )
        for index in range(5)
    ]


def test_manifest_uses_only_current_generation_contract(tmp_path: Path):
    source = tmp_path / "source.jpg"
    Image.new("RGB", (16, 32), "gray").save(source)
    output_root = tmp_path / "output"
    config = GenerationConfig(
        pretrained_model_path=tmp_path / "sd",
        pasd_model_path=tmp_path / "pasd",
        output_root=output_root,
        target_height=64,
        target_width=32,
    )
    tasks = five_view_tasks(source)
    generate_source_group(FakeMultiviewGenerator(config), tasks, output_root, batch_size=5)
    assert source_is_complete(tasks, output_root, config)
    assert completed_source_count([tasks], output_root, config) == 1

    unrelated = output_root / "metadata" / "cam9" / "9999" / "old.json"
    unrelated.parent.mkdir(parents=True)
    unrelated.write_text('{"views": []}', encoding="utf-8")
    summary = consolidate_manifest(output_root, tasks, config)
    assert summary["complete"]
    assert summary["source_count"] == 1
    assert summary["view_count"] == 5

    changed = [*tasks]
    changed[0] = GenerationTask(**{**changed[0].__dict__, "caption": "changed caption"})
    assert not source_is_complete(changed, output_root, config)
    assert completed_source_count([changed], output_root, config) == 0
    (output_root / tasks[0].output).unlink()
    assert not source_is_complete(tasks, output_root, config)
