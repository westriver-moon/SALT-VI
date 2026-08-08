import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from pasd_offline.config import GenerationConfig
from pasd_offline.contracts import (
    prepare_contracts,
    prepare_dataset_scope,
    prepare_generation_identity,
)
from pasd_offline.generate import (
    consolidate_manifest,
    generate_batch,
    generate_source_group,
    generate_task,
    invalidate_invalid_sources,
    source_is_generated,
    source_is_validated,
)
from pasd_offline.scheduler import generated_source_count
from pasd_offline.tasks import GenerationTask, load_tasks


TEST_ENVIRONMENT = {
    "requirements_lock": {"sha256": "lock-one"},
    "python": {"implementation": "CPython", "version": "3.10.19"},
    "packages": {
        "torch": "2.2.2+cu118",
        "diffusers": "0.29.2",
        "transformers": "4.37.2",
        "xformers": "0.0.25.post1+cu118",
    },
    "cuda_runtime": "11.8",
    "cudnn": 8700,
    "gpus": [{"name": "NVIDIA GeForce RTX 3090", "compute_capability": "8.6"}],
}


class FakeGenerator:
    config = SimpleNamespace(png_compress_level=4)

    def generate(
        self, image_path: Path, caption: str, seed: int, modality: str
    ) -> Image.Image:
        if modality == "ir":
            return Image.new("L", (32, 64), 96).convert("RGB")
        return Image.new("RGB", (32, 64), (128, 32, 16))


class FakeMultiviewGenerator:
    def __init__(self, config):
        self.config = config

    def generate_views(self, image_path, captions, seeds, modality, batch_size):
        pixels = np.zeros(
            (self.config.target_height, self.config.target_width, 3), dtype=np.uint8
        )
        pixels[:, :, 0] = np.arange(self.config.target_width, dtype=np.uint8)
        image = Image.fromarray(pixels, "RGB")
        return [image.copy() for _ in captions], {"mode": "test"}


def five_view_tasks(source: Path, source_key: str = "cam1/0001/0001.jpg"):
    stem = Path(source_key).with_suffix("")
    return [
        GenerationTask(
            image=source,
            caption=f"caption {index}",
            output=Path("images") / stem / f"view_{index:02d}.png",
            seed=100 + index,
            modality="rgb",
            identity=Path(source_key).parts[1],
            source_key=source_key,
            view_index=index,
            camera=1,
            split="train",
            task_kind="five_view",
        )
        for index in range(5)
    ]


def contract_fixture(tmp_path: Path, source: Path, tasks: list[GenerationTask]):
    sd = tmp_path / "sd"
    pasd = tmp_path / "pasd"
    sd.mkdir(exist_ok=True)
    pasd.mkdir(exist_ok=True)
    (sd / "model.bin").write_bytes(b"sd-one")
    (pasd / "model.bin").write_bytes(b"pasd-one")
    detector = tmp_path / "yolo.pt"
    detector.write_bytes(b"yolo-one")
    implementation = tmp_path / "implementation"
    (implementation / "pasd_offline").mkdir(parents=True, exist_ok=True)
    (implementation / "vendor" / "pasd").mkdir(parents=True, exist_ok=True)
    (implementation / "pasd_offline" / "runtime.py").write_text(
        "VERSION = 1\n", encoding="utf-8"
    )
    (implementation / "vendor" / "pasd" / "pipeline.py").write_text(
        "VERSION = 1\n", encoding="utf-8"
    )
    records = tmp_path / "records.jsonl"
    records.write_text('{"scope":"pilot"}\n', encoding="utf-8")
    config = GenerationConfig(
        pretrained_model_path=sd,
        pasd_model_path=pasd,
        person_detector_model=detector,
        output_root=tmp_path / "output",
        target_height=64,
        target_width=32,
    )
    prepare_contracts(config, records, tasks, implementation, TEST_ENVIRONMENT)
    return config, records, implementation


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


def test_manifest_uses_current_source_and_dataset_contracts(tmp_path: Path):
    source = tmp_path / "source.jpg"
    Image.new("RGB", (16, 32), "gray").save(source)
    tasks = five_view_tasks(source)
    config, records, implementation = contract_fixture(tmp_path, source, tasks)
    generate_source_group(FakeMultiviewGenerator(config), tasks, config.output_root, 5)

    assert source_is_generated(tasks, config.output_root, config)
    assert source_is_validated(tasks, config.output_root, config)
    assert generated_source_count([tasks], config.output_root, config) == 1
    summary = consolidate_manifest(
        config.output_root,
        tasks,
        config,
        records,
        implementation_root=implementation,
        environment=TEST_ENVIRONMENT,
    )
    assert summary["complete"]
    assert summary["source_count"] == 1
    assert summary["view_count"] == 5

    changed = [*tasks]
    changed[0] = GenerationTask(**{**changed[0].__dict__, "caption": "changed caption"})
    assert not source_is_generated(changed, config.output_root, config)


def test_corrupt_png_enters_repair_cycle_and_regenerates(tmp_path: Path):
    source = tmp_path / "source.jpg"
    Image.new("RGB", (16, 32), "gray").save(source)
    tasks = five_view_tasks(source)
    config, records, implementation = contract_fixture(tmp_path, source, tasks)
    generator = FakeMultiviewGenerator(config)
    generate_source_group(generator, tasks, config.output_root, 5)
    output = config.output_root / tasks[0].output
    payload = bytearray(output.read_bytes())
    payload[len(payload) // 2] ^= 1
    output.write_bytes(payload)

    assert source_is_generated(tasks, config.output_root, config)
    assert not source_is_validated(tasks, config.output_root, config)
    failed = consolidate_manifest(config.output_root, tasks, config, records)
    assert failed["generated_complete"]
    assert not failed["validated_complete"]
    assert invalidate_invalid_sources(config.output_root, failed) == [tasks[0].source_key]
    assert not source_is_generated(tasks, config.output_root, config)

    generate_source_group(generator, tasks, config.output_root, 5)
    repaired = consolidate_manifest(
        config.output_root,
        tasks,
        config,
        records,
        implementation_root=implementation,
        environment=TEST_ENVIRONMENT,
    )
    assert repaired["validated_complete"]


def test_pilot_outputs_survive_full_dataset_scope(tmp_path: Path):
    pilot_source = tmp_path / "pilot.jpg"
    full_source = tmp_path / "full.jpg"
    Image.new("RGB", (16, 32), "gray").save(pilot_source)
    Image.new("RGB", (16, 32), "white").save(full_source)
    pilot_tasks = five_view_tasks(pilot_source)
    config, pilot_records, implementation = contract_fixture(
        tmp_path, pilot_source, pilot_tasks
    )
    generation_sha = config.generation_identity_sha256
    generate_source_group(
        FakeMultiviewGenerator(config), pilot_tasks, config.output_root, 5
    )

    full_tasks = pilot_tasks + five_view_tasks(full_source, "cam1/0002/0001.jpg")
    full_records = tmp_path / "full-records.jsonl"
    full_records.write_text('{"scope":"full"}\n', encoding="utf-8")
    pilot_scope_sha = config.dataset_scope_sha256
    prepare_dataset_scope(config, full_records, full_tasks)

    assert config.generation_identity_sha256 == generation_sha
    assert config.dataset_scope_sha256 != pilot_scope_sha
    assert source_is_generated(pilot_tasks, config.output_root, config)
    assert source_is_validated(pilot_tasks, config.output_root, config)

    changed_environment = {**TEST_ENVIRONMENT, "cuda_runtime": "12.1"}
    prepare_generation_identity(config, implementation, changed_environment)
    assert config.generation_identity_sha256 != generation_sha
    assert not source_is_generated(pilot_tasks, config.output_root, config)


def test_final_publish_rechecks_generation_and_dataset_identities(tmp_path: Path):
    source = tmp_path / "source.jpg"
    Image.new("RGB", (16, 32), "gray").save(source)
    tasks = five_view_tasks(source)
    config, records, implementation = contract_fixture(tmp_path, source, tasks)
    generate_source_group(FakeMultiviewGenerator(config), tasks, config.output_root, 5)

    (config.pasd_model_path / "model.bin").write_bytes(b"pasd-two")
    with pytest.raises(ValueError, match="generation identity changed"):
        consolidate_manifest(
            config.output_root,
            tasks,
            config,
            records,
            implementation_root=implementation,
            environment=TEST_ENVIRONMENT,
        )

    (config.pasd_model_path / "model.bin").write_bytes(b"pasd-one")
    records.write_text('{"scope":"changed"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="dataset scope changed"):
        consolidate_manifest(
            config.output_root,
            tasks,
            config,
            records,
            implementation_root=implementation,
            environment=TEST_ENVIRONMENT,
        )


def test_source_changed_after_scope_creation_cannot_enter_final_manifest(tmp_path: Path):
    source = tmp_path / "source.jpg"
    Image.new("RGB", (16, 32), "gray").save(source)
    tasks = five_view_tasks(source)
    config, records, implementation = contract_fixture(tmp_path, source, tasks)

    Image.new("RGB", (16, 32), "white").save(source)
    generate_source_group(FakeMultiviewGenerator(config), tasks, config.output_root, 5)

    with pytest.raises(ValueError, match="dataset scope changed"):
        consolidate_manifest(
            config.output_root,
            tasks,
            config,
            records,
            implementation_root=implementation,
            environment=TEST_ENVIRONMENT,
        )

def test_generic_five_caption_batch_uses_task_manifest(tmp_path: Path, monkeypatch):
    source = tmp_path / "source.png"
    Image.new("RGB", (16, 32), "gray").save(source)
    records = tmp_path / "records.jsonl"
    records.write_text(
        json.dumps(
            {
                "image": str(source),
                "captions": [f"caption {index}" for index in range(5)],
                "output": "images/source.png",
                "modality": "rgb",
            }
        ),
        encoding="utf-8",
    )
    tasks = load_tasks(records, "all", 13)
    config, _, implementation = contract_fixture(tmp_path, source, tasks)
    prepare_dataset_scope(config, records, tasks)
    monkeypatch.setitem(
        sys.modules,
        "pasd_offline.runtime",
        SimpleNamespace(PASDGenerator=lambda unused_config: FakeGenerator()),
    )

    entries = generate_batch(
        config,
        tasks,
        records,
        implementation_root=implementation,
        environment=TEST_ENVIRONMENT,
    )

    assert len(entries) == 5
    summary = json.loads((config.output_root / "manifest.json").read_text(encoding="utf-8"))
    assert summary["task_count"] == 5
    assert not (config.output_root / "metadata").exists()
    assert not (config.output_root / ".locks").exists()
