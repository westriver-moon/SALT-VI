from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from semantic_imagination.regional.calibration import CalibrationWeights, calibrate_world_weights
from semantic_imagination.regional.cli import _qwen_server_command
from semantic_imagination.regional.config import Asset, RegionalConfig
from semantic_imagination.regional.manifest import consolidate_manifests
from semantic_imagination.regional.pipeline import RegionalImaginationPipeline
from semantic_imagination.regional.qwen import _json_object
from semantic_imagination.regional.schema import Candidate, Region, SourceItem, World
from semantic_imagination.regional.tta import qri_tta_specs


class FakeSwin:
    def restore(self, image, modality):
        return image.convert("RGB").resize((256, 512), Image.Resampling.BICUBIC)


class FakeROI:
    def regions(self, image, modality):
        specs = (
            ("eyes", "eyewear", (80, 40, 176, 110)),
            ("head", "headwear", (70, 10, 186, 100)),
            ("left_wrist", "wrist_accessory", (20, 210, 75, 280)),
            ("torso", "clothing_detail", (55, 120, 200, 330)),
        )
        regions = []
        for region_id, category, bbox in specs:
            mask = np.zeros((512, 256), dtype=bool)
            left, top, right, bottom = bbox
            mask[top:bottom, left:right] = True
            regions.append(Region(region_id, category, bbox, mask))
        return regions


class FakeReasoner:
    model_id = "fake-qwen"

    def propose(self, lr, swin, regions):
        sentinel = {
            "eyewear": "absent",
            "headwear": "absent",
            "wrist_accessory": "absent",
            "clothing_detail": "no_additional_detail",
        }
        detail = {
            "eyewear": "frame_style",
            "headwear": "cap",
            "wrist_accessory": "watch",
            "clothing_detail": "pattern",
        }
        return {
            region.region_id: [
                Candidate(sentinel[region.category], sentinel[region.category], evidence_source="abstain"),
                Candidate(detail[region.category], detail[region.category], evidence_source="compatible_prior_only"),
            ]
            for region in regions
        }

    def sample_world(self, lr, swin, regions, proposals, seed):
        offset = seed % 3
        return {
            region.region_id: proposals[region.region_id][
                1 if (index + offset) % 3 else 0
            ]
            for index, region in enumerate(regions)
        }

    def critique(self, lr, swin, regions, assignments):
        return [
            {
                region_id: {
                    "label": candidate.evidence_source,
                    "score": 0.5,
                    "evidence": "fake check",
                }
                for region_id, candidate in world.items()
            }
            for world in assignments
        ]


class BrokenReasoner(FakeReasoner):
    def propose(self, lr, swin, regions):
        raise RuntimeError("offline fake failure")


class FakePASD:
    def generate(self, control_path, captions, seeds, modality):
        with Image.open(control_path) as image:
            base = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        outputs = []
        for index in range(len(captions)):
            candidate = base.copy()
            candidate[..., index % 3] = np.clip(candidate[..., index % 3] + 12, 0, 255)
            outputs.append(Image.fromarray(candidate))
        return outputs


class FakeIdentity:
    def feature(self, image, modality):
        values = np.asarray(image, dtype=np.float32).mean(axis=(0, 1)) + 1.0
        return values


def config(tmp_path: Path) -> RegionalConfig:
    return RegionalConfig(
        schema_version=1,
        dataset_root=tmp_path / "SYSU-MM01",
        output_root=tmp_path / "qri",
    ).validate()


def source(tmp_path: Path) -> SourceItem:
    image_path = tmp_path / "SYSU-MM01" / "cam1" / "0001" / "0001.jpg"
    image_path.parent.mkdir(parents=True)
    array = np.zeros((256, 128, 3), dtype=np.uint8)
    array[..., 0] = np.arange(128, dtype=np.uint8)[None, :]
    array[..., 1] = np.arange(256, dtype=np.uint8)[:, None]
    Image.fromarray(array).save(image_path)
    return SourceItem("cam1/0001/0001.jpg", image_path, "0001", "cam1", "rgb")


def pipeline(tmp_path: Path, reasoner=None):
    return RegionalImaginationPipeline(
        config(tmp_path),
        swin=FakeSwin(),
        roi=FakeROI(),
        reasoner=reasoner or FakeReasoner(),
        pasd=FakePASD(),
        identity=FakeIdentity(),
    )


def pipeline_with_stats(tmp_path: Path, stats: dict):
    return RegionalImaginationPipeline(
        config(tmp_path),
        swin=FakeSwin(),
        roi=FakeROI(),
        reasoner=FakeReasoner(),
        pasd=FakePASD(),
        identity=FakeIdentity(),
        category_stats=stats,
    )


def test_qri_tta_budget_is_preregistered():
    assert len(qri_tta_specs()) == 12
    assert [spec.name for spec in qri_tta_specs()].count("shift") == 4


def test_qwen_json_parser_ignores_reasoning_braces_before_final_object():
    response = (
        "compare options {A, B}; neither fragment is JSON.\n"
        "Final answer:\n```json\n"
        '{"regions":[{"region_id":"eyes","candidates":[]}]}\n'
        "```"
    )
    assert _json_object(response) == {
        "regions": [{"region_id": "eyes", "candidates": []}]
    }


def test_regional_pipeline_materializes_joint_worlds_and_three_weights(tmp_path: Path):
    runtime = pipeline(tmp_path)
    record = runtime.process(source(tmp_path), allow_fallback=False)

    assert record["fallback"] is False
    assert len(record["selected_region_ids"]) == 3
    assert 1 <= len(record["worlds"]) <= 5
    assert sum(world["uniform_weight"] for world in record["worlds"]) == pytest.approx(1)
    assert sum(world["proposal_weight"] for world in record["worlds"]) == pytest.approx(1)
    assert sum(world["posterior_weight"] for world in record["worlds"]) == pytest.approx(1)
    assert all((runtime.config.output_root / world["output"]).is_file() for world in record["worlds"])

    summaries = consolidate_manifests(
        runtime.config.output_root,
        [record],
        expected_source_count=1,
        build_sha256=runtime.build_sha256,
    )
    assert set(summaries) == {"uniform", "proposal", "posterior"}
    posterior = runtime.config.output_root / "manifests" / "manifest.posterior.jsonl"
    rows = [json.loads(line) for line in posterior.read_text().splitlines()]
    assert sum(row["hypothesis_weight"] for row in rows) == pytest.approx(1)


def test_regional_failure_falls_back_to_one_swin_world(tmp_path: Path):
    runtime = pipeline(tmp_path, BrokenReasoner())
    record = runtime.process(source(tmp_path), allow_fallback=True)

    assert record["fallback"] is True
    assert len(record["worlds"]) == 1
    world = record["worlds"][0]
    assert world["uniform_weight"] == world["proposal_weight"] == world["posterior_weight"] == 1
    assert (runtime.config.output_root / world["output"]).is_file()


def test_posterior_calibration_prefers_consistent_world():
    good = World("good", [], 1, 0.5, proposal_weight=0.5, e_lr=0.01, e_id=0.01, e_edit=0.01)
    bad = World("bad", [], 1, 0.5, proposal_weight=0.5, e_lr=0.2, e_id=0.2, e_edit=0.2)
    calibrate_world_weights([good, bad], CalibrationWeights())
    assert good.posterior_weight > bad.posterior_weight
    assert good.posterior_weight + bad.posterior_weight == pytest.approx(1)


def test_qwen_server_command_always_loads_projector_and_quantized_cache(tmp_path: Path):
    runtime = config(tmp_path)
    runtime.assets.update(
        {
            "qwen_model": Asset(tmp_path / "model.gguf", "a" * 64),
            "qwen_mmproj": Asset(tmp_path / "mmproj.gguf", "b" * 64),
        }
    )
    runtime.qwen.update(
        {
            "server_binary": tmp_path / "llama-server",
            "model_id": "third-party-test",
            "context_size": 8192,
        }
    )
    command = _qwen_server_command(runtime)
    assert command[command.index("--mmproj") + 1].endswith("mmproj.gguf")
    assert command[command.index("--cache-type-k") + 1] == "q8_0"
    assert command[command.index("--cache-type-v") + 1] == "q8_0"
    assert command[command.index("--reasoning-budget") + 1] == "1024"
    assert command[command.index("--flash-attn") + 1] == "on"
    assert command[command.index("--parallel") + 1] == "1"


def test_category_statistics_are_part_of_cache_build_identity(tmp_path: Path):
    first = pipeline_with_stats(tmp_path, {"eyewear": {"median": 0.1, "iqr": 0.2}})
    second = pipeline_with_stats(tmp_path, {"eyewear": {"median": 0.2, "iqr": 0.2}})
    assert first.build_sha256 != second.build_sha256
