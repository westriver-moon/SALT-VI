from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from qwen_imagination.regional.calibration import (
    CalibrationWeights,
    calibrate_world_weights,
)
from qwen_imagination.regional.cli import _qwen_server_command
from qwen_imagination.regional.config import Asset, RegionalConfig
from qwen_imagination.regional.manifest import consolidate_manifests
from qwen_imagination.regional.pipeline import RegionalImaginationPipeline
from qwen_imagination.regional.qwen import (
    LlamaServerQwenReasoner,
    _json_object,
    sample_joint_worlds,
)
from qwen_imagination.regional.qwen_v2 import ImaginativeQwenReasoner
from qwen_imagination.regional.roi import HumanROIGenerator
from qwen_imagination.regional.runtime import (
    PASDGeneration,
    PASDGenerationOptions,
    _autocast_context,
    validate_qri_pasd_generation,
)
from qwen_imagination.regional.composite import roi_crop_box
from qwen_imagination.regional.schema import Candidate, Region, SourceItem, World
from qwen_imagination.regional.tta import qri_tta_specs
from qwen_imagination.regional.visual_context import roi_comparison_board


def test_identity_runtime_enables_autocast_only_for_cuda():
    calls = []
    sentinel = object()

    class Amp:
        @staticmethod
        def autocast(device_type, *, enabled):
            calls.append((device_type, enabled))
            return sentinel

    class Torch:
        amp = Amp()

    class Device:
        def __init__(self, device_type):
            self.type = device_type

    assert _autocast_context(Torch(), Device("cuda")) is sentinel
    assert _autocast_context(Torch(), Device("cpu")) is sentinel
    assert calls == [("cuda", True), ("cuda", False)]


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
                Candidate(
                    sentinel[region.category],
                    sentinel[region.category],
                    evidence_source="abstain",
                ),
                Candidate(
                    detail[region.category],
                    detail[region.category],
                    evidence_source="compatible_prior_only",
                ),
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
    def __init__(self):
        self.calls = []

    def generate(self, control_path, captions, seeds, modality, options=None):
        self.calls.append(
            {
                "control_path": Path(control_path),
                "captions": list(captions),
                "seeds": list(seeds),
                "modality": modality,
                "options": options,
            }
        )
        with Image.open(control_path) as image:
            base = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        outputs = []
        for index in range(len(captions)):
            candidate = base.copy()
            candidate[..., index % 3] = np.clip(candidate[..., index % 3] + 12, 0, 255)
            outputs.append(Image.fromarray(candidate))
        return PASDGeneration(
            images=outputs,
            geometry={
                "mode": "direct_rewrite",
                "source_size": [256, 512],
                "target_size": [256, 512],
                "resized_size": [256, 512],
                "padding": [0, 0, 0, 0],
                "transform": {
                    "scale_x": 1.0,
                    "scale_y": 1.0,
                    "offset_x": 0.0,
                    "offset_y": 0.0,
                },
                "background_restoration": False,
            },
        )


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


def config_v2(tmp_path: Path) -> RegionalConfig:
    return RegionalConfig(
        schema_version=2,
        plugin_version="qri-v2",
        dataset_root=tmp_path / "SYSU-MM01",
        output_root=tmp_path / "qri-v2",
        proposal_rounds=3,
        coverage_sampling=True,
        ensure_editing_world_per_region=True,
        roi_board_size_px=512,
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


class ScriptedImaginativeReasoner(ImaginativeQwenReasoner):
    def __init__(self, responses):
        super().__init__(proposal_rounds=3, roi_board_size_px=512)
        self.responses = list(responses)
        self.instructions = []

    def _complete(self, content, instruction, **kwargs):
        self.instructions.append(instruction)
        if not self.responses:
            raise AssertionError("scripted Qwen response exhausted")
        return self.responses.pop(0)


class ScriptedV1Reasoner(LlamaServerQwenReasoner):
    def __init__(self, responses):
        super().__init__()
        self.responses = list(responses)
        self.instructions = []

    def _complete(self, content, instruction, **kwargs):
        self.instructions.append(instruction)
        return self.responses.pop(0)


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
    assert sum(world["uniform_weight"] for world in record["worlds"]) == pytest.approx(
        1
    )
    assert sum(world["proposal_weight"] for world in record["worlds"]) == pytest.approx(
        1
    )
    assert sum(
        world["posterior_weight"] for world in record["worlds"]
    ) == pytest.approx(1)
    assert all(
        (runtime.config.output_root / world["output"]).is_file()
        for world in record["worlds"]
    )
    assert record["pasd_geometry"]["mode"] == "direct_rewrite"

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
    assert (
        world["uniform_weight"]
        == world["proposal_weight"]
        == world["posterior_weight"]
        == 1
    )
    assert (runtime.config.output_root / world["output"]).is_file()


def test_posterior_calibration_prefers_consistent_world():
    good = World(
        "good", [], 1, 0.5, proposal_weight=0.5, e_lr=0.01, e_id=0.01, e_edit=0.01
    )
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


def test_qri_v2_isolated_config_and_manifest_identity(tmp_path: Path):
    pasd = FakePASD()
    runtime = RegionalImaginationPipeline(
        config_v2(tmp_path),
        swin=FakeSwin(),
        roi=FakeROI(),
        reasoner=FakeReasoner(),
        pasd=pasd,
        identity=FakeIdentity(),
    )
    record = runtime.process(source(tmp_path), allow_fallback=False)
    assert record["plugin"] == "qwen-regional-imagination-v2"
    assert runtime.config.output_root.name == "qri-v2"
    assert record["pasd_geometry"]["mode"] == "roi_direct_rewrite"
    assert record["pasd_geometry"]["pasd_target_size"] == [256, 512]
    assert any(
        any(
            assignment["state"] not in {"absent", "no_additional_detail", "unresolved"}
            for assignment in world["assignments"]
        )
        for world in record["worlds"]
    )
    summaries = consolidate_manifests(
        runtime.config.output_root,
        [record],
        expected_source_count=1,
        build_sha256=runtime.build_sha256,
    )
    assert summaries["posterior"]["plugin"] == "qwen-regional-imagination-v2"
    selected = [
        region
        for region in record["regions"]
        if region["region_id"] in record["selected_region_ids"]
    ]
    assert all(region["u_qwen"] == region["u_qwen_compatible"] for region in selected)
    assert pasd.calls
    assert all(len(call["captions"]) == 1 for call in pasd.calls)
    assert all(
        Image.open(call["control_path"]).size == (256, 512) for call in pasd.calls
    )
    assert all(
        isinstance(call["options"], PASDGenerationOptions) for call in pasd.calls
    )
    assert all(call["options"].guidance_scale == 9.0 for call in pasd.calls)
    assert all(call["options"].conditioning_scale == 0.5 for call in pasd.calls)
    assert all(
        "new accessories" not in call["options"].negative_prompt for call in pasd.calls
    )
    assert any("no glasses" in call["options"].negative_prompt for call in pasd.calls)
    realized = [
        realization
        for world in record["worlds"]
        for realization in world["realizations"]
    ]
    assert realized
    assert all(realization["crop_box_xyxy"] for realization in realized)
    assert all(
        (runtime.config.output_root / realization["pasd_output"]).is_file()
        for realization in realized
    )


def test_v2_proposer_guarantees_positive_absent_and_unresolved_without_abstention_bias():
    response = {
        "regions": [
            {
                "region_id": "eyes",
                "candidates": [
                    {
                        "state": "absent",
                        "value": "no eyewear",
                        "evidence": "not resolved",
                        "evidence_source": "prior_plausible",
                    },
                    {
                        "state": "unresolved",
                        "value": "unresolved",
                        "evidence": "face turned away",
                        "evidence_source": "unresolved",
                    },
                ],
            }
        ]
    }
    reasoner = ScriptedImaginativeReasoner([response, response, response])
    lr = Image.new("RGB", (128, 256), "gray")
    swin = lr.resize((256, 512))
    mask = np.ones((512, 256), dtype=bool)
    region = Region("eyes", "eyewear", (64, 12, 204, 104), mask)
    proposals = reasoner.propose(lr, swin, [region])
    states = {candidate.state for candidate in proposals["eyes"]}
    assert {"eyewear_present", "absent", "unresolved"} <= states
    assert all(
        "Prefer abstention" not in instruction for instruction in reasoner.instructions
    )
    assert all(
        "Never equate unobservable with absent" in instruction
        for instruction in reasoner.instructions
    )


def test_v1_unresolved_fallback_is_not_silently_rewritten_as_absent():
    response = {
        "regions": [
            {
                "region_id": "eyes",
                "candidates": [
                    {
                        "state": "eyewear_type",
                        "value": "possible glasses",
                        "evidence": "faint temple line",
                        "evidence_source": "compatible_prior_only",
                    }
                ],
            }
        ]
    }
    reasoner = ScriptedV1Reasoner([response])
    lr = Image.new("RGB", (128, 256), "gray")
    swin = lr.resize((256, 512))
    region = Region(
        "eyes", "eyewear", (64, 12, 204, 104), np.ones((512, 256), dtype=bool)
    )
    proposals = reasoner.propose(lr, swin, [region])
    states = {candidate.state for candidate in proposals["eyes"]}
    assert "eyewear_type" in states
    assert "no_additional_detail" in states
    assert "absent" not in states
    assert all(
        "Prefer abstention" not in instruction for instruction in reasoner.instructions
    )


def test_v2_coverage_schedule_exposes_every_candidate_before_free_sampling():
    lr = Image.new("RGB", (128, 256), "gray")
    swin = lr.resize((256, 512))
    mask = np.ones((512, 256), dtype=bool)
    region = Region("eyes", "eyewear", (64, 12, 204, 104), mask)
    proposals = {
        "eyes": [
            Candidate("eyewear_present", "possible glasses"),
            Candidate("absent", "absent"),
            Candidate("unresolved", "unresolved"),
        ]
    }
    samples = sample_joint_worlds(
        FakeReasoner(),
        lr,
        swin,
        [region],
        proposals,
        3,
        7,
        coverage_first=True,
    )
    assert {sample.assignments["eyes"].state for sample in samples} == {
        "eyewear_present",
        "absent",
        "unresolved",
    }
    assert all(sample.origin == "coverage" for sample in samples)


def test_v2_roi_board_has_tight_and_context_views_without_changing_canvas():
    lr = Image.new("RGB", (128, 256), "gray")
    swin = lr.resize((256, 512))
    mask = np.ones((512, 256), dtype=bool)
    region = Region("eyes", "eyewear", (64, 12, 204, 104), mask)
    board = roi_comparison_board(lr, swin, region, size_px=512)
    assert board.size == (512, 512)


def test_v2_roi_crop_is_in_bounds_contains_target_and_preserves_pasd_aspect():
    crop = roi_crop_box(
        (92, 0, 189, 45),
        (256, 512),
        context_scale=1.75,
        target_size=(256, 512),
    )
    left, top, right, bottom = crop
    assert 0 <= left <= 92 < 189 <= right <= 256
    assert 0 <= top <= 0 < 45 <= bottom <= 512
    assert (right - left) / (bottom - top) == pytest.approx(0.5, abs=0.01)


def test_eyewear_roi_is_a_tight_eye_band_not_a_sam_face_mask():
    class Pose:
        def infer(self, image):
            return {
                "bbox_xyxy": (10, 0, 110, 200),
                "keypoints": {
                    "left_eye": (42, 20, 0.9),
                    "right_eye": (78, 20, 0.9),
                },
            }

    class Parsing:
        def infer(self, image):
            return np.zeros((image.height, image.width), dtype=np.uint8)

    class SAM:
        def refine(self, image, bbox, seed_mask):
            return np.ones((image.height, image.width), dtype=bool)

    regions = HumanROIGenerator(Pose(), Parsing(), SAM(), strict=True).regions(
        Image.new("RGB", (128, 256), "gray"), "rgb"
    )
    eyes = next(region for region in regions if region.region_id == "eyes")
    left, top, right, bottom = eyes.bbox_xyxy
    assert top > 0
    assert bottom - top <= 12
    assert int(eyes.mask.sum()) == (right - left) * (bottom - top)


def test_v2_localized_pasd_defaults_reject_contradictory_negative_prompt(
    tmp_path: Path,
):
    runtime = config_v2(tmp_path)
    assert runtime.pasd["realization"] == "roi-direct-rewrite-then-soft-mask-composite"
    assert runtime.pasd["guidance_scale"] == 9.0
    assert runtime.pasd["conditioning_scale"] == 0.5
    runtime.pasd["localized_negative_prompt"] += ", new accessories"
    with pytest.raises(ValueError, match="cannot suppress"):
        runtime.validate()


def test_qri_rejects_size_only_alignment_without_identity_coordinates():
    generation = PASDGeneration(
        images=[Image.new("RGB", (256, 512), "gray")],
        geometry={
            "mode": "person_fit_blurred_background",
            "source_size": [256, 512],
            "target_size": [256, 512],
        },
    )
    with pytest.raises(ValueError, match="direct_rewrite"):
        validate_qri_pasd_generation(generation, (256, 512))
