from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from qwen_imagination.regional.schema import Region, SourceItem
from qwen_imagination.text_annotation.cli import (
    collect_sources,
    group_tracks,
    run,
    select_shard,
)
from qwen_imagination.text_annotation.config import TextAnnotationConfig
from qwen_imagination.text_annotation.pipeline import TextAnnotationPipeline
from qwen_imagination.text_annotation.reasoner import (
    normalize_annotation,
    normalize_hypotheses,
    sample_joint_text_worlds,
)
from qwen_imagination.text_annotation.track_anchor import (
    TrackAnchorTextAnnotationPipeline,
)


def _config(tmp_path: Path, *, modalities=("rgb",)) -> TextAnnotationConfig:
    assets = {}
    for name in (
        "qwen_model",
        "qwen_mmproj",
        "swinir_model",
        "yolo_pose",
        "schp_lip",
        "sam_vit_b",
    ):
        path = tmp_path / "assets" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        assets[name] = path
    schp = tmp_path / "schp"
    sam = tmp_path / "sam"
    swin = tmp_path / "swin"
    for path in (schp, sam, swin):
        path.mkdir(parents=True, exist_ok=True)
    return TextAnnotationConfig(
        schema_version=1,
        annotation_version="test-v1",
        dataset_root=tmp_path / "SYSU-MM01",
        output_root=tmp_path / "output",
        modalities=tuple(modalities),
        strategy="exact",
        assets=assets,
        roi={"device": "cpu", "schp_root": str(schp), "sam_root": str(sam)},
        swinir={"root": str(swin)},
        qwen={"endpoint": "http://127.0.0.1:1", "model_id": "fake"},
    ).validate()


def _region(region_id: str, x0: int) -> Region:
    mask = np.zeros((512, 256), dtype=bool)
    mask[100:180, x0 : x0 + 24] = True
    return Region(region_id, "carried_object", (x0, 100, x0 + 24, 180), mask)


def _annotation(regions: list[Region]) -> dict:
    return {
        "global": {
            "caption": "a person wearing a dark top and light trousers",
            "observations": [],
            "unresolved": [],
        },
        "regions": [
            {
                "region_id": region.region_id,
                "category": region.category,
                "region_summary": "a blurry carried region",
                "observations": [],
                "world_knowledge": [],
                "hypotheses": [
                    {
                        "description": f"{region.region_id} bag",
                        "probability": 0.7,
                        "basis": "mixed",
                        "observable_support": "dark hanging shape",
                        "uncertainty": "blur",
                    },
                    {
                        "description": f"{region.region_id} unresolved",
                        "probability": 0.3,
                        "basis": "unresolved",
                        "observable_support": "coarse pixels",
                        "uncertainty": "high",
                    },
                ],
                "unresolved": ["exact object type"],
            }
            for region in regions
        ],
    }


def _write_split(root: Path) -> None:
    (root / "exp").mkdir(parents=True, exist_ok=True)
    (root / "exp" / "train_id.txt").write_text("1", encoding="utf-8")
    (root / "exp" / "val_id.txt").write_text("2", encoding="utf-8")
    (root / "exp" / "test_id.txt").write_text("3", encoding="utf-8")
    for camera, identity, names in (
        ("cam1", "0001", ("0001.jpg", "0002.jpg")),
        ("cam3", "0001", ("0001.jpg",)),
        ("cam1", "0003", ("0001.jpg",)),
        ("cam3", "0003", ("0001.jpg", "0002.jpg")),
    ):
        directory = root / camera / identity
        directory.mkdir(parents=True, exist_ok=True)
        for name in names:
            Image.new("RGB", (128, 256), (80, 90, 100)).save(directory / name)


def test_hypotheses_are_normalized_without_adding_candidates():
    rows = normalize_hypotheses(
        [
            {"description": "specific bag", "probability": 3, "basis": "mixed"},
            {"description": "unresolved", "probability": 1, "basis": "unresolved"},
        ]
    )
    assert len(rows) == 2
    assert [row["probability"] for row in rows] == [0.75, 0.25]


def test_annotation_requires_every_selected_roi():
    regions = [_region("r0", 10), _region("r1", 50)]
    raw = _annotation(regions[:1])
    try:
        normalize_annotation(raw, regions)
    except ValueError as error:
        assert "omitted" in str(error)
    else:
        raise AssertionError("missing ROI must be rejected")


def test_joint_world_sampling_is_deterministic_and_probability_weighted():
    regions = _annotation([_region("r0", 10), _region("r1", 50)])["regions"]
    first = sample_joint_text_worlds(regions, sample_count=64, max_worlds=4, seed=17)
    second = sample_joint_text_worlds(regions, sample_count=64, max_worlds=4, seed=17)
    assert first == second
    assert first["sample_count"] == 64
    assert abs(sum(row["selected_weight"] for row in first["worlds"]) - 1.0) < 1e-9


def test_sysu_train_eval_iteration_and_sharding(tmp_path):
    config = _config(tmp_path, modalities=("rgb", "ir"))
    _write_split(config.dataset_root)
    train = collect_sources(config, "train")
    evaluation = collect_sources(config, "evaluation")
    assert [row.modality for row in train[:2]] == ["rgb", "ir"]
    assert len(train) == 3
    assert len(evaluation) == 3
    shard0 = select_shard(train, shard_index=0, num_shards=2)
    shard1 = select_shard(train, shard_index=1, num_shards=2)
    assert {row.source_key for row in shard0}.isdisjoint(
        {row.source_key for row in shard1}
    )
    assert len(shard0) + len(shard1) == len(train)
    tracks = group_tracks(train)
    assert sorted(len(track) for track in tracks) == [1, 2]


class _FakeSwin:
    def restore(self, image, modality):
        return image.resize((256, 512))


class _FakeROI:
    def regions(self, image, modality):
        return [_region(f"r{index}", 10 + index * 40) for index in range(4)]


class _FakeReasoner:
    model_id = "fake"

    def annotate(self, lr, swin, regions, *, modality, seed):
        return _annotation(regions), {"elapsed_seconds": 0.01, "usage": {}}


def test_pipeline_auto_roi_top3_and_text_only_result(tmp_path):
    config = _config(tmp_path)
    source_path = tmp_path / "source.jpg"
    Image.new("RGB", (128, 256), (70, 80, 90)).save(source_path)
    source = SourceItem("cam1/0001/0001.jpg", source_path, "0001", "cam1", "rgb")
    pipeline = TextAnnotationPipeline(
        config, swin=_FakeSwin(), roi=_FakeROI(), reasoner=_FakeReasoner()
    )
    record = pipeline.process(source)
    assert record["status"] == "complete"
    assert record["selected_region_ids"] == ["r0", "r1", "r2"]
    assert record["annotation"]["global"]["caption"].startswith("a person")
    assert len(record["annotation"]["regions"]) == 3
    assert record["sampled_text_worlds"]["sample_count"] == 64
    assert not config.output_root.exists()


def test_threshold_selection_keeps_all_qualified_regions_up_to_cap(tmp_path):
    config = _config(tmp_path)
    config.selected_region_count = 2
    config.max_selected_region_count = 5
    config.roi_selection_threshold = 0.6
    pipeline = TextAnnotationPipeline(
        config, swin=_FakeSwin(), roi=_FakeROI(), reasoner=_FakeReasoner()
    )
    regions = [_region("eyes", 5)] + [
        _region(f"r{index}", 10 + index * 30) for index in range(1, 6)
    ]
    scores = {
        "eyes": 0.1,
        "r1": 0.95,
        "r2": 0.8,
        "r3": 0.61,
        "r4": 0.6,
        "r5": 0.59,
    }
    for region in regions:
        region.u_swin_normalized = scores[region.region_id]
        region.u_blur = scores[region.region_id]
    selected, rule = pipeline._select_regions(regions)
    assert [region.region_id for region in selected] == [
        "eyes",
        "r1",
        "r2",
        "r3",
        "r4",
    ]
    assert "threshold-0.600-min-2-max-5" in rule


def test_deferred_empirical_mode_writes_no_self_reported_world_sampling(tmp_path):
    config = _config(tmp_path)
    config.probability_mode = "deferred_empirical"
    config.probability_spec = "semantic_imagination/MATHEMATICAL_SPEC.md"
    source_path = tmp_path / "source.jpg"
    Image.new("RGB", (128, 256), (70, 80, 90)).save(source_path)
    source = SourceItem("cam1/0001/0001.jpg", source_path, "0001", "cam1", "rgb")
    pipeline = TextAnnotationPipeline(
        config, swin=_FakeSwin(), roi=_FakeROI(), reasoner=_FakeReasoner()
    )
    record = pipeline.process(source)
    assert "sampled_text_worlds" not in record
    assert record["probability_design"] == {
        "mode": "deferred_empirical",
        "specification": "semantic_imagination/MATHEMATICAL_SPEC.md",
        "vlm_self_reported_probability": False,
        "status": "deferred",
    }
    assert record["annotation_provenance"]["vlm_visual_input"] == "swinir_only"


class _CountingPipeline:
    def __init__(self, config):
        self.config = config
        self.calls = 0

    def process(self, source):
        self.calls += 1
        regions = [_region(f"r{index}", 10 + index * 40) for index in range(3)]
        return {
            "schema_version": 1,
            "annotation_version": self.config.annotation_version,
            "run_signature": self.config.run_signature(),
            "source_key": source.source_key,
            "image": str(source.image),
            "identity": source.identity,
            "camera": source.camera,
            "modality": source.modality,
            "split": source.split,
            "status": "complete",
            "selected_region_ids": [region.region_id for region in regions],
            "annotation": _annotation(regions),
            "sampled_text_worlds": sample_joint_text_worlds(
                _annotation(regions)["regions"], sample_count=16, max_worlds=4, seed=3
            ),
        }


def test_run_writes_per_image_records_manifest_and_resumes(tmp_path):
    config = _config(tmp_path)
    _write_split(config.dataset_root)
    pipeline = _CountingPipeline(config)
    first = run(
        config,
        split="train",
        shard_index=0,
        num_shards=1,
        limit=1,
        fail_fast=True,
        overwrite=False,
        pipeline=pipeline,
    )
    assert first["complete"] is True
    assert pipeline.calls == 1
    second = run(
        config,
        split="train",
        shard_index=0,
        num_shards=1,
        limit=1,
        fail_fast=True,
        overwrite=False,
        pipeline=pipeline,
    )
    assert pipeline.calls == 1
    assert second["cached_source_count"] == 1
    metadata = config.output_root / "metadata" / "cam1" / "0001" / "0001.json"
    manifest = (
        config.output_root
        / "manifests"
        / "train.shard-00000-of-00001.jsonl"
    )
    assert metadata.is_file()
    assert manifest.is_file()
    row = json.loads(manifest.read_text(encoding="utf-8").strip())
    assert row["global"]["caption"].startswith("a person")
    assert row["sampled_text_worlds"]["worlds"]


class _CountingTrackPipeline:
    def __init__(self, config):
        self.config = config
        self.calls = 0

    def process_track(self, sources):
        self.calls += 1
        base = _CountingPipeline(self.config)
        anchor = sources[-1].source_key
        records = []
        for source in sources:
            record = base.process(source)
            record["annotation_provenance"] = {
                "strategy": "track_anchor",
                "anchor_source_key": anchor,
                "direct_vlm": source.source_key == anchor,
            }
            records.append(record)
        return records


def test_track_anchor_run_shards_by_track_and_caches_every_source(tmp_path):
    config = _config(tmp_path)
    config.strategy = "track_anchor"
    config.precomputed_swinir_root = tmp_path / "derived"
    config.precomputed_swinir_root.mkdir()
    config.validate()
    _write_split(config.dataset_root)
    pipeline = _CountingTrackPipeline(config)
    first = run(
        config,
        split="train",
        shard_index=0,
        num_shards=1,
        limit=1,
        fail_fast=True,
        overwrite=False,
        pipeline=pipeline,
    )
    assert first["track_count"] == 1
    assert first["source_count"] == 2
    assert first["vlm_request_count"] == 1
    assert pipeline.calls == 1
    second = run(
        config,
        split="train",
        shard_index=0,
        num_shards=1,
        limit=1,
        fail_fast=True,
        overwrite=False,
        pipeline=pipeline,
    )
    assert second["cached_source_count"] == 2
    assert second["cached_track_count"] == 1
    assert second["vlm_request_count"] == 0
    assert pipeline.calls == 1


class _FakePrecomputedStore:
    def image(self, source):
        with Image.open(source.image) as image:
            return image.convert("RGB").resize((256, 512))


class _CountingROI(_FakeROI):
    def __init__(self):
        self.calls = 0

    def regions(self, image, modality):
        self.calls += 1
        return super().regions(image, modality)


def test_track_prescan_cache_reuses_all_frames_and_rebuilds_only_anchor(tmp_path):
    config = _config(tmp_path)
    config.strategy = "track_anchor"
    config.precomputed_swinir_root = tmp_path / "derived"
    config.precomputed_swinir_root.mkdir()
    config.validate()
    directory = tmp_path / "track"
    directory.mkdir()
    sources = []
    for index, value in enumerate((60, 90), start=1):
        path = directory / f"{index:04d}.jpg"
        Image.new("RGB", (128, 256), (value, value, value)).save(path)
        sources.append(
            SourceItem(
                f"cam1/0001/{index:04d}.jpg",
                path,
                "0001",
                "cam1",
                "rgb",
            )
        )
    roi = _CountingROI()
    pipeline = TrackAnchorTextAnnotationPipeline(
        config,
        roi=roi,
        reasoner=_FakeReasoner(),
        store=_FakePrecomputedStore(),
    )
    first = pipeline.process_track(sources)
    assert roi.calls == 2
    assert len(first) == 2
    assert sum(row["annotation_provenance"]["direct_vlm"] for row in first) == 1
    second = pipeline.process_track(sources)
    assert roi.calls == 3
    assert [row["selected_region_ids"] for row in first] == [
        row["selected_region_ids"] for row in second
    ]
def test_text_annotation_pipeline_contains_no_pixel_generator_imports():
    package = Path(__file__).parents[1] / "qwen_imagination" / "text_annotation"
    text = "\n".join(path.read_text(encoding="utf-8") for path in package.glob("*.py"))
    assert "ExistingPASDBackend" not in text
    assert "diffusers" not in text
