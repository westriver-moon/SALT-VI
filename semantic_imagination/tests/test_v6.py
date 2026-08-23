from __future__ import annotations

import pytest

from semantic_imagination.v6 import (
    AtomicDetail,
    BackendDescriptor,
    BackendResult,
    JointWorld,
    Observation,
    RegionObservation,
    RegionSpec,
    SourceSpec,
    V6Config,
    V6Engine,
    V6Pipeline,
)
from semantic_imagination.v6.clustering import cluster_joint_worlds
from semantic_imagination.v6.contracts import ordered_world


REGIONS = (
    RegionSpec("eyes", "eyewear", (10, 10, 30, 30)),
    RegionSpec("wrist", "wrist_accessory", (40, 40, 60, 60)),
)


def _positive(glasses: str = "thin glasses", watch: str = "metal watch") -> JointWorld:
    return JointWorld(
        (
            AtomicDetail("eyes", "eyewear", "eyewear_type", glasses, "face"),
            AtomicDetail("wrist", "wrist_accessory", "watch", watch, "wrist"),
        )
    )


def _absent() -> JointWorld:
    return JointWorld(
        (
            AtomicDetail("eyes", "eyewear", "absent", "none", "none"),
            AtomicDetail("wrist", "wrist_accessory", "absent", "none", "none"),
        )
    )


class FakeVLM:
    def __init__(self, outcomes, *, atomic_temperature: float = 0.7):
        self.outcomes = iter(outcomes)
        self.sample_calls = 0
        self.observe_calls = 0
        self.descriptor = BackendDescriptor(
            "fake-vlm",
            "fake-vision-model",
            "weights-sha256:abc",
            {
                "prompt_version": "qri-v6-observe-and-joint-v1",
                "atomic_temperature": atomic_temperature,
                "top_p": 0.9,
                "max_tokens": 512,
            },
        )

    def observe(self, source):
        self.observe_calls += 1
        return BackendResult(
            Observation(
                "Person in dark clothing.",
                tuple(RegionObservation(region.region_id) for region in source.regions),
            ),
            {"prompt_tokens": 10, "completion_tokens": 4},
        )

    def sample_joint_world(self, source, observation, seed):
        self.sample_calls += 1
        outcome = next(self.outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return BackendResult(outcome, {"prompt_tokens": 8, "completion_tokens": 3})


class FakeEncoder:
    descriptor = BackendDescriptor(
        "fake-encoder",
        "fake-semantic-encoder",
        "weights-sha256:def",
        {"normalization": "l2"},
    )

    def encode(self, texts):
        vectors = [
            (0.0, 1.0) if "state=absent" in text else (1.0, 0.0) for text in texts
        ]
        return BackendResult(vectors, {"encoded_texts": len(texts)})


class FakeRewriter:
    descriptor = BackendDescriptor(
        "fake-rewriter",
        "fake-text-model",
        "weights-sha256:ghi",
        {"prompt_version": "qri-v6-rewrite-v1", "temperature": 0.0},
    )

    def rewrite(self, request):
        details = ", ".join(item.value for item in request.world.assignments)
        return BackendResult(
            f"{request.observation.caption} Plausible unresolved details: {details}.",
            {"prompt_tokens": 6, "completion_tokens": 5},
        )


def _source(tmp_path):
    image = tmp_path / "person.jpg"
    image.write_bytes(b"qri-v6-image")
    return SourceSpec("cam1/0001/person.jpg", image, "rgb", REGIONS)


def _config(tmp_path, **changes):
    values = {
        "schema_version": 6,
        "plugin_version": "qri-v6",
        "output_root": tmp_path / "outputs",
        "joint_sample_count": 4,
        "max_worlds": 4,
        "similarity_threshold": 0.85,
        "request_max_attempts": 2,
        "seed": 17,
    }
    values.update(changes)
    return V6Config(**values).validate()


def test_v6_clusters_direct_joint_draws_without_cross_region_products(tmp_path):
    vlm = FakeVLM(
        [
            _positive(),
            _positive("rimmed spectacles", "dark timepiece"),
            _absent(),
            _absent(),
        ]
    )
    engine = V6Engine(
        _config(tmp_path), vlm=vlm, encoder=FakeEncoder(), rewriter=FakeRewriter()
    )
    record = engine.build_manifest(_source(tmp_path))

    assert record["plugin_version"] == "qri-v6"
    assert len(record["worlds"]) == 2
    assert [world["sample_count"] for world in record["worlds"]] == [2, 2]
    assert sum(world["selected_weight"] for world in record["worlds"]) == 1.0
    assert all(
        world["caption"].startswith("Person in dark clothing.")
        for world in record["worlds"]
    )
    combinations = {
        tuple(item["state"] for item in world["assignments"])
        for world in record["worlds"]
    }
    assert combinations == {("eyewear_type", "watch"), ("absent", "absent")}
    assert (
        record["sampling_contract"]["algorithm"]["world_sampling"]
        == "direct_joint_vlm_draws"
    )


def test_v6_similarity_threshold_actively_splits_same_state_values():
    worlds = [_positive("red glasses"), _positive("blue glasses")]
    clusters = cluster_joint_worlds(worlds, [(1.0, 0.0), (0.0, 1.0)], 0.9)
    assert len(clusters) == 2
    merged = cluster_joint_worlds(worlds, [(1.0, 0.0), (0.8, 0.2)], 0.9)
    assert len(merged) == 1


def test_v6_joint_world_rejects_duplicate_region_assignments():
    duplicate = JointWorld(
        (_positive().assignments[0],) * 2 + (_absent().assignments[1],)
    )
    with pytest.raises(ValueError, match="repeat"):
        ordered_world(duplicate, REGIONS)


def test_v6_retries_request_failures_and_counts_every_attempt(tmp_path):
    vlm = FakeVLM([TimeoutError("transient"), _positive(), _absent()])
    engine = V6Engine(
        _config(tmp_path, joint_sample_count=2, max_worlds=2),
        vlm=vlm,
        encoder=FakeEncoder(),
        rewriter=FakeRewriter(),
    )
    record = engine.build_manifest(_source(tmp_path))
    phase = record["telemetry"]["by_phase"]["vlm_joint_sample"]
    assert phase == {
        "operations": 2,
        "attempts": 3,
        "successes": 2,
        "retries": 1,
        "failures": 0,
        "elapsed_seconds": phase["elapsed_seconds"],
        "usage": {"completion_tokens": 6.0, "prompt_tokens": 16.0},
    }
    assert record["sampling_diagnostics"]["request_failed"] == 0


def test_v6_excludes_exhausted_requests_without_aborting_other_draws(tmp_path):
    vlm = FakeVLM([TimeoutError("one"), TimeoutError("two"), _positive(), _absent()])
    engine = V6Engine(
        _config(tmp_path, joint_sample_count=3, max_worlds=2),
        vlm=vlm,
        encoder=FakeEncoder(),
        rewriter=FakeRewriter(),
    )
    record = engine.build_manifest(_source(tmp_path))
    assert [sample["status"] for sample in record["samples"]] == [
        "request_failed",
        "valid",
        "valid",
    ]
    assert record["sampling_diagnostics"]["valid_mass"] == 2 / 3
    assert sum(world["empirical_mass"] for world in record["worlds"]) == 2 / 3


def test_v6_cache_signature_includes_backend_sampling_parameters(tmp_path):
    source = _source(tmp_path)
    first_vlm = FakeVLM([_positive(), _absent()])
    first = V6Pipeline(
        _config(tmp_path, joint_sample_count=2, max_worlds=2),
        vlm=first_vlm,
        encoder=FakeEncoder(),
        rewriter=FakeRewriter(),
    )
    assert first.run(source).cached is False
    assert first.run(source).cached is True
    assert first_vlm.sample_calls == 2

    changed_vlm = FakeVLM([_positive(), _absent()], atomic_temperature=0.9)
    changed = V6Pipeline(
        _config(tmp_path, joint_sample_count=2, max_worlds=2),
        vlm=changed_vlm,
        encoder=FakeEncoder(),
        rewriter=FakeRewriter(),
    )
    result = changed.run(source)
    assert result.cached is False
    assert changed_vlm.sample_calls == 2
    parameters = result.record["run_signature"]["payload"]["backends"]["vlm"][
        "parameters"
    ]
    assert parameters["atomic_temperature"] == 0.9
    assert parameters["prompt_version"] == "qri-v6-observe-and-joint-v1"


def test_v6_cache_signature_includes_region_geometry(tmp_path):
    source = _source(tmp_path)
    first = V6Pipeline(
        _config(tmp_path, joint_sample_count=2, max_worlds=2),
        vlm=FakeVLM([_positive(), _absent()]),
        encoder=FakeEncoder(),
        rewriter=FakeRewriter(),
    )
    assert first.run(source).cached is False

    changed_regions = (
        RegionSpec("eyes", "eyewear", (11, 10, 30, 30)),
        REGIONS[1],
    )
    changed_source = SourceSpec(
        source.source_key,
        source.image,
        source.modality,
        changed_regions,
    )
    changed = V6Pipeline(
        _config(tmp_path, joint_sample_count=2, max_worlds=2),
        vlm=FakeVLM([_positive(), _absent()]),
        encoder=FakeEncoder(),
        rewriter=FakeRewriter(),
    )
    result = changed.run(changed_source)
    assert result.cached is False
    assert result.record["sampling_contract"]["source"]["regions"][0]["bbox_xyxy"] == [
        11,
        10,
        30,
        30,
    ]


def test_v6_source_key_rejects_non_posix_absolute_paths(tmp_path):
    with pytest.raises(ValueError, match="POSIX"):
        SourceSpec("C:/outside/person.jpg", tmp_path / "person.jpg", "rgb", REGIONS)
