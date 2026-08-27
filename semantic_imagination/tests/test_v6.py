from __future__ import annotations

import json

import pytest

from semantic_imagination.v6 import (
    AtomicDetail,
    BackendDescriptor,
    BackendRequestError,
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
    def __init__(self, outcomes, *, generation_temperature: float = 0.7):
        self.outcomes = iter(outcomes)
        self.generation_calls = 0
        self.generation_requests = []
        self.observe_calls = 0
        self.descriptor = BackendDescriptor(
            "fake-vlm",
            "fake-vision-model",
            "weights-sha256:abc",
            {
                "prompt_version": "qri-v6-observe-and-candidates-v1",
                "generation_temperature": generation_temperature,
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

    def generate_joint_worlds(self, source, observation, count, seed):
        self.generation_calls += 1
        self.generation_requests.append((count, seed))
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
        "candidate_world_count": 4,
        "similarity_threshold": 0.85,
        "request_max_attempts": 2,
        "seed": 17,
    }
    values.update(changes)
    return V6Config(**values).validate()


def test_v6_generates_one_batch_and_weights_unique_worlds_equally(tmp_path):
    vlm = FakeVLM(
        [[
            _positive(),
            _positive("rimmed spectacles", "dark timepiece"),
            _positive(),
            _absent(),
        ]]
    )
    engine = V6Engine(
        _config(tmp_path), vlm=vlm, encoder=FakeEncoder(), rewriter=FakeRewriter()
    )
    record = engine.build_manifest(_source(tmp_path))

    assert record["plugin_version"] == "qri-v6"
    assert vlm.generation_calls == 1
    assert vlm.generation_requests[0][0] == 4
    assert len(record["worlds"]) == 2
    assert [len(world["member_candidate_indices"]) for world in record["worlds"]] == [3, 1]
    assert [world["selected_weight"] for world in record["worlds"]] == [0.5, 0.5]
    assert all(
        world["caption"].startswith("Person in dark clothing.")
        for world in record["worlds"]
    )
    combinations = {
        tuple(item["state"] for item in world["assignments"])
        for world in record["worlds"]
    }
    assert combinations == {("eyewear_type", "watch"), ("absent", "absent")}
    assert record["generation_diagnostics"] == {
        "requested_candidate_count": 4,
        "returned_candidate_count": 4,
        "valid_candidate_count": 4,
        "invalid_candidate_count": 0,
        "duplicate_candidate_count": 2,
        "world_count": 2,
    }
    for world in record["worlds"]:
        representative = record["candidates"][world["representative_candidate_index"]]
        assert world["assignments"] == representative["assignments"]
        for index in world["member_candidate_indices"]:
            assert record["candidates"][index]["world_id"] == world["world_id"]
        assert not {
            "sample_count", "empirical_mass", "valid_weight", "valid_weight_interval_95"
        }.intersection(world)
    assert not {"samples", "sampling_diagnostics", "sampling_contract"}.intersection(record)
    assert record["telemetry"]["by_phase"]["llm_rewrite"]["operations"] == 2


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


def test_v6_retries_one_generation_request_with_the_same_seed(tmp_path):
    vlm = FakeVLM([TimeoutError("transient"), [_positive(), _absent()]])
    engine = V6Engine(
        _config(tmp_path, candidate_world_count=2),
        vlm=vlm,
        encoder=FakeEncoder(),
        rewriter=FakeRewriter(),
    )
    record = engine.build_manifest(_source(tmp_path))
    phase = record["telemetry"]["by_phase"]["vlm_joint_generation"]
    assert phase == {
        "operations": 1,
        "attempts": 2,
        "successes": 1,
        "retries": 1,
        "failures": 0,
        "elapsed_seconds": phase["elapsed_seconds"],
        "usage": {"completion_tokens": 3.0, "prompt_tokens": 8.0},
    }
    assert vlm.generation_requests[0] == vlm.generation_requests[1]


def test_v6_exhausted_generation_does_not_write_a_complete_record(tmp_path):
    vlm = FakeVLM([TimeoutError("one"), TimeoutError("two")])
    pipeline = V6Pipeline(
        _config(tmp_path),
        vlm=vlm,
        encoder=FakeEncoder(),
        rewriter=FakeRewriter(),
    )
    source = _source(tmp_path)
    with pytest.raises(BackendRequestError, match="vlm_joint_generation failed"):
        pipeline.run(source)
    assert vlm.generation_calls == 2
    assert not pipeline.record_path(source).exists()


@pytest.mark.parametrize(
    ("candidates", "expected_weights", "invalid_count"),
    [
        ([_positive(), _absent()], [0.5, 0.5], 0),
        ([_positive(), _positive()], [1.0], 0),
        ([_positive(), JointWorld(())], [1.0], 1),
    ],
)
def test_v6_uses_actual_unique_count_without_filling_missing_candidates(
    tmp_path, candidates, expected_weights, invalid_count
):
    vlm = FakeVLM([candidates])
    record = V6Engine(
        _config(tmp_path),
        vlm=vlm,
        encoder=FakeEncoder(),
        rewriter=FakeRewriter(),
    ).build_manifest(_source(tmp_path))
    assert vlm.generation_calls == 1
    assert [world["selected_weight"] for world in record["worlds"]] == expected_weights
    assert record["generation_diagnostics"]["invalid_candidate_count"] == invalid_count
    for candidate in record["candidates"]:
        if candidate["status"] == "invalid":
            assert "world_id" not in candidate


@pytest.mark.parametrize("candidates", [[], [_positive()] * 5, [JointWorld(())]])
def test_v6_rejects_unusable_batches_without_silent_top_k(tmp_path, candidates):
    vlm = FakeVLM([candidates])
    pipeline = V6Pipeline(
        _config(tmp_path, request_max_attempts=1),
        vlm=vlm,
        encoder=FakeEncoder(),
        rewriter=FakeRewriter(),
    )
    source = _source(tmp_path)
    with pytest.raises(BackendRequestError):
        pipeline.run(source)
    assert not pipeline.record_path(source).exists()


def test_v6_keeps_every_distinct_world_without_frequency_ranking(tmp_path):
    class DistinctEncoder(FakeEncoder):
        def encode(self, texts):
            return BackendResult([
                [float(i == j) for j in range(len(texts))]
                for i in range(len(texts))
            ])

    candidates = [_positive("red glasses"), _positive("blue glasses"), _absent()]
    record = V6Engine(
        _config(tmp_path),
        vlm=FakeVLM([candidates]),
        encoder=DistinctEncoder(),
        rewriter=FakeRewriter(),
    ).build_manifest(_source(tmp_path))
    assert [world["assignments"] for world in record["worlds"]] == [
        world.manifest() for world in candidates
    ]
    assert [world["selected_weight"] for world in record["worlds"]] == [1 / 3] * 3


def test_v6_cache_signature_includes_backend_generation_parameters(tmp_path):
    source = _source(tmp_path)
    first_vlm = FakeVLM([[_positive(), _absent()]])
    first = V6Pipeline(
        _config(tmp_path, candidate_world_count=2),
        vlm=first_vlm,
        encoder=FakeEncoder(),
        rewriter=FakeRewriter(),
    )
    assert first.run(source).cached is False
    assert first.run(source).cached is True
    assert first_vlm.generation_calls == 1

    changed_vlm = FakeVLM([[_positive(), _absent()]], generation_temperature=0.9)
    changed = V6Pipeline(
        _config(tmp_path, candidate_world_count=2),
        vlm=changed_vlm,
        encoder=FakeEncoder(),
        rewriter=FakeRewriter(),
    )
    result = changed.run(source)
    assert result.cached is False
    assert changed_vlm.generation_calls == 1
    parameters = result.record["run_signature"]["payload"]["backends"]["vlm"][
        "parameters"
    ]
    assert parameters["generation_temperature"] == 0.9
    assert parameters["prompt_version"] == "qri-v6-observe-and-candidates-v1"


def test_v6_cache_signature_includes_region_geometry(tmp_path):
    source = _source(tmp_path)
    first = V6Pipeline(
        _config(tmp_path, candidate_world_count=2),
        vlm=FakeVLM([[_positive(), _absent()]]),
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
        _config(tmp_path, candidate_world_count=2),
        vlm=FakeVLM([[_positive(), _absent()]]),
        encoder=FakeEncoder(),
        rewriter=FakeRewriter(),
    )
    result = changed.run(changed_source)
    assert result.cached is False
    assert result.record["generation_contract"]["source"]["regions"][0]["bbox_xyxy"] == [
        11,
        10,
        30,
        30,
    ]


def test_v6_source_key_rejects_non_posix_absolute_paths(tmp_path):
    with pytest.raises(ValueError, match="POSIX"):
        SourceSpec("C:/outside/person.jpg", tmp_path / "person.jpg", "rgb", REGIONS)


def test_v6_invalidates_frequency_based_v6_cache(tmp_path):
    source = _source(tmp_path)
    vlm = FakeVLM([[_positive(), _absent()], [_positive(), _absent()]])
    pipeline = V6Pipeline(
        _config(tmp_path),
        vlm=vlm,
        encoder=FakeEncoder(),
        rewriter=FakeRewriter(),
    )
    initial = pipeline.run(source)
    legacy = initial.record
    legacy["sampling_contract"] = legacy.pop("generation_contract")
    legacy["sampling_contract_sha256"] = legacy.pop("generation_contract_sha256")
    legacy["run_signature"]["payload"]["algorithm"] = {
        "joint_sample_count": 32,
        "max_worlds": 8,
        "world_sampling": "direct_joint_vlm_draws",
        "weight_estimator": "joint_cluster_frequency",
    }
    initial.path.write_text(json.dumps(legacy), encoding="utf-8")
    refreshed = pipeline.run(source)
    assert refreshed.cached is False
    assert vlm.generation_calls == 2
    assert "sampling_contract" not in refreshed.record
    assert [w["selected_weight"] for w in refreshed.record["worlds"]] == [0.5, 0.5]


def test_v6_candidate_count_changes_invalidate_cache(tmp_path):
    source = _source(tmp_path)
    first = V6Pipeline(
        _config(tmp_path, candidate_world_count=1),
        vlm=FakeVLM([[_positive()]]),
        encoder=FakeEncoder(),
        rewriter=FakeRewriter(),
    )
    assert first.run(source).cached is False
    vlm = FakeVLM([[_positive(), _absent()]])
    changed = V6Pipeline(
        _config(tmp_path, candidate_world_count=2),
        vlm=vlm,
        encoder=FakeEncoder(),
        rewriter=FakeRewriter(),
    )
    result = changed.run(source)
    assert result.cached is False
    assert vlm.generation_requests[0][0] == 2
    assert len(result.record["worlds"]) == 2
