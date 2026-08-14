import math
from pathlib import Path

import pytest

from semantic_imagination import (
    CATEGORY_STATES,
    build_hypothesis_manifest,
    cluster_hypothesis_samples,
    to_pasd_record,
    validate_atomic_response,
)


class FakeBackend:
    model_id = "fake-vlm"

    def __init__(self):
        self.outputs = iter(("black backpack", "dark backpack", "baseball cap"))

    def observe(self, image):
        return "red upper garment"

    def perturb(self, image, seed):
        return image, seed

    def imagine(self, image, observed, instruction, seed):
        return next(self.outputs)

    def embed(self, texts):
        return ((1.0, 0.0), (0.99, 0.01), (0.0, 1.0))


def test_builds_cluster_mass_and_exports_weighted_pasd_views(tmp_path: Path):
    image = tmp_path / "person.jpg"
    image.write_bytes(b"image")
    manifest = build_hypothesis_manifest(
        image,
        "cam1/0001/person.jpg",
        FakeBackend(),
        "imagine only unobserved details",
        sample_count=3,
        seed=7,
        similarity_threshold=0.95,
    )

    assert [item["count"] for item in manifest["hypotheses"]] == [2, 1]
    assert [item["weight"] for item in manifest["hypotheses"]] == pytest.approx(
        [2 / 3, 1 / 3]
    )
    assert manifest["hypotheses"][0]["representative"] == "black backpack"
    assert sum(item["weight"] for item in manifest["hypotheses"]) == pytest.approx(1.0)
    assert manifest["cluster_linkage"] == "complete"
    interval = manifest["hypotheses"][0]["weight_interval_95"]
    assert interval["lower"] < 2 / 3 < interval["upper"]

    record = to_pasd_record(manifest, "images/cam1/0001/person", modality="rgb")
    assert [view["hypothesis_weight"] for view in record["views"]] == pytest.approx(
        [2 / 3, 1 / 3]
    )
    assert record["views"][0]["caption"] == "red upper garment black backpack"
    assert record["views"][0]["hypothesis_weight_interval_95"] == interval
    assert record["imagination_contract_sha256"] == manifest["sampling_contract_sha256"]


def test_complete_link_prevents_bridge_sentence_chain():
    radians = [math.radians(angle) for angle in (0, 20, 40)]
    vectors = [(math.cos(value), math.sin(value)) for value in radians]
    texts = ["watch", "watch and glasses", "glasses"]

    corrected = cluster_hypothesis_samples(texts, vectors, 0.9)
    legacy = cluster_hypothesis_samples(texts, vectors, 0.9, "single")

    # A~B and B~C exceed 0.9, while A~C is only cos(40°)=0.766.
    assert sorted(item["count"] for item in corrected) == [1, 2]
    assert [item["count"] for item in legacy] == [3]
    for cluster in corrected:
        members = cluster["member_indices"]
        for left in members:
            for right in members:
                dot = sum(a * b for a, b in zip(vectors[left], vectors[right]))
                assert dot >= 0.9


def test_rejects_unknown_cluster_linkage():
    with pytest.raises(ValueError, match="cluster_linkage"):
        cluster_hypothesis_samples(["watch"], [(1.0, 0.0)], 0.9, "average")


def test_atomic_categories_never_merge():
    texts = [
        "category=eyewear; value=dark frame; location=face",
        "category=wrist_accessory; value=dark strap; location=wrist",
    ]
    clusters = cluster_hypothesis_samples(texts, [(1.0, 0.0), (1.0, 0.0)], 0.9)
    assert [item["count"] for item in clusters] == [1, 1]
    assert [item["conditional_weight"] for item in clusters] == [1.0, 1.0]
    assert [item["category_weight"] for item in clusters] == [0.5, 0.5]


def test_atomic_absence_and_abstention_are_hard_states():
    texts = [
        "category=headwear; value=cap; location=head",
        "category=headwear; value=absent; location=head",
        "category=headwear; value=no_additional_detail; location=none",
    ]
    clusters = cluster_hypothesis_samples(
        texts, [(1.0, 0.0), (1.0, 0.0), (1.0, 0.0)], 0.9
    )
    assert [item["count"] for item in clusters] == [1, 1, 1]
    assert all(item["category_weight"] == 1.0 for item in clusters)
    assert all(item["conditional_weight"] == pytest.approx(1 / 3) for item in clusters)


def test_controlled_state_defines_semantic_equivalence_class():
    texts = [
        "category=footwear_detail; state=strap; value=strap; location=foot",
        "category=footwear_detail; state=strap; value=straps; location=feet",
        "category=footwear_detail; state=laces; value=laces; location=feet",
    ]
    # Same controlled state merges even when its free-text vectors are dissimilar;
    # different controlled states never merge even when their vectors are identical.
    clusters = cluster_hypothesis_samples(
        texts, [(1.0, 0.0), (0.0, 1.0), (1.0, 0.0)], 0.99
    )
    assert [item["count"] for item in clusters] == [2, 1]
    assert [item["state"] for item in clusters] == ["strap", "laces"]


def test_sampling_strata_are_balanced_and_hashed(tmp_path: Path):
    class StratifiedBackend(FakeBackend):
        def __init__(self):
            self.instructions = []

        def imagine(self, image, observed, instruction, seed):
            self.instructions.append(instruction)
            category = instruction.split("Target category: ", 1)[1].split(".", 1)[0]
            return f"category={category}; value=absent; location=none"

        def embed(self, texts):
            return [(1.0, 0.0) for _ in texts]

    image = tmp_path / "person.jpg"
    image.write_bytes(b"image")
    backend = StratifiedBackend()
    manifest = build_hypothesis_manifest(
        image,
        "person.jpg",
        backend,
        "one atom",
        5,
        3,
        0.9,
        sampling_strata=("headwear", "body_marking"),
    )

    assert [sample["stratum"] for sample in manifest["samples"]] == [
        "headwear",
        "body_marking",
        "headwear",
        "body_marking",
        "headwear",
    ]
    assert manifest["sampling_contract"]["sampling_strata"] == [
        "headwear",
        "body_marking",
    ]
    assert all("Target category:" in value for value in backend.instructions)


def test_rejects_empty_imagination_sample(tmp_path: Path):
    backend = FakeBackend()
    backend.outputs = iter(("",))
    image = tmp_path / "person.jpg"
    image.write_bytes(b"image")
    with pytest.raises(ValueError, match="non-empty"):
        build_hypothesis_manifest(image, "person.jpg", backend, "q", 1, 0, 0.9)


def test_validator_rejects_state_value_semantic_mismatch():
    result = validate_atomic_response(
        "ATOM | clothing_detail | graphic | red | t-shirt",
        "clothing_detail",
    )
    assert not result.valid
    assert [issue.code for issue in result.issues] == ["state_value_mismatch"]

    corrected = validate_atomic_response(
        "ATOM | clothing_detail | color_detail | red | t-shirt",
        "clothing_detail",
    )
    assert corrected.valid
    assert corrected.hypothesis.state == "color_detail"


def test_validation_failure_is_excluded_not_converted_to_abstention(tmp_path: Path):
    class ValidatingBackend(FakeBackend):
        def __init__(self):
            self.outputs = iter(
                (
                    "not an atom",
                    "still invalid",
                    "ATOM | headwear | absent | absent | head",
                    "ATOM | eyewear | absent | absent | face",
                )
            )

        def embed(self, texts):
            assert texts == ["category=eyewear; state=absent; value=absent; location=face"]
            return [(1.0, 0.0)]

    image = tmp_path / "person.jpg"
    image.write_bytes(b"image")
    manifest = build_hypothesis_manifest(
        image,
        "person.jpg",
        ValidatingBackend(),
        "one atom",
        sample_count=2,
        seed=3,
        similarity_threshold=0.9,
        sampling_strata=("headwear", "eyewear"),
        validator=validate_atomic_response,
        max_attempts=2,
        validation_failure_policy="exclude",
    )

    assert [sample["status"] for sample in manifest["samples"]] == [
        "validation_failed",
        "valid",
    ]
    assert "text" not in manifest["samples"][0]
    assert manifest["sampling_diagnostics"]["validation_failed"] == 1
    assert manifest["sampling_diagnostics"]["active_abstentions"] == 0
    assert manifest["hypotheses"][0]["category"] == "eyewear"
    assert manifest["hypotheses"][0]["conditional_weight"] == 1.0
    # Balanced scheduled prior remains explicit; failed headwear mass is not silently
    # reassigned to the surviving eyewear state.
    assert manifest["hypotheses"][0]["weight"] == 0.5
    record = to_pasd_record(manifest, tmp_path / "pasd")
    assert sum(view["hypothesis_weight"] for view in record["views"]) == 1.0
    assert record["views"][0]["hypothesis_empirical_mass"] == 0.5
    assert record["imagination_unrepresented_category_mass"] == 0.5
    assert record["imagination_validation_failure_rate"] == 0.5


def test_active_abstention_remains_a_valid_semantic_sample(tmp_path: Path):
    class AbstainingBackend(FakeBackend):
        def __init__(self):
            self.outputs = iter(
                ("ATOM | headwear | no_additional_detail | no_additional_detail | none",)
            )

        def embed(self, texts):
            return [(1.0, 0.0)]

    image = tmp_path / "person.jpg"
    image.write_bytes(b"image")
    manifest = build_hypothesis_manifest(
        image,
        "person.jpg",
        AbstainingBackend(),
        "one atom",
        1,
        3,
        0.9,
        sampling_strata=("headwear",),
        validator=validate_atomic_response,
        max_attempts=2,
        validation_failure_policy="exclude",
    )
    assert manifest["samples"][0]["status"] == "valid"
    assert manifest["sampling_diagnostics"]["active_abstentions"] == 1
    assert manifest["sampling_diagnostics"]["validation_failed"] == 0
    assert manifest["hypotheses"][0]["weight"] == 1.0


def test_active_abstention_is_canonicalized():
    result = validate_atomic_response(
        "ATOM | pocket_item | no_additional_detail | no additional detail | side",
        "pocket_item",
    )
    assert result.valid
    assert result.hypothesis.value == "no_additional_detail"
    assert result.hypothesis.location == "none"


def test_category_taxonomy_contains_all_sampling_categories():
    assert len(CATEGORY_STATES) == 8
    assert all("no_additional_detail" in states for states in CATEGORY_STATES.values())
