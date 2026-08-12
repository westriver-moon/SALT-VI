from pathlib import Path

import pytest

from semantic_imagination import build_hypothesis_manifest, to_pasd_record


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

    record = to_pasd_record(manifest, "images/cam1/0001/person", modality="rgb")
    assert [view["hypothesis_weight"] for view in record["views"]] == pytest.approx(
        [2 / 3, 1 / 3]
    )
    assert record["views"][0]["caption"] == "red upper garment black backpack"
    assert record["imagination_contract_sha256"] == manifest["sampling_contract_sha256"]


def test_rejects_empty_imagination_sample(tmp_path: Path):
    backend = FakeBackend()
    backend.outputs = iter(("",))
    image = tmp_path / "person.jpg"
    image.write_bytes(b"image")
    with pytest.raises(ValueError, match="non-empty"):
        build_hypothesis_manifest(image, "person.jpg", backend, "q", 1, 0, 0.9)
