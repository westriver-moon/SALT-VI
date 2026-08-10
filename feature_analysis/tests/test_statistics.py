import numpy as np

from salt_feature_analysis.statistics import _label_retrieval, analyze_feature_artifact, compare_artifacts
from salt_feature_analysis.storage import FeatureArtifact


def _artifact(features, labels, prefix="sample"):
    count = len(features)
    return FeatureArtifact(
        np.asarray(features, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
        np.full(count, -1, dtype=np.int64),
        np.asarray([f"{prefix}-{index}" for index in range(count)]),
        {},
    )


def test_separated_features_have_positive_cosine_margin():
    rng = np.random.RandomState(3)
    centers = np.eye(3, 6)
    labels = np.repeat(np.arange(3), 8)
    features = np.vstack([centers[label] + rng.normal(0, 0.02, 6) for label in labels])
    summary, _ = analyze_feature_artifact(_artifact(features, labels), 0, 200, 100)
    assert summary["cosine_separation"] > 0.8
    assert summary["nearest_centroid_accuracy"] == 1.0


def test_identical_artifacts_have_unit_cka_and_cosine():
    rng = np.random.RandomState(7)
    features = rng.normal(size=(20, 8))
    labels = np.repeat(np.arange(5), 4)
    artifact = _artifact(features, labels)
    summary, _ = compare_artifacts(artifact, artifact)
    assert np.isclose(summary["same_sample_cosine_mean"], 1.0)
    assert np.isclose(summary["linear_cka"], 1.0)
    assert summary["orthogonal_procrustes_residual"] < 1e-10


def test_rotation_is_detected_as_rigid_drift():
    rng = np.random.RandomState(11)
    features = rng.normal(size=(30, 6))
    q, _ = np.linalg.qr(rng.normal(size=(6, 6)))
    labels = np.repeat(np.arange(10), 3)
    left = _artifact(features, labels)
    right = _artifact(features @ q, labels)
    summary, _ = compare_artifacts(left, right)
    assert summary["linear_cka"] > 0.999999
    assert summary["orthogonal_procrustes_residual"] < 1e-6


def test_centroid_retrieval_is_reported_independently_of_sample_count():
    left = _artifact(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]],
        [0, 0, 1, 1],
        prefix="left",
    )
    right = _artifact([[1.0, 0.0], [0.0, 1.0]], [0, 1], prefix="right")
    summary, _ = compare_artifacts(left, right, compute_retrieval=True)
    assert summary["label_centroid_retrieval_top1"] == 1.0
    assert summary["label_centroid_retrieval_top5"] == 1.0
    assert summary["label_centroid_retrieval_mrr"] == 1.0


def test_label_retrieval_uses_best_positive_rank_without_full_sort():
    query = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    query_labels = np.asarray([0, 1])
    gallery = np.asarray([[0.9, 0.0], [0.8, 0.0], [0.1, 0.0]])
    gallery_labels = np.asarray([2, 0, 0])
    result = _label_retrieval(query, query_labels, gallery, gallery_labels, chunk_size=1)
    assert result["label_retrieval_top1"] == 0.0
    assert result["label_retrieval_top5"] == 0.5
    assert result["label_retrieval_mrr"] == 0.25


def test_sample_retrieval_can_be_skipped_while_centroids_are_retained():
    left = _artifact([[1.0, 0.0], [0.0, 1.0]], [0, 1], prefix="left")
    right = _artifact([[1.0, 0.0], [0.0, 1.0]], [0, 1], prefix="right")
    summary, _ = compare_artifacts(
        left,
        right,
        compute_retrieval=True,
        compute_sample_retrieval=False,
    )
    assert "label_retrieval_top1" not in summary
    assert summary["label_centroid_retrieval_top1"] == 1.0
