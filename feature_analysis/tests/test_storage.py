from pathlib import Path

import numpy as np

from salt_feature_analysis.storage import FeatureArtifact, load_feature_artifact, save_feature_artifact


def test_feature_artifact_roundtrip(tmp_path: Path):
    artifact = FeatureArtifact(
        features=np.arange(12, dtype=np.float32).reshape(3, 4),
        labels=np.asarray([1, 2, 3]),
        cameras=np.asarray([3, 3, 6]),
        sample_ids=np.asarray(["a", "b", "c"]),
        metadata={"checkpoint_sha256": "abc"},
    )
    path = tmp_path / "features.npz"
    save_feature_artifact(path, artifact)
    restored = load_feature_artifact(path)
    np.testing.assert_array_equal(restored.features, artifact.features)
    np.testing.assert_array_equal(restored.labels, artifact.labels)
    assert restored.metadata["checkpoint_sha256"] == "abc"

