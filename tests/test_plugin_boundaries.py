from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _python_text(root):
    return "\n".join(path.read_text(encoding="utf-8") for path in root.rglob("*.py"))


def test_shared_preprocessing_has_no_consumer_dependency():
    text = _python_text(ROOT / "person_preprocessing/person_preprocessing")
    assert "salt_vi" not in text
    assert "qwen_imagination" not in text
    assert "from pact" not in text


def test_pact_never_runs_pose_inference():
    text = _python_text(ROOT / "plugins/pact/pact")
    assert "ultralytics" not in text
    assert "PoseEstimator" not in text


def test_prepared_qwen_never_runs_pose_inference():
    text = (ROOT / "plugins/qwen_imagination/qwen_imagination/text_annotation/prepared.py").read_text(
        encoding="utf-8"
    )
    assert "ultralytics" not in text
    assert "PoseEstimator" not in text
    assert "PoseModels" not in text
