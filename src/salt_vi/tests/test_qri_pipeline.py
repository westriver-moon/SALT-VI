import importlib.util
import json
from pathlib import Path

import pytest

from salt_vi.data.sysu_sources import collect_test_source_records


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = PROJECT_ROOT / "scripts/experiments/run_qri_v1_pipeline.py"
SPEC = importlib.util.spec_from_file_location("run_qri_v1_pipeline", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_qri_formal_launch_requires_c3_b96_final_epoch(tmp_path: Path):
    events = tmp_path / "c3_b96.jsonl"
    events.write_text(
        json.dumps({"event": "train_epoch", "epoch": 22}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="remains gated"):
        MODULE.validate_base_completion(events, 23)

    with events.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"event": "train_epoch", "epoch": 23}) + "\n")
    result = MODULE.validate_base_completion(events, 23)
    assert result["final_epoch"] == 23


def test_qri_manifest_gate_requires_complete_dynamic_summary(tmp_path: Path):
    root = tmp_path / "qri"
    manifest = root / "manifests" / "manifest.posterior.jsonl"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}\n", encoding="utf-8")
    manifest.with_suffix(".json").write_text(
        json.dumps({"complete": True, "views_per_source": 0}), encoding="utf-8"
    )
    result = MODULE.validate_manifest(root, "manifests/manifest.posterior.jsonl")
    assert result["summary"]["views_per_source"] == 0

    manifest.with_suffix(".json").write_text(
        json.dumps({"complete": True, "views_per_source": 5}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="dynamic complete contract"):
        MODULE.validate_manifest(root, "manifests/manifest.posterior.jsonl")


def test_qri_formal_sources_cover_all_possible_sysu_gallery_images(tmp_path: Path):
    root = tmp_path / "SYSU-MM01"
    (root / "exp").mkdir(parents=True)
    (root / "exp" / "test_id.txt").write_text("1\n", encoding="utf-8")
    expected = {
        "rgb": ("cam1/0001/a.jpg", "cam1/0001/b.jpg", "cam4/0001/c.jpg"),
        "ir": ("cam3/0001/d.jpg",),
    }
    for paths in expected.values():
        for relative in paths:
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"test")
    for modality, paths in expected.items():
        records = collect_test_source_records(root, modality)
        assert tuple(record.source_key for record in records) == paths
