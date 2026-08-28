import json

from PIL import Image
import pytest

from person_preprocessing import PersonAssetStore
from person_preprocessing import pipeline


def _fixture(tmp_path):
    native = tmp_path / "native"
    rows = []
    gate = []
    for index, decision in enumerate(("pass", "fallback"), 1):
        source = f"cam1/0001/{index:04d}.jpg"
        output = f"cam1/0001/{index:04d}.png"
        path = native / "sysu" / output
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (20, 50), "gray").save(path)
        rows.append({
            "dataset": "sysu", "source": source, "output": output,
            "modality": "rgb", "width": 20, "height": 50,
            "references": [], "aliases": [],
        })
        gate.append({
            "dataset": "sysu", "source_key": source,
            "native_image": str(path), "width": 20, "height": 50,
            "annotation_bbox": [2.0, 1.0, 18.0, 49.0],
            "machine_decision": "pass_pending_visual",
            "final_decision": decision, "final_reason": "fixture",
            "risk_reasons": [], "schp_risks": [],
        })
    manifest = tmp_path / "images.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    final_gate = tmp_path / "final_gate.jsonl"
    final_gate.write_text("".join(json.dumps(row) + "\n" for row in gate))
    config = {
        "input_root": str(native), "input_manifest": str(manifest),
        "quality_gate": str(final_gate), "output_root": str(tmp_path / "assets"),
        "size_hw": [64, 32], "person_margin": 0.05, "blur_radius": 2,
        "models": {},
        "quality_policy": {
            "partial_box_max_top_ratio": 0.05,
            "partial_box_min_bottom_ratio": 0.75,
        },
        "expected_counts": {"sysu": {"total": 2, "pass": 1, "fallback": 1}},
    }
    return config, gate


def test_materialize_current_gate_preserves_complete_inventory(tmp_path):
    config, _gate = _fixture(tmp_path)
    summary = pipeline.materialize(config, ["sysu"], workers=1)[0]
    store = PersonAssetStore(tmp_path / "assets", "sysu")
    assert summary == {
        "dataset": "sysu", "total": 2, "passed": 1, "fallback": 1,
        "fallback_rate": 0.5, "complete_inventory": True,
    }
    assert store.image("cam1/0001/0001.jpg").size == (32, 64)
    assert store.record("cam1/0001/0001.jpg")["localization"]["status"] == "detected_person"
    assert store.record("cam1/0001/0002.jpg")["localization"]["status"] == "fallback_full_frame"


def test_validate_gate_rejects_known_partial_person_failure(tmp_path):
    config, gate = _fixture(tmp_path)
    gate[0]["annotation_bbox"] = [2.0, 0.0, 18.0, 20.0]
    gate[0]["risk_reasons"] = ["missing_lower_body_evidence"]
    with open(config["quality_gate"], "w") as stream:
        for row in gate:
            stream.write(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="unsafe partial-person box"):
        pipeline.materialize(config, ["sysu"], workers=1)
