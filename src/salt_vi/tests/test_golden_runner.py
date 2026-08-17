from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import yaml


SCRIPT = Path(__file__).resolve().parents[3] / "scripts/evaluation/run_golden_evaluation.py"


def _runner():
    spec = importlib.util.spec_from_file_location("run_golden_evaluation", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_only_runs_preserve_complete_manifest_index(tmp_path, monkeypatch):
    runner = _runner()
    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    output = tmp_path / "reports/golden"
    evaluations = []
    for entry_id in ("a", "b", "c"):
        config = tmp_path / f"{entry_id}.yaml"
        checkpoint = tmp_path / f"{entry_id}.pth"
        config.write_text("dataset: sysu\n", encoding="utf-8")
        checkpoint.write_bytes(b"checkpoint")
        evaluations.append(
            {
                "id": entry_id,
                "config_path": str(config),
                "checkpoint_path": str(checkpoint),
            }
        )
        golden = output / entry_id / "golden_evaluation.json"
        golden.parent.mkdir(parents=True)
        golden.write_text("{}\n", encoding="utf-8")
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(yaml.safe_dump({"evaluations": evaluations}), encoding="utf-8")

    assert runner.main(["--manifest", str(manifest), "--output-root", str(output), "--only", "a"]) == 0
    first = (output / "index.json").read_bytes()
    assert runner.main(["--manifest", str(manifest), "--output-root", str(output), "--only", "b"]) == 0
    assert (output / "index.json").read_bytes() == first
    index = json.loads(first)
    assert [item["id"] for item in index["results"]] == ["a", "b", "c"]
    assert {item["status"] for item in index["results"]} == {"completed"}
    assert all(not Path(item["golden_evaluation_path"]).is_absolute() for item in index["results"])
