from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("text_only_planner.py")
SPEC = importlib.util.spec_from_file_location("text_only_planner", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_probability_normalization_and_provenance_are_preserved():
    rows = MODULE.normalize_hypotheses(
        {
            "hypotheses": [
                {
                    "description": "known cartoon face on blue bag",
                    "probability": 3,
                    "basis": "world_knowledge",
                    "observable_support": "dark hair and round face",
                    "uncertainty": "character identity is blurred",
                },
                {
                    "description": "generic animal patch",
                    "probability": 1,
                    "basis": "unresolved",
                    "observable_support": "compact high-contrast patch",
                    "uncertainty": "facial parts are unresolved",
                },
            ]
        }
    )
    assert [round(row["probability"], 2) for row in rows] == [0.75, 0.25]
    assert rows[0]["basis"] == "world_knowledge"
    assert rows[1]["basis"] == "unresolved"


def test_probability_sampling_is_deterministic_and_does_not_add_candidates():
    rows = [
        {
            "description": "candidate a",
            "probability": 0.8,
            "basis": "visual_evidence",
            "observable_support": "edge",
            "uncertainty": "blur",
        },
        {
            "description": "candidate b",
            "probability": 0.2,
            "basis": "unresolved",
            "observable_support": "none",
            "uncertainty": "high",
        },
    ]
    first = MODULE.sample_text_worlds(rows, 32, 17)
    second = MODULE.sample_text_worlds(rows, 32, 17)
    assert first == second
    assert len(first["empirical_worlds"]) == len(rows)
    assert {item["description"] for item in first["empirical_worlds"]} == {
        "candidate a",
        "candidate b",
    }


def test_schema_rejects_unsupported_evidence_basis():
    try:
        MODULE.normalize_hypotheses(
            {
                "hypotheses": [
                    {"description": "a", "probability": 0.5, "basis": "template"},
                    {"description": "b", "probability": 0.5, "basis": "unresolved"},
                ]
            }
        )
    except ValueError as error:
        assert "unsupported basis" in str(error)
    else:
        raise AssertionError("unsupported basis must be rejected")
