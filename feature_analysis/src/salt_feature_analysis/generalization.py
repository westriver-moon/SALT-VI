from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

from .reporting import write_csv


COMPARISON_PATTERN = re.compile(
    r"^(?P<checkpoint>.+)_(?P<split>train|test)_ir_to_(?P<target>rgb|fusion|text)$"
)
TARGET_ORDER = ("rgb", "fusion", "text")


def build_generalization_rows(comparison_rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for source in comparison_rows:
        match = COMPARISON_PATTERN.fullmatch(str(source.get("comparison_id", "")))
        if not match:
            continue
        rows.append(
            {
                "checkpoint": match.group("checkpoint"),
                "split": match.group("split"),
                "target": match.group("target"),
                "identity_count": _number(source.get("common_identity_count")),
                "centroid_cosine": _number(source.get("label_centroid_cosine_mean")),
                "centroid_top1": _number(source.get("label_centroid_retrieval_top1")),
                "centroid_top5": _number(source.get("label_centroid_retrieval_top5")),
                "centroid_mrr": _number(source.get("label_centroid_retrieval_mrr")),
                "sample_top1": _number(source.get("label_retrieval_top1")),
                "sample_top5": _number(source.get("label_retrieval_top5")),
                "sample_mrr": _number(source.get("label_retrieval_mrr")),
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            str(row["checkpoint"]),
            0 if row["split"] == "train" else 1,
            TARGET_ORDER.index(str(row["target"])),
        ),
    )


def render_generalization_summary(
    comparison_csv: Path,
    table_path: Path,
    figure_path: Path,
) -> Dict[str, str]:
    with comparison_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = build_generalization_rows(csv.DictReader(handle))
    if not rows:
        raise ValueError("No <checkpoint>_(train|test)_ir_to_(rgb|fusion|text) comparisons found")
    write_csv(table_path, rows)
    _plot(rows, figure_path)
    return {"table": str(table_path.resolve()), "figure": str(figure_path.resolve())}


def _number(value: Any) -> float:
    if value in (None, ""):
        return float("nan")
    return float(value)


def _plot(rows: List[Dict[str, Any]], destination: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lookup = {(row["checkpoint"], row["split"], row["target"]): row for row in rows}
    groups = sorted(
        {(str(row["checkpoint"]), str(row["split"])) for row in rows},
        key=lambda group: (group[0], 0 if group[1] == "train" else 1),
    )
    labels = [f"{checkpoint}\n{split_name}" for checkpoint, split_name in groups]
    x = np.arange(len(groups), dtype=np.float64)
    width = 0.24
    colors = {"rgb": "#2563eb", "fusion": "#7c3aed", "text": "#ea580c"}
    figure, axes = plt.subplots(1, 3, figsize=(17, 4.8))

    for target_index, target in enumerate(TARGET_ORDER):
        offset = (target_index - 1) * width
        centroid_top1 = [
            100.0 * lookup[(checkpoint, split_name, target)]["centroid_top1"]
            for checkpoint, split_name in groups
        ]
        centroid_cosine = [
            lookup[(checkpoint, split_name, target)]["centroid_cosine"]
            for checkpoint, split_name in groups
        ]
        axes[0].bar(x + offset, centroid_top1, width, label=target, color=colors[target])
        axes[1].bar(x + offset, centroid_cosine, width, label=target, color=colors[target])

    test_groups = [group for group in groups if group[1] == "test"]
    test_x = np.arange(len(test_groups), dtype=np.float64)
    for target_index, target in enumerate(TARGET_ORDER):
        offset = (target_index - 1) * width
        values = [
            100.0 * lookup[(checkpoint, split_name, target)]["sample_top1"]
            for checkpoint, split_name in test_groups
        ]
        axes[2].bar(test_x + offset, values, width, label=target, color=colors[target])

    axes[0].set_title("Identity-centroid retrieval")
    axes[0].set_ylabel("Top-1 (%)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylim(0, 105)
    axes[0].legend()

    axes[1].set_title("IR-to-target centroid alignment")
    axes[1].set_ylabel("Mean cosine")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].axhline(0.0, color="#111827", linewidth=0.8)

    axes[2].set_title("Held-out test sample retrieval")
    axes[2].set_ylabel("Top-1 (%)")
    axes[2].set_xticks(test_x)
    axes[2].set_xticklabels([checkpoint for checkpoint, _ in test_groups])
    axes[2].set_ylim(0, 75)

    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=200, bbox_inches="tight")
    plt.close(figure)
