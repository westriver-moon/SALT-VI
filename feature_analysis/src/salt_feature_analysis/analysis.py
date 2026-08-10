from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .plotting import plot_comparison, plot_feature_diagnostics
from .reporting import write_csv, write_report
from .statistics import analyze_feature_artifact, compare_artifacts
from .storage import ArtifactLayout, load_feature_artifact, write_json


def analyze_all(config: Dict[str, Any]) -> Dict[str, Any]:
    layout = ArtifactLayout(config["output_root"], config["run_id"])
    catalog_path = layout.manifest_root / "catalog.json"
    if not catalog_path.is_file():
        raise FileNotFoundError(f"Feature catalog not found; run extract first: {catalog_path}")
    with catalog_path.open("r", encoding="utf-8") as handle:
        catalog = json.load(handle)
    entries = {item["artifact_key"]: item for item in catalog["artifacts"]}
    artifacts = {key: load_feature_artifact(Path(item["feature_path"])) for key, item in entries.items()}
    options = config["analysis"]
    feature_rows = []
    warnings = []
    for key, artifact in artifacts.items():
        summary, distributions = analyze_feature_artifact(
            artifact,
            seed=int(config["runtime"]["seed"]),
            max_pair_samples=int(options["max_pair_samples"]),
            max_svd_samples=int(options["max_svd_samples"]),
        )
        row = {"artifact_key": key, **summary}
        feature_rows.append(row)
        write_json(layout.table_root / "feature_sets" / f"{_slug(key)}.json", row)
        if bool(options["make_figures"]):
            warning = plot_feature_diagnostics(
                artifact.features,
                artifact.labels,
                distributions,
                layout.figure_root / "feature_sets" / _slug(key),
                int(options["max_plot_samples"]),
                int(config["runtime"]["seed"]),
            )
            if warning:
                warnings.append(warning)

    comparison_specs = list(config.get("comparisons") or [])
    if bool(options.get("auto_checkpoint_comparisons", True)):
        comparison_specs.extend(_automatic_comparisons(entries, comparison_specs))
    comparison_rows = []
    for item in comparison_specs:
        left_key, right_key = item["left"], item["right"]
        if left_key not in artifacts or right_key not in artifacts:
            raise KeyError(f"Comparison {item['id']} references missing artifact: {left_key}, {right_key}")
        summary, distributions = compare_artifacts(
            artifacts[left_key],
            artifacts[right_key],
            compute_retrieval=bool(item.get("compute_retrieval", False)),
            compute_sample_retrieval=bool(item.get("compute_sample_retrieval", True)),
            retrieval_chunk_size=int(item.get("retrieval_chunk_size", 1024)),
        )
        row = {
            "comparison_id": item["id"],
            "left": left_key,
            "right": right_key,
            **summary,
        }
        comparison_rows.append(row)
        write_json(layout.table_root / "comparisons" / f"{_slug(item['id'])}.json", row)
        if bool(options["make_figures"]):
            warning = plot_comparison(
                distributions,
                layout.figure_root / "comparisons" / f"{_slug(item['id'])}.png",
            )
            if warning:
                warnings.append(warning)

    write_csv(layout.table_root / "feature_summary.csv", feature_rows)
    write_csv(layout.table_root / "comparison_summary.csv", comparison_rows)
    write_report(
        layout.report_root / "report.md",
        config["run_id"],
        feature_rows,
        comparison_rows,
        warnings,
    )
    result = {
        "run_id": config["run_id"],
        "feature_set_count": len(feature_rows),
        "comparison_count": len(comparison_rows),
        "warnings": sorted(set(warnings)),
        "report": str((layout.report_root / "report.md").resolve()),
    }
    write_json(layout.manifest_root / "analysis_summary.json", result)
    return result


def _automatic_comparisons(entries: Dict[str, Dict[str, Any]], existing: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    used_pairs = {tuple(sorted((item["left"], item["right"]))) for item in existing}
    groups = defaultdict(list)
    for key, entry in entries.items():
        groups[(entry["split_tag"], entry["representation"]["name"])].append(key)
    generated = []
    for (split_tag, representation), keys in sorted(groups.items()):
        keys = sorted(keys)
        for left_index, left in enumerate(keys):
            for right in keys[left_index + 1 :]:
                pair = tuple(sorted((left, right)))
                if pair in used_pairs:
                    continue
                generated.append(
                    {
                        "id": _slug(f"auto_{split_tag}_{representation}_{left}_vs_{right}"),
                        "left": left,
                        "right": right,
                        "compute_retrieval": False,
                    }
                )
                used_pairs.add(pair)
    return generated


def _slug(value: str) -> str:
    return "_".join(part for part in value.replace("::", "_").replace("/", "_").split() if part)
