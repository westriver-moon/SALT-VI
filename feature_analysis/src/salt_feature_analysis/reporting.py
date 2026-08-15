from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List



def _scalar(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _scalar(row.get(key)) for key in fields})
    temporary.replace(path)


def write_report(
    path: Path,
    run_id: str,
    feature_rows: List[Dict[str, Any]],
    comparison_rows: List[Dict[str, Any]],
    warnings: List[str],
) -> None:
    lines = [f"# Feature analysis report: {run_id}", ""]
    lines.extend(["## Feature sets", ""])
    lines.append("| Artifact | N | Dim | IDs | Norm mean | Within cos | Between cos | Separation | Effective rank |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in feature_rows:
        lines.append(
            "| {artifact_key} | {sample_count} | {feature_dim} | {identity_count} | {norm_mean:.4f} | "
            "{within_identity_cosine_mean:.4f} | {between_identity_cosine_mean:.4f} | "
            "{cosine_separation:.4f} | {effective_rank:.2f} |".format(**row)
        )
    if comparison_rows:
        lines.extend(["", "## Comparisons", ""])
        lines.append("| Comparison | Common samples | Same-sample cos | CKA | Procrustes residual | Centroid cos |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for row in comparison_rows:
            lines.append(
                "| {comparison_id} | {common_sample_count} | {same_sample_cosine_mean:.4f} | "
                "{linear_cka:.4f} | {orthogonal_procrustes_residual:.4f} | "
                "{label_centroid_cosine_mean:.4f} |".format(**row)
            )
    if warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend([f"- {warning}" for warning in sorted(set(warnings))])
    lines.extend(
        [
            "",
            "## Interpretation notes",
            "",
            "- Larger within-minus-between cosine separation indicates clearer identity geometry.",
            "- Effective rank measures how many feature directions are materially used.",
            "- Same-sample cosine and linear CKA quantify checkpoint drift on identical samples.",
            "- Procrustes residual separates a global rotation from non-rigid representation change.",
            "- Label-centroid cosine measures cross-modality or cross-protocol identity alignment.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")

