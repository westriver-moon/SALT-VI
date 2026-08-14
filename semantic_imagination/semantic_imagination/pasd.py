from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path


def to_pasd_record(
    manifest: Mapping[str, object],
    output_dir: str | Path,
    pasd_seed: int = 0,
    **source_fields: object,
) -> dict:
    source_key = str(manifest["source_key"])
    hypotheses = list(manifest["hypotheses"])
    if not hypotheses:
        raise ValueError("hypothesis manifest has no hypotheses")
    empirical_mass = sum(float(hypothesis["weight"]) for hypothesis in hypotheses)
    if empirical_mass <= 0:
        raise ValueError("hypothesis empirical mass must be positive")
    output_dir = Path(output_dir)
    views = []
    for view_index, hypothesis in enumerate(hypotheses):
        hypothesis_id = f"h{int(hypothesis['cluster_id']):02d}"
        digest = hashlib.sha256(
            f"{pasd_seed}:{source_key}:{hypothesis_id}".encode("utf-8")
        ).digest()
        view = {
            "view_index": view_index,
            "hypothesis_id": hypothesis_id,
            # PASD consumers require a probability simplex over generated views.
            # Preserve the pre-normalization model mass separately so validation
            # failures remain visible and are not turned into semantic views.
            "hypothesis_weight": float(hypothesis["weight"]) / empirical_mass,
            "hypothesis_empirical_mass": float(hypothesis["weight"]),
            "caption": str(hypothesis["caption"]),
            "seed": int.from_bytes(digest[:4], "big") & 0x7FFFFFFF,
            "output": str(output_dir / f"{hypothesis_id}.png").replace("\\", "/"),
        }
        if "weight_interval_95" in hypothesis:
            view.update(
                {
                    "hypothesis_weight_interval_95": dict(hypothesis["weight_interval_95"]),
                    "hypothesis_category": str(hypothesis["category"]),
                    "hypothesis_category_weight": float(hypothesis["category_weight"]),
                    "hypothesis_conditional_weight": float(hypothesis["conditional_weight"]),
                    "hypothesis_conditional_weight_interval_95": dict(
                        hypothesis["conditional_weight_interval_95"]
                    ),
                }
            )
        views.append(view)
    return {
        "image": str(manifest["image"]),
        "source_key": source_key,
        "imagination_contract_sha256": manifest["sampling_contract_sha256"],
        "imagination_sampling_diagnostics": dict(
            manifest.get("sampling_diagnostics", {})
        ),
        "imagination_valid_semantic_mass": empirical_mass,
        "imagination_unrepresented_category_mass": max(0.0, 1.0 - empirical_mass),
        "imagination_validation_failure_rate": (
            float(manifest.get("sampling_diagnostics", {}).get("validation_failed", 0))
            / max(1, int(manifest.get("sampling_diagnostics", {}).get("scheduled", 0)))
        ),
        **source_fields,
        "views": views,
    }
