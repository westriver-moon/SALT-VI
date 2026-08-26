from __future__ import annotations

import hashlib
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from PIL import Image

from ..regional.schema import Region


_CANONICAL_ROOT = Path(__file__).resolve().parents[4] / "semantic_imagination"
if str(_CANONICAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_CANONICAL_ROOT))

from semantic_imagination import build_hypothesis_manifest  # noqa: E402
from semantic_imagination.validator import validate_atomic_response  # noqa: E402


def _atom_fields(text: str) -> dict[str, str]:
    return {
        part.split("=", 1)[0].strip().casefold(): part.split("=", 1)[1].strip()
        for part in str(text).split(";")
        if "=" in part
    }


def _hashed_embedding(texts: Sequence[str], dimensions: int = 128) -> list[list[float]]:
    vectors: list[list[float]] = []
    for text in texts:
        fields = _atom_fields(text)
        source = " ".join(
            (fields.get("value", ""), fields.get("location", ""))
        ).casefold()
        vector = [0.0] * dimensions
        for token in re.findall(r"[a-z0-9_]+", source):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % dimensions
            vector[index] += 1.0
        if not any(vector):
            vector[0] = 1.0
        norm = sum(value * value for value in vector) ** 0.5
        vectors.append([value / norm for value in vector])
    return vectors


class _AtomicBackend:
    """Legacy V5 adapter retained for historical text-annotation reproduction."""

    def __init__(
        self,
        reasoner: Any,
        swin: Image.Image,
        region: Region,
        modality: str,
        observed: str,
    ) -> None:
        self.reasoner = reasoner
        self.swin = swin
        self.region = region
        self.modality = modality
        self.observed = str(observed).strip()
        self.model_id = str(getattr(reasoner, "model_id", type(reasoner).__name__))

    def observe(self, image: Path) -> str:
        return self.observed

    def perturb(self, image: Path, seed: int) -> Image.Image:
        # V5 keeps one authoritative SwinIR image. Independent Qwen seeds provide
        # the decoding randomness; the identity perturbation is recorded in the
        # sampling contract instead of inventing additional pixel evidence.
        return self.swin

    def imagine(
        self,
        image: Image.Image,
        observed: str,
        instruction: str,
        seed: int,
    ) -> str:
        return self.reasoner.sample_atomic(
            image,
            self.region,
            modality=self.modality,
            observed=observed,
            instruction=instruction,
            seed=seed,
        )

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return _hashed_embedding(texts)


def _region_manifest(
    reasoner: Any,
    swin: Image.Image,
    region: Region,
    *,
    modality: str,
    source_key: str,
    observed: str,
    sample_count: int,
    seed: int,
    similarity_threshold: float,
    max_attempts: int,
) -> dict[str, Any]:
    backend = _AtomicBackend(reasoner, swin, region, modality, observed)
    return build_hypothesis_manifest(
        image=Path(source_key),
        source_key=f"{source_key}::{region.region_id}",
        backend=backend,
        instruction=(
            "Infer one atomic detail that remains genuinely ambiguous in the "
            "authoritative SwinIR observation. Use only the target category. "
            "Return exactly one ATOM record; never return a probability, confidence "
            "score, multiple details, or a sentence."
        ),
        sample_count=int(sample_count),
        seed=int(seed),
        similarity_threshold=float(similarity_threshold),
        compose=lambda _observed, hypothesis: hypothesis,
        contract={
            "pipeline": "legacy-qri-v5-swin-only",
            "perturbation": "identity-authoritative-swinir-v1",
            "sampling": "canonical-semantic-imagination-atomic-v1",
            "category": region.category,
            "prompt_version": str(
                getattr(backend.reasoner, "prompt_version", "legacy-unspecified")
            ),
            "roi_board_size_px": int(
                getattr(backend.reasoner, "roi_board_size_px", 512)
            ),
            "decoding": {
                "temperature": float(
                    getattr(backend.reasoner, "atomic_temperature", 0.75)
                ),
                "top_p": 0.9,
                "max_tokens": min(
                    220, int(getattr(backend.reasoner, "max_tokens", 2048))
                ),
                "thinking": False,
                "reasoning_effort": "none",
            },
        },
        cluster_linkage="complete",
        sampling_strata=(region.category,),
        validator=validate_atomic_response,
        max_attempts=int(max_attempts),
        validation_failure_policy="exclude",
    )


def _compact_region_result(manifest: dict[str, Any]) -> dict[str, Any]:
    scheduled = int(manifest["sampling_diagnostics"]["scheduled"])
    valid = int(manifest["sampling_diagnostics"]["valid"])
    clusters = []
    for hypothesis in manifest["hypotheses"]:
        cluster = {
            "cluster_id": int(hypothesis["cluster_id"]),
            "category": str(hypothesis["category"]),
            "state": hypothesis.get("state"),
            "representative": str(hypothesis["representative"]),
            "sample_count": int(hypothesis["count"]),
            "empirical_mass": int(hypothesis["count"]) / max(1, scheduled),
            "weight": float(hypothesis["weight"]),
            "weight_interval_95": dict(hypothesis["weight_interval_95"]),
            "member_indices": [int(index) for index in hypothesis["member_indices"]],
        }
        clusters.append(cluster)
    samples = []
    for sample in manifest["samples"]:
        item = {
            "seed": int(sample["seed"]),
            "status": str(sample["status"]),
        }
        if "stratum" in sample:
            item["stratum"] = str(sample["stratum"])
        if "text" in sample:
            item["text"] = str(sample["text"])
        if "cluster_id" in sample:
            item["cluster_id"] = int(sample["cluster_id"])
        samples.append(item)
    return {
        "source_key": str(manifest["source_key"]),
        "sampling_contract": dict(manifest["sampling_contract"]),
        "sampling_contract_sha256": str(manifest["sampling_contract_sha256"]),
        "sampling_diagnostics": dict(manifest["sampling_diagnostics"]),
        "scheduled_sample_count": scheduled,
        "valid_sample_count": valid,
        "valid_mass": valid / max(1, scheduled),
        "clusters": clusters,
        "samples": samples,
    }


def _sample_empirical_worlds(
    regions: list[dict[str, Any]],
    *,
    sample_count: int,
    max_worlds: int,
    seed: int,
) -> dict[str, Any]:
    if sample_count < 1:
        raise ValueError("world sample_count must be positive")
    if not 1 <= max_worlds <= sample_count:
        raise ValueError("max_worlds must be within [1, world sample_count]")
    rng = random.Random(int(seed))
    draws: list[tuple[int, ...]] = []
    for _ in range(int(sample_count)):
        draw = []
        for region in regions:
            clusters = list(region["clusters"])
            if not clusters:
                raise ValueError(
                    f"region {region['region_id']} has no valid empirical clusters"
                )
            draw.append(
                rng.choices(
                    range(len(clusters)),
                    weights=[float(item["weight"]) for item in clusters],
                    k=1,
                )[0]
            )
        draws.append(tuple(draw))
    counts = Counter(draws)
    retained = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[
        : int(max_worlds)
    ]
    retained_count = sum(count for _, count in retained)
    worlds = []
    for world_index, (indices, count) in enumerate(retained):
        assignments = []
        for region, cluster_index in zip(regions, indices):
            cluster = region["clusters"][cluster_index]
            assignments.append(
                {
                    "region_id": str(region["region_id"]),
                    "category": str(region["category"]),
                    "cluster_id": int(cluster["cluster_id"]),
                    "state": cluster.get("state"),
                    "description": str(cluster["representative"]),
                    "empirical_mass": float(cluster["empirical_mass"]),
                    "weight": float(cluster["weight"]),
                }
            )
        worlds.append(
            {
                "world_id": f"w{world_index:02d}",
                "assignments": assignments,
                "sample_count": int(count),
                "sample_frequency": count / float(sample_count),
                "selected_weight": count / float(max(1, retained_count)),
            }
        )
    return {
        "seed": int(seed),
        "sample_count": int(sample_count),
        "unique_world_count": len(counts),
        "retained_world_count": len(worlds),
        "retained_sample_mass": retained_count / float(sample_count),
        "worlds": worlds,
    }


def build_empirical_sampling(
    reasoner: Any,
    swin: Image.Image,
    regions: list[Region],
    *,
    modality: str,
    source_key: str,
    observed: str,
    atomic_sample_count: int,
    world_sample_count: int,
    max_worlds: int,
    seed: int,
    similarity_threshold: float,
    max_attempts: int,
) -> dict[str, Any]:
    if not regions:
        raise ValueError("empirical sampling requires at least one selected ROI")
    empirical_regions = []
    for index, region in enumerate(regions):
        manifest = _region_manifest(
            reasoner,
            swin,
            region,
            modality=modality,
            source_key=source_key,
            observed=observed,
            sample_count=atomic_sample_count,
            seed=(int(seed) + index * 1000003) % (2**31),
            similarity_threshold=similarity_threshold,
            max_attempts=max_attempts,
        )
        result = _compact_region_result(manifest)
        result["region_id"] = region.region_id
        result["category"] = region.category
        empirical_regions.append(result)

    return {
        "version": "empirical-atomic-v1",
        "weights_source": "repeated_atomic_samples_cluster_frequency",
        "specification": "docs/reference/semantic_imagination_mathematical_spec.md",
        "source_scope": "source_image",
        "atomic_sample_count": int(atomic_sample_count),
        "world_sample_count": int(world_sample_count),
        "regions": empirical_regions,
        "worlds": _sample_empirical_worlds(
            empirical_regions,
            sample_count=int(world_sample_count),
            max_worlds=int(max_worlds),
            seed=int(seed),
        ),
    }
