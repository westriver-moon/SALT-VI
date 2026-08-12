from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Callable, Mapping, Protocol, Sequence


class ImaginationBackend(Protocol):
    """Model-specific operations; the plugin owns their statistical aggregation."""

    model_id: str

    def observe(self, image: Path) -> str: ...

    def perturb(self, image: Path, seed: int) -> object: ...

    def imagine(
        self, image: object, observed: str, instruction: str, seed: int
    ) -> str: ...

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        raise ValueError("semantic embeddings must be non-zero")
    return dot / (left_norm * right_norm)


def _components(vectors: Sequence[Sequence[float]], threshold: float) -> list[list[int]]:
    parent = list(range(len(vectors)))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for left in range(len(vectors)):
        for right in range(left + 1, len(vectors)):
            if _cosine(vectors[left], vectors[right]) >= threshold:
                a, b = root(left), root(right)
                parent[max(a, b)] = min(a, b)
    groups: dict[int, list[int]] = {}
    for index in range(len(vectors)):
        groups.setdefault(root(index), []).append(index)
    return sorted(groups.values(), key=lambda group: group[0])


def _medoid(indices: Sequence[int], vectors: Sequence[Sequence[float]]) -> int:
    return min(
        indices,
        key=lambda index: (
            sum(1.0 - _cosine(vectors[index], vectors[other]) for other in indices),
            index,
        ),
    )


def _default_compose(observed: str, hypothesis: str) -> str:
    return " ".join(part.strip() for part in (observed, hypothesis) if part.strip())


def build_hypothesis_manifest(
    image: str | Path,
    source_key: str,
    backend: ImaginationBackend,
    instruction: str,
    sample_count: int,
    seed: int,
    similarity_threshold: float,
    compose: Callable[[str, str], str] = _default_compose,
    contract: Mapping[str, object] | None = None,
) -> dict:
    """Sample, cluster, and weight semantic hypotheses for one source image."""

    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    if not -1.0 <= similarity_threshold <= 1.0:
        raise ValueError("similarity_threshold must be in [-1, 1]")
    image = Path(image).expanduser().resolve()
    observed = str(backend.observe(image)).strip()
    rng = random.Random(seed)
    samples = []
    for _ in range(sample_count):
        sample_seed = rng.randrange(2**31)
        perturbed = backend.perturb(image, sample_seed)
        text = str(
            backend.imagine(perturbed, observed, instruction, sample_seed)
        ).strip()
        if not text:
            raise ValueError("imagination samples must be non-empty")
        samples.append({"seed": sample_seed, "text": text})

    vectors = [list(map(float, vector)) for vector in backend.embed([s["text"] for s in samples])]
    if len(vectors) != sample_count or not vectors or not vectors[0]:
        raise ValueError("backend returned an invalid embedding matrix")
    dimension = len(vectors[0])
    if any(
        len(vector) != dimension or any(not math.isfinite(value) for value in vector)
        for vector in vectors
    ):
        raise ValueError("semantic embeddings must be finite and rectangular")

    hypotheses = []
    for cluster_id, members in enumerate(_components(vectors, similarity_threshold)):
        representative = _medoid(members, vectors)
        weight = len(members) / sample_count
        hypotheses.append(
            {
                "cluster_id": cluster_id,
                "representative": samples[representative]["text"],
                "member_indices": members,
                "count": len(members),
                "weight": weight,
                "caption": compose(observed, samples[representative]["text"]),
            }
        )
        for member in members:
            samples[member]["cluster_id"] = cluster_id

    sampling_contract = {
        "schema_version": 1,
        "backend_id": str(getattr(backend, "model_id", type(backend).__name__)),
        "instruction": instruction,
        "sample_count": sample_count,
        "seed": int(seed),
        "similarity_threshold": float(similarity_threshold),
        "backend_contract": dict(contract or {}),
    }
    encoded = json.dumps(
        sampling_contract, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "schema_version": 1,
        "source_key": source_key,
        "image": str(image),
        "observed": observed,
        "sampling_contract": sampling_contract,
        "sampling_contract_sha256": hashlib.sha256(encoded).hexdigest(),
        "samples": samples,
        "hypotheses": hypotheses,
    }


def to_pasd_record(
    manifest: Mapping[str, object],
    output_dir: str | Path,
    pasd_seed: int = 0,
    **source_fields: object,
) -> dict:
    """Convert one hypothesis manifest to a dynamic-view PASD source record."""

    source_key = str(manifest["source_key"])
    hypotheses = list(manifest["hypotheses"])
    if not hypotheses:
        raise ValueError("hypothesis manifest has no hypotheses")
    output_dir = Path(output_dir)
    views = []
    for view_index, hypothesis in enumerate(hypotheses):
        hypothesis_id = f"h{int(hypothesis['cluster_id']):02d}"
        digest = hashlib.sha256(
            f"{pasd_seed}:{source_key}:{hypothesis_id}".encode("utf-8")
        ).digest()
        views.append(
            {
                "view_index": view_index,
                "hypothesis_id": hypothesis_id,
                "hypothesis_weight": float(hypothesis["weight"]),
                "caption": str(hypothesis["caption"]),
                "seed": int.from_bytes(digest[:4], "big") & 0x7FFFFFFF,
                "output": str(output_dir / f"{hypothesis_id}.png").replace("\\", "/"),
            }
        )
    return {
        "image": str(manifest["image"]),
        "source_key": source_key,
        "imagination_contract_sha256": manifest["sampling_contract_sha256"],
        **source_fields,
        "views": views,
    }
