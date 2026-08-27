from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from .contracts import JointWorld


@dataclass(frozen=True)
class WorldCluster:
    member_indices: tuple[int, ...]
    representative_index: int


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        raise ValueError("semantic vectors must be non-zero")
    return dot / (left_norm * right_norm)


def _validated_vectors(
    vectors: Sequence[Sequence[float]], expected: int
) -> list[list[float]]:
    values = [list(map(float, vector)) for vector in vectors]
    if len(values) != expected or not values or not values[0]:
        raise ValueError("semantic encoder returned an invalid vector matrix")
    dimension = len(values[0])
    if any(
        len(vector) != dimension or any(not math.isfinite(value) for value in vector)
        for vector in values
    ):
        raise ValueError("semantic vectors must be finite and rectangular")
    return values


def _minimum_similarity(
    left: Sequence[int], right: Sequence[int], vectors: Sequence[Sequence[float]]
) -> float:
    return min(cosine(vectors[a], vectors[b]) for a in left for b in right)


def _complete_link(
    indices: Sequence[int],
    vectors: Sequence[Sequence[float]],
    threshold: float,
) -> list[list[int]]:
    clusters = [[index] for index in indices]
    while True:
        eligible = []
        for left_index, left in enumerate(clusters):
            for right_index, right in enumerate(
                clusters[left_index + 1 :], left_index + 1
            ):
                similarity = _minimum_similarity(left, right, vectors)
                if similarity >= threshold:
                    eligible.append((similarity, left_index, right_index))
        if not eligible:
            return clusters
        _, left_index, right_index = max(
            eligible,
            key=lambda item: (
                item[0],
                -clusters[item[1]][0],
                -clusters[item[2]][0],
            ),
        )
        clusters[left_index] = sorted(clusters[left_index] + clusters[right_index])
        del clusters[right_index]


def _medoid(indices: Sequence[int], vectors: Sequence[Sequence[float]]) -> int:
    return min(
        indices,
        key=lambda index: (
            sum(1.0 - cosine(vectors[index], vectors[other]) for other in indices),
            index,
        ),
    )


def cluster_joint_worlds(
    worlds: Sequence[JointWorld],
    vectors: Sequence[Sequence[float]],
    similarity_threshold: float,
) -> list[WorldCluster]:
    if not worlds:
        raise ValueError("joint-world clustering requires at least one candidate")
    if not -1.0 <= similarity_threshold <= 1.0:
        raise ValueError("similarity_threshold must be within [-1, 1]")
    embedded = _validated_vectors(vectors, len(worlds))
    buckets: dict[tuple[tuple[str, str, str], ...], list[int]] = defaultdict(list)
    for index, world in enumerate(worlds):
        buckets[world.state_signature()].append(index)

    groups = [
        group
        for signature in sorted(buckets)
        for group in _complete_link(buckets[signature], embedded, similarity_threshold)
    ]
    groups.sort(key=lambda group: group[0])
    return [WorldCluster(tuple(group), _medoid(group, embedded)) for group in groups]
