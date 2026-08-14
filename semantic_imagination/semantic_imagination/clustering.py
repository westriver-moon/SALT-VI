from __future__ import annotations

import math
import re
from collections.abc import Sequence


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        raise ValueError("semantic embeddings must be non-zero")
    return dot / (left_norm * right_norm)


def _single_link(vectors: Sequence[Sequence[float]], threshold: float) -> list[list[int]]:
    parent = list(range(len(vectors)))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for left in range(len(vectors)):
        for right in range(left + 1, len(vectors)):
            if cosine(vectors[left], vectors[right]) >= threshold:
                a, b = root(left), root(right)
                parent[max(a, b)] = min(a, b)
    groups: dict[int, list[int]] = {}
    for index in range(len(vectors)):
        groups.setdefault(root(index), []).append(index)
    return sorted(groups.values(), key=lambda group: group[0])


def _complete_link(vectors: Sequence[Sequence[float]], threshold: float) -> list[list[int]]:
    clusters: dict[int, list[int]] = {index: [index] for index in range(len(vectors))}
    similarities = {
        (left, right): cosine(vectors[left], vectors[right])
        for left in range(len(vectors))
        for right in range(left + 1, len(vectors))
    }
    while True:
        active = sorted(clusters)
        eligible = [
            (similarities[(left, right)], left, right)
            for position, left in enumerate(active)
            for right in active[position + 1 :]
            if similarities[(left, right)] >= threshold
        ]
        if not eligible:
            break
        _, left, right = max(
            eligible,
            key=lambda item: (item[0], -clusters[item[1]][0], -clusters[item[2]][0]),
        )
        others = [index for index in active if index not in (left, right)]
        updated = {
            (min(left, other), max(left, other)): min(
                similarities[(min(left, other), max(left, other))],
                similarities[(min(right, other), max(right, other))],
            )
            for other in others
        }
        clusters[left] = sorted(clusters[left] + clusters[right])
        del clusters[right]
        similarities = {
            key: value
            for key, value in similarities.items()
            if right not in key and left not in key
        }
        similarities.update(updated)
    return sorted(clusters.values(), key=lambda group: group[0])


def wilson_interval(count: int, total: int) -> dict[str, float | str]:
    if total < 1:
        raise ValueError("Wilson interval requires a positive total")
    z = 1.959963984540054
    estimate = count / total
    denominator = 1.0 + z * z / total
    center = (estimate + z * z / (2.0 * total)) / denominator
    margin = z / denominator * math.sqrt(
        estimate * (1.0 - estimate) / total + z * z / (4.0 * total * total)
    )
    return {
        "method": "wilson-95-conditional-on-fixed-clusters",
        "lower": max(0.0, center - margin),
        "upper": min(1.0, center + margin),
    }


def _validate_vectors(vectors: Sequence[Sequence[float]], expected: int) -> list[list[float]]:
    values = [list(map(float, vector)) for vector in vectors]
    if len(values) != expected or not values or not values[0]:
        raise ValueError("backend returned an invalid embedding matrix")
    dimension = len(values[0])
    if any(
        len(vector) != dimension or any(not math.isfinite(value) for value in vector)
        for vector in values
    ):
        raise ValueError("semantic embeddings must be finite and rectangular")
    return values


def _medoid(indices: Sequence[int], vectors: Sequence[Sequence[float]]) -> int:
    return min(
        indices,
        key=lambda index: (
            sum(1.0 - cosine(vectors[index], vectors[other]) for other in indices),
            index,
        ),
    )


def cluster_hypothesis_samples(
    texts: Sequence[str],
    vectors: Sequence[Sequence[float]],
    similarity_threshold: float,
    cluster_linkage: str = "complete",
) -> list[dict]:
    values = [str(text).strip() for text in texts]
    if not values or any(not value for value in values):
        raise ValueError("hypothesis texts must be non-empty")
    if not -1.0 <= similarity_threshold <= 1.0:
        raise ValueError("similarity_threshold must be in [-1, 1]")
    if cluster_linkage not in {"complete", "single"}:
        raise ValueError("cluster_linkage must be 'complete' or 'single'")
    embedded = _validate_vectors(vectors, len(values))

    category_pattern = re.compile(r"(?:^|;\s*)category=([a-z0-9_]+)(?:;|$)", re.I)
    state_pattern = re.compile(r"(?:^|;\s*)state=([a-z0-9_]+)(?:;|$)", re.I)
    value_pattern = re.compile(r"(?:^|;\s*)value=([^;]+)(?:;|$)", re.I)
    buckets: dict[str, list[int]] = {}
    controlled: set[str] = set()
    categories: list[str] = []
    for index, value in enumerate(values):
        category_match = category_pattern.search(value)
        category = category_match.group(1).casefold() if category_match else "__unstructured__"
        categories.append(category)
        state_match = state_pattern.search(value)
        if state_match:
            key = f"controlled:{category}:{state_match.group(1).casefold()}"
            controlled.add(key)
        else:
            value_match = value_pattern.search(value)
            atom = value_match.group(1).strip().casefold() if value_match else ""
            hard_state = atom if atom in {"absent", "no_additional_detail"} else "__positive__"
            key = f"{category}:{hard_state}"
        buckets.setdefault(key, []).append(index)

    groups: list[list[int]] = []
    for key, indices in buckets.items():
        if key in controlled:
            groups.append(list(indices))
            continue
        local = [embedded[index] for index in indices]
        local_groups = (
            _complete_link(local, similarity_threshold)
            if cluster_linkage == "complete"
            else _single_link(local, similarity_threshold)
        )
        groups.extend([[indices[index] for index in group] for group in local_groups])
    groups.sort(key=lambda group: group[0])

    category_totals = {category: categories.count(category) for category in set(categories)}
    clusters = []
    for cluster_id, members in enumerate(groups):
        representative = _medoid(members, embedded)
        count = len(members)
        category = categories[representative]
        state_match = state_pattern.search(values[representative])
        state = state_match.group(1).casefold() if state_match else None
        category_total = category_totals[category]
        clusters.append(
            {
                "cluster_id": cluster_id,
                "category": category,
                **({"state": state} if state is not None else {}),
                "representative_index": representative,
                "representative": values[representative],
                "member_indices": members,
                "count": count,
                "weight": count / len(values),
                "weight_interval_95": wilson_interval(count, len(values)),
                "category_weight": category_total / len(values),
                "conditional_weight": count / category_total,
                "conditional_weight_interval_95": wilson_interval(count, category_total),
            }
        )
    return clusters
