"""Pure NumPy evaluation operations used by the metric-boost evaluator.

This module intentionally has no dataset, checkpoint, or CUDA dependency so its
protocol-sensitive math can be checked on CPU before any experiment is launched.
"""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np


EPS = 1e-12


def l2_normalize(features: np.ndarray, axis: int = 1) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32)
    norms = np.linalg.norm(values, ord=2, axis=axis, keepdims=True)
    return values / np.maximum(norms, EPS)


def aggregate_tta(feature_views: Sequence[np.ndarray]) -> np.ndarray:
    if not feature_views:
        raise ValueError("feature_views must contain at least one view")
    shapes = {tuple(np.asarray(view).shape) for view in feature_views}
    if len(shapes) != 1:
        raise ValueError(f"TTA feature shapes do not match: {sorted(shapes)}")
    normalized = [l2_normalize(view) for view in feature_views]
    return l2_normalize(np.mean(normalized, axis=0))


def similarity(query: np.ndarray, gallery: np.ndarray, normalize: bool = True) -> np.ndarray:
    query_values = l2_normalize(query) if normalize else np.asarray(query, dtype=np.float32)
    gallery_values = l2_normalize(gallery) if normalize else np.asarray(gallery, dtype=np.float32)
    if query_values.ndim != 2 or gallery_values.ndim != 2:
        raise ValueError("query and gallery features must be rank-2 arrays")
    if query_values.shape[1] != gallery_values.shape[1]:
        raise ValueError(
            f"Feature dimensions differ: {query_values.shape[1]} vs {gallery_values.shape[1]}"
        )
    return np.matmul(query_values, gallery_values.T)


def weighted_mer_score(
    fusion_query: np.ndarray,
    ir_query: np.ndarray,
    text_query: np.ndarray,
    gallery: np.ndarray,
    weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    normalize: bool = True,
) -> np.ndarray:
    if len(weights) != 3:
        raise ValueError("MER weights must be (fusion, ir, text)")
    fusion_weight, ir_weight, text_weight = (float(item) for item in weights)
    return (
        fusion_weight * similarity(fusion_query, gallery, normalize=normalize)
        + ir_weight * similarity(ir_query, gallery, normalize=normalize)
        + text_weight * similarity(text_query, gallery, normalize=normalize)
    )


def parse_scales(scales: Iterable[Sequence[int]]) -> Tuple[Tuple[int, int], ...]:
    parsed = []
    for scale in scales:
        if len(scale) != 2:
            raise ValueError(f"Scale must be [height, width], got {scale!r}")
        height, width = int(scale[0]), int(scale[1])
        if height <= 0 or width <= 0:
            raise ValueError(f"Scale must be positive, got {height}x{width}")
        value = (height, width)
        if value not in parsed:
            parsed.append(value)
    if not parsed:
        raise ValueError("At least one test scale is required")
    return tuple(parsed)


def _k_reciprocal_rerank(original_dist: np.ndarray, query_count: int, k1: int, k2: int, lambda_value: float) -> np.ndarray:
    all_count = original_dist.shape[0]
    if original_dist.shape != (all_count, all_count):
        raise ValueError("original_dist must be square")
    if not 0 < query_count < all_count:
        raise ValueError("query_count must split a non-empty query and gallery set")
    if k1 < 1 or k1 >= all_count:
        raise ValueError(f"k1 must be in [1, {all_count - 1}], got {k1}")
    if k2 < 1:
        raise ValueError("k2 must be positive")
    if not 0.0 <= lambda_value <= 1.0:
        raise ValueError("lambda_value must be between 0 and 1")

    original_dist = np.asarray(original_dist, dtype=np.float32)
    column_max = np.maximum(np.max(original_dist, axis=0, keepdims=True), EPS)
    original_dist = np.transpose(original_dist / column_max)
    initial_rank = np.argsort(original_dist, axis=1).astype(np.int32)
    v_matrix = np.zeros_like(original_dist, dtype=np.float32)
    half_k = int(np.around(k1 / 2.0))

    for index in range(all_count):
        forward = initial_rank[index, : k1 + 1]
        backward = initial_rank[forward, : k1 + 1]
        reciprocal = forward[np.where(backward == index)[0]]
        expansion = reciprocal
        for candidate in reciprocal:
            candidate_forward = initial_rank[candidate, : half_k + 1]
            candidate_backward = initial_rank[candidate_forward, : half_k + 1]
            candidate_reciprocal = candidate_forward[np.where(candidate_backward == candidate)[0]]
            overlap = np.intersect1d(candidate_reciprocal, reciprocal).size
            if overlap > (2.0 / 3.0) * max(1, candidate_reciprocal.size):
                expansion = np.append(expansion, candidate_reciprocal)
        expansion = np.unique(expansion)
        weights = np.exp(-original_dist[index, expansion])
        v_matrix[index, expansion] = weights / np.maximum(np.sum(weights), EPS)

    if k2 > 1:
        v_matrix = np.asarray(
            [np.mean(v_matrix[initial_rank[index, :k2]], axis=0) for index in range(all_count)],
            dtype=np.float32,
        )

    inverted = [np.where(v_matrix[:, index] != 0)[0] for index in range(all_count)]
    jaccard = np.zeros((query_count, all_count), dtype=np.float32)
    for query_index in range(query_count):
        temp_min = np.zeros(all_count, dtype=np.float32)
        nonzero = np.where(v_matrix[query_index] != 0)[0]
        related = np.unique(np.concatenate([inverted[index] for index in nonzero])) if nonzero.size else []
        for candidate in related:
            shared = np.intersect1d(nonzero, np.where(v_matrix[candidate] != 0)[0], assume_unique=True)
            temp_min[candidate] = np.sum(
                np.minimum(v_matrix[query_index, shared], v_matrix[candidate, shared])
            )
        jaccard[query_index] = 1.0 - temp_min / np.maximum(2.0 - temp_min, EPS)

    return (
        jaccard[:, query_count:] * (1.0 - lambda_value)
        + original_dist[:query_count, query_count:] * lambda_value
    )


def rerank_cosine(
    query: np.ndarray,
    gallery: np.ndarray,
    k1: int = 20,
    k2: int = 6,
    lambda_value: float = 0.3,
) -> np.ndarray:
    """Return a query-gallery distance matrix using standard k-reciprocal re-ranking."""
    query_values = l2_normalize(query)
    gallery_values = l2_normalize(gallery)
    combined = np.concatenate([query_values, gallery_values], axis=0)
    cosine_distance = np.maximum(0.0, 2.0 - 2.0 * np.matmul(combined, combined.T))
    return _k_reciprocal_rerank(
        cosine_distance,
        query_count=query_values.shape[0],
        k1=int(k1),
        k2=int(k2),
        lambda_value=float(lambda_value),
    )

