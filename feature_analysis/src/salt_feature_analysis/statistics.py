from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from .storage import FeatureArtifact


EPS = 1e-12


def _row_normalize(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, EPS)


def _quantiles(values: np.ndarray) -> Dict[str, float]:
    if not len(values):
        return {name: float("nan") for name in ("min", "p05", "p25", "median", "p75", "p95", "max")}
    points = np.quantile(values, [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0])
    return dict(zip(("min", "p05", "p25", "median", "p75", "p95", "max"), map(float, points)))


def _sample_pair_cosines(
    unit: np.ndarray,
    labels: np.ndarray,
    maximum: int,
    rng: np.random.RandomState,
) -> Tuple[np.ndarray, np.ndarray]:
    groups = {int(label): np.flatnonzero(labels == label) for label in np.unique(labels)}
    eligible = [indices for indices in groups.values() if len(indices) >= 2]
    within = []
    if eligible:
        for _ in range(maximum):
            indices = eligible[rng.randint(len(eligible))]
            pair = rng.choice(indices, 2, replace=False)
            within.append(float(np.dot(unit[pair[0]], unit[pair[1]])))
    between = []
    attempts = 0
    while len(between) < maximum and attempts < maximum * 20:
        left, right = rng.randint(len(unit), size=2)
        attempts += 1
        if left != right and labels[left] != labels[right]:
            between.append(float(np.dot(unit[left], unit[right])))
    return np.asarray(within, dtype=np.float32), np.asarray(between, dtype=np.float32)


def _centroids(unit: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    unique = np.unique(labels)
    values = np.stack([unit[labels == label].mean(axis=0) for label in unique])
    return unique.astype(np.int64), values.astype(np.float32)


def analyze_feature_artifact(
    artifact: FeatureArtifact,
    seed: int,
    max_pair_samples: int,
    max_svd_samples: int,
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    features = np.asarray(artifact.features, dtype=np.float64)
    labels = artifact.labels
    finite_mask = np.isfinite(features)
    row_finite = finite_mask.all(axis=1)
    clean = np.where(finite_mask, features, 0.0)
    norms = np.linalg.norm(clean, axis=1)
    unit = _row_normalize(clean)
    rng = np.random.RandomState(seed)
    within, between = _sample_pair_cosines(unit, labels, int(max_pair_samples), rng)
    unique_labels, centroids = _centroids(unit, labels)
    centroid_unit = _row_normalize(centroids)
    centroid_lookup = {int(label): index for index, label in enumerate(unique_labels)}
    assigned = np.asarray([centroid_lookup[int(label)] for label in labels], dtype=np.int64)
    compactness = np.sum(unit * centroid_unit[assigned], axis=1)
    predictions = unique_labels[np.argmax(unit @ centroid_unit.T, axis=1)]

    if len(clean) > int(max_svd_samples):
        sample = clean[rng.choice(len(clean), int(max_svd_samples), replace=False)]
    else:
        sample = clean
    centered = sample - sample.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    eigenvalues = (singular ** 2) / max(len(centered) - 1, 1)
    total = float(eigenvalues.sum())
    probabilities = eigenvalues / max(total, EPS)
    entropy = -float(np.sum(probabilities * np.log(np.maximum(probabilities, EPS))))
    effective_rank = float(np.exp(entropy))
    participation_ratio = float(total ** 2 / max(float(np.square(eigenvalues).sum()), EPS))

    random_left = rng.randint(len(unit), size=min(int(max_pair_samples), max(len(unit), 1)))
    random_right = rng.randint(len(unit), size=len(random_left))
    anisotropy_values = np.sum(unit[random_left] * unit[random_right], axis=1)
    summary = {
        "sample_count": int(features.shape[0]),
        "feature_dim": int(features.shape[1]),
        "identity_count": int(len(unique_labels)),
        "finite_value_fraction": float(finite_mask.mean()),
        "finite_row_fraction": float(row_finite.mean()),
        "zero_norm_fraction": float(np.mean(norms <= EPS)),
        "norm_mean": float(norms.mean()),
        "norm_std": float(norms.std()),
        "norm_quantiles": _quantiles(norms),
        "feature_mean_l2": float(np.linalg.norm(clean.mean(axis=0))),
        "mean_dimension_variance": float(clean.var(axis=0).mean()),
        "anisotropy_mean_pair_cosine": float(anisotropy_values.mean()),
        "within_identity_cosine_mean": float(within.mean()) if len(within) else float("nan"),
        "within_identity_cosine_std": float(within.std()) if len(within) else float("nan"),
        "between_identity_cosine_mean": float(between.mean()) if len(between) else float("nan"),
        "between_identity_cosine_std": float(between.std()) if len(between) else float("nan"),
        "cosine_separation": float(within.mean() - between.mean()) if len(within) and len(between) else float("nan"),
        "identity_centroid_compactness_mean": float(compactness.mean()),
        "nearest_centroid_accuracy": float(np.mean(predictions == labels)),
        "effective_rank": effective_rank,
        "participation_ratio": participation_ratio,
        "top_eigenvalue_fraction": float(probabilities[0]) if len(probabilities) else float("nan"),
    }
    distributions = {
        "norms": norms.astype(np.float32),
        "within_cosine": within,
        "between_cosine": between,
    }
    return summary, distributions


def compare_artifacts(
    left: FeatureArtifact,
    right: FeatureArtifact,
    compute_retrieval: bool = False,
    compute_sample_retrieval: bool = True,
    retrieval_chunk_size: int = 1024,
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    if left.features.shape[1] != right.features.shape[1]:
        raise ValueError("Feature dimensions differ; checkpoint/protocol comparison is undefined")
    left_unit = _row_normalize(np.asarray(left.features, dtype=np.float64))
    right_unit = _row_normalize(np.asarray(right.features, dtype=np.float64))

    left_map = {value: index for index, value in enumerate(left.sample_ids.tolist())}
    right_map = {value: index for index, value in enumerate(right.sample_ids.tolist())}
    common_ids = sorted(set(left_map) & set(right_map))
    same_cosine = np.asarray([], dtype=np.float32)
    same_l2 = np.asarray([], dtype=np.float32)
    cka = float("nan")
    procrustes = float("nan")
    if common_ids:
        left_aligned = np.stack([left_unit[left_map[value]] for value in common_ids])
        right_aligned = np.stack([right_unit[right_map[value]] for value in common_ids])
        same_cosine = np.sum(left_aligned * right_aligned, axis=1).astype(np.float32)
        same_l2 = np.linalg.norm(left_aligned - right_aligned, axis=1).astype(np.float32)
        if len(common_ids) >= 2:
            x = left_aligned - left_aligned.mean(axis=0, keepdims=True)
            y = right_aligned - right_aligned.mean(axis=0, keepdims=True)
            numerator = float(np.square(x.T @ y).sum())
            denominator = np.sqrt(float(np.square(x.T @ x).sum()) * float(np.square(y.T @ y).sum()))
            cka = numerator / max(denominator, EPS)
            u, _, vt = np.linalg.svd(x.T @ y, full_matrices=False)
            rotation = u @ vt
            procrustes = float(np.linalg.norm(x @ rotation - y) / max(np.linalg.norm(y), EPS))

    left_labels, left_centroids = _centroids(left_unit, left.labels)
    right_labels, right_centroids = _centroids(right_unit, right.labels)
    left_centroid_map = {int(label): index for index, label in enumerate(left_labels)}
    right_centroid_map = {int(label): index for index, label in enumerate(right_labels)}
    common_labels = sorted(set(left_centroid_map) & set(right_centroid_map))
    centroid_cosine = np.asarray(
        [
            float(
                np.dot(
                    _row_normalize(left_centroids[[left_centroid_map[label]]])[0],
                    _row_normalize(right_centroids[[right_centroid_map[label]]])[0],
                )
            )
            for label in common_labels
        ],
        dtype=np.float32,
    )

    summary = {
        "left_sample_count": int(len(left.features)),
        "right_sample_count": int(len(right.features)),
        "common_sample_count": int(len(common_ids)),
        "common_identity_count": int(len(common_labels)),
        "same_sample_cosine_mean": float(same_cosine.mean()) if len(same_cosine) else float("nan"),
        "same_sample_cosine_std": float(same_cosine.std()) if len(same_cosine) else float("nan"),
        "same_sample_l2_mean": float(same_l2.mean()) if len(same_l2) else float("nan"),
        "linear_cka": float(cka),
        "orthogonal_procrustes_residual": float(procrustes),
        "label_centroid_cosine_mean": float(centroid_cosine.mean()) if len(centroid_cosine) else float("nan"),
        "label_centroid_cosine_std": float(centroid_cosine.std()) if len(centroid_cosine) else float("nan"),
    }
    if compute_retrieval:
        if compute_sample_retrieval:
            summary.update(
                _label_retrieval(left_unit, left.labels, right_unit, right.labels, retrieval_chunk_size)
            )
        centroid_retrieval = _label_retrieval(
            _row_normalize(left_centroids),
            left_labels,
            _row_normalize(right_centroids),
            right_labels,
            retrieval_chunk_size,
        )
        summary.update(
            {
                key.replace("label_retrieval_", "label_centroid_retrieval_"): value
                for key, value in centroid_retrieval.items()
            }
        )
    distributions = {
        "same_sample_cosine": same_cosine,
        "same_sample_l2": same_l2,
        "label_centroid_cosine": centroid_cosine,
    }
    return summary, distributions


def _label_retrieval(
    query: np.ndarray,
    query_labels: np.ndarray,
    gallery: np.ndarray,
    gallery_labels: np.ndarray,
    chunk_size: int,
) -> Dict[str, float]:
    top1 = 0
    top5 = 0
    reciprocal_ranks = []
    for start in range(0, len(query), int(chunk_size)):
        similarities = query[start : start + chunk_size] @ gallery.T
        batch_labels = query_labels[start : start + len(similarities)]
        matches = gallery_labels[None, :] == batch_labels[:, None]
        has_positive = matches.any(axis=1)
        # The first relevant rank is one plus the number of gallery scores
        # strictly above the best positive score. This is equivalent to a full
        # descending sort for continuous similarities but avoids sorting every
        # training-gallery row merely to recover top-k and MRR.
        best_positive = np.max(np.where(matches, similarities, -np.inf), axis=1)
        first_positive_rank = 1 + np.sum(similarities > best_positive[:, None], axis=1)
        top1 += int(np.sum(has_positive & (first_positive_rank <= 1)))
        top5 += int(np.sum(has_positive & (first_positive_rank <= 5)))
        reciprocal_ranks.extend(
            np.where(has_positive, 1.0 / first_positive_rank, 0.0).astype(np.float64).tolist()
        )
    denominator = max(len(query), 1)
    return {
        "label_retrieval_top1": float(top1 / denominator),
        "label_retrieval_top5": float(top5 / denominator),
        "label_retrieval_mrr": float(np.mean(reciprocal_ranks)),
    }
