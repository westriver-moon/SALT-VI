from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np


def _pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    return plt


def plot_feature_diagnostics(
    features: np.ndarray,
    labels: np.ndarray,
    distributions: Dict[str, np.ndarray],
    destination: Path,
    max_plot_samples: int,
    seed: int,
) -> Optional[str]:
    plt = _pyplot()
    if plt is None:
        return "matplotlib is not installed; feature figures were skipped"
    destination.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(seed)
    count = min(len(features), int(max_plot_samples))
    indices = rng.choice(len(features), count, replace=False) if count < len(features) else np.arange(len(features))
    sample = np.asarray(features[indices], dtype=np.float64)
    sample -= sample.mean(axis=0, keepdims=True)
    u, singular, _ = np.linalg.svd(sample, full_matrices=False)
    coordinates = u[:, :2] * singular[:2]
    if coordinates.shape[1] < 2:
        coordinates = np.pad(coordinates, ((0, 0), (0, 2 - coordinates.shape[1])))

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].hist(distributions["norms"], bins=60, color="#2563eb", alpha=0.85)
    axes[0].set_title("Feature norm")
    axes[0].set_xlabel("L2 norm")
    axes[0].set_ylabel("Count")

    within = distributions["within_cosine"]
    between = distributions["between_cosine"]
    if len(within):
        axes[1].hist(within, bins=60, density=True, alpha=0.65, label="within ID", color="#16a34a")
    if len(between):
        axes[1].hist(between, bins=60, density=True, alpha=0.60, label="between ID", color="#dc2626")
    axes[1].set_title("Cosine distributions")
    axes[1].set_xlabel("Cosine similarity")
    axes[1].legend()

    colors = labels[indices] % 20
    axes[2].scatter(coordinates[:, 0], coordinates[:, 1], c=colors, cmap="tab20", s=7, alpha=0.55)
    axes[2].set_title("PCA projection (ID color modulo 20)")
    axes[2].set_xlabel("PC1")
    axes[2].set_ylabel("PC2")
    figure.tight_layout()
    figure.savefig(destination / "feature_diagnostics.png", dpi=180, bbox_inches="tight")
    plt.close(figure)
    return None


def plot_comparison(distributions: Dict[str, np.ndarray], destination: Path) -> Optional[str]:
    plt = _pyplot()
    if plt is None:
        return "matplotlib is not installed; comparison figures were skipped"
    available = [(name, values) for name, values in distributions.items() if len(values)]
    if not available:
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 4))
    if len(available) == 1:
        axes = [axes]
    for axis, (name, values) in zip(axes, available):
        axis.hist(values, bins=60, color="#7c3aed", alpha=0.8)
        axis.set_title(name.replace("_", " "))
        axis.set_ylabel("Count")
    figure.tight_layout()
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return None
