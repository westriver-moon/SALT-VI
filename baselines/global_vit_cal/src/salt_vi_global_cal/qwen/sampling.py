from __future__ import annotations

import random
from collections import Counter

from .hypotheses import RegionHypotheses


def sample_joint_worlds(
    regions: list[RegionHypotheses],
    *,
    sample_count: int = 64,
    max_worlds: int = 8,
    seed: int = 0,
) -> dict[str, object]:
    """Sample empirical regional tuples and compress them to retained worlds."""

    if not regions:
        raise ValueError("joint-world sampling requires regional hypotheses")
    if sample_count < 1 or not 1 <= max_worlds <= sample_count:
        raise ValueError("invalid joint-world sample_count or max_worlds")
    rng = random.Random(int(seed))
    draws: list[tuple[int, ...]] = []
    for _ in range(sample_count):
        draw = []
        for region in regions:
            if not region.clusters:
                raise ValueError(f"region {region.region_id} has no valid clusters")
            draw.append(
                rng.choices(
                    range(len(region.clusters)),
                    weights=[cluster.weight for cluster in region.clusters],
                    k=1,
                )[0]
            )
        draws.append(tuple(draw))
    counts = Counter(draws)
    retained = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:max_worlds]
    retained_count = sum(count for _, count in retained)
    worlds = []
    for world_index, (indices, count) in enumerate(retained):
        assignments = []
        for region, cluster_index in zip(regions, indices):
            cluster = region.clusters[cluster_index]
            assignments.append(
                {
                    "region_id": region.region_id,
                    "category": region.category,
                    "cluster_id": cluster.cluster_id,
                    "state": cluster.state,
                    "description": cluster.representative,
                    "empirical_mass": cluster.empirical_mass,
                    "weight": cluster.weight,
                }
            )
        worlds.append(
            {
                "world_id": f"w{world_index:02d}",
                "assignments": assignments,
                "sample_count": count,
                "sample_frequency": count / sample_count,
                "selected_weight": count / max(1, retained_count),
            }
        )
    return {
        "seed": int(seed),
        "sample_count": sample_count,
        "unique_world_count": len(counts),
        "retained_world_count": len(worlds),
        "retained_sample_mass": retained_count / sample_count,
        "worlds": worlds,
    }
