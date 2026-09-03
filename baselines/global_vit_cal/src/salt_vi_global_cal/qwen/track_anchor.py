from __future__ import annotations

import statistics
from dataclasses import dataclass

from .schema import Region


@dataclass(frozen=True)
class TrackFrame:
    source_key: str
    global_quality: float
    regions: tuple[Region, ...]


def select_shared_region_ids(
    frames: list[TrackFrame], selected_count: int = 3
) -> list[str]:
    if not frames:
        raise ValueError("track must contain at least one frame")
    shared = {region.region_id for region in frames[0].regions}
    for frame in frames[1:]:
        shared.intersection_update(region.region_id for region in frame.regions)
    if len(shared) < selected_count:
        raise ValueError("track has too few ROI ids shared by every frame")
    uncertainty: dict[str, list[float]] = {name: [] for name in shared}
    for frame in frames:
        for region in frame.regions:
            if region.region_id in shared:
                uncertainty[region.region_id].append(float(region.uncertainty))
    ranked = sorted(
        shared,
        key=lambda name: (-statistics.median(uncertainty[name]), name),
    )
    return ranked[:selected_count]


def select_track_anchor(frames: list[TrackFrame], selected_ids: list[str]) -> TrackFrame:
    if not frames:
        raise ValueError("track must contain at least one frame")
    wanted = set(selected_ids)

    def score(frame: TrackFrame) -> tuple[float, str]:
        regions = {region.region_id: region for region in frame.regions}
        if not wanted.issubset(regions):
            raise ValueError("anchor candidate omits a selected ROI")
        clarity = sum(1.0 - regions[name].blur_uncertainty for name in wanted)
        return frame.global_quality + 0.1 * clarity, frame.source_key

    return max(frames, key=score)
