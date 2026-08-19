from __future__ import annotations

from collections import Counter

from semantic_imagination.taxonomy import SENTINEL_STATES

from .qwen import RegionalReasoner, regional_entropy
from .schema import Assignment, Candidate, JointSample, Region, World


NON_EDIT_STATES = set(SENTINEL_STATES) | {"unresolved"}


def _world_key(world: dict[str, Candidate]) -> tuple[tuple[str, str, str], ...]:
    return tuple(
        (region_id, candidate.state, candidate.value)
        for region_id, candidate in sorted(world.items())
    )


def _caption(assignments: list[Assignment]) -> str:
    details = [
        f"{assignment.region_id}: {assignment.value}"
        for assignment in assignments
        if assignment.state not in NON_EDIT_STATES
    ]
    if not details:
        return "faithful pedestrian restoration with no additional uncertain detail"
    return (
        "faithful pedestrian restoration; preserve identity and pose; regional details: "
        + "; ".join(details)
    )


def build_worlds(
    reasoner: RegionalReasoner,
    lr,
    swin,
    regions: list[Region],
    proposals: dict[str, list[Candidate]],
    samples: list[JointSample],
    *,
    max_worlds: int,
    seed: int,
    ensure_editing_coverage: bool = False,
) -> list[World]:
    if not samples:
        raise ValueError("joint world construction requires Qwen samples")
    for region in regions:
        region.candidates = list(proposals[region.region_id])
        region.u_qwen_proposal = regional_entropy(
            samples, region.region_id, len(proposals[region.region_id])
        )

    representatives: dict[tuple[tuple[str, str, str], ...], dict[str, Candidate]] = {}
    counts = Counter()
    coverage_counts = Counter()
    free_counts = Counter()
    for sample in samples:
        key = _world_key(sample.assignments)
        representatives.setdefault(key, sample.assignments)
        counts[key] += 1
        if sample.origin == "coverage":
            coverage_counts[key] += 1
        else:
            free_counts[key] += 1
    ordered_keys = sorted(counts, key=lambda key: (-counts[key], key))
    unique = [representatives[key] for key in ordered_keys]
    critiques = reasoner.critique(lr, swin, regions, unique)
    accepted: list[
        tuple[tuple[tuple[str, str, str], ...], dict[str, Candidate], dict]
    ] = []
    region_by_id = {region.region_id: region for region in regions}
    for key, sample, critic in zip(ordered_keys, unique, critiques):
        for region_id, result in critic.items():
            region_by_id[region_id].critic.append(
                {
                    "world_key": [list(item) for item in key],
                    **result,
                }
            )
        if any(result["label"] == "contradicted" for result in critic.values()):
            continue
        accepted.append((key, sample, critic))
    if not accepted:
        raise ValueError("Qwen critic rejected every sampled world")
    compatible_samples = [
        JointSample(sample, "free")
        for key, sample, _ in accepted
        for _ in range(counts[key])
    ]
    for region in regions:
        region.u_qwen_compatible = regional_entropy(
            compatible_samples,
            region.region_id,
            len(proposals[region.region_id]),
        )
        region.u_qwen = region.u_qwen_compatible

    if ensure_editing_coverage:
        selected = []
        baseline = next(
            (
                item
                for item in accepted
                if all(
                    candidate.state in NON_EDIT_STATES for candidate in item[1].values()
                )
            ),
            None,
        )
        if baseline is not None:
            selected.append(baseline)
        for region in regions:
            candidate = next(
                (
                    item
                    for item in accepted
                    if item not in selected
                    and item[1][region.region_id].state not in NON_EDIT_STATES
                ),
                None,
            )
            if candidate is not None:
                selected.append(candidate)
        selected.extend(item for item in accepted if item not in selected)
        accepted = selected[: int(max_worlds)]
    else:
        accepted = accepted[: int(max_worlds)]
    raw_mass = [counts[key] / len(samples) for key, _, _ in accepted]
    selected_mass = sum(raw_mass)
    if selected_mass <= 0:
        raise ValueError("selected Qwen worlds have no proposal mass")
    worlds = []
    for index, ((key, sample, critic), mass) in enumerate(zip(accepted, raw_mass)):
        assignments = [
            Assignment(
                region_id=region_id,
                category=region_by_id[region_id].category,
                state=candidate.state,
                value=candidate.value,
                critic_label=critic[region_id]["label"],
                critic_score=float(critic[region_id]["score"]),
            )
            for region_id, candidate in sorted(sample.items())
        ]
        worlds.append(
            World(
                world_id=f"w{index:02d}",
                assignments=assignments,
                sample_count=int(counts[key]),
                proposal_mass=float(mass),
                coverage_sample_count=int(coverage_counts[key]),
                free_sample_count=int(free_counts[key]),
                proposal_weight=float(mass / selected_mass),
                caption=_caption(assignments),
                seed=int(seed + index * 1009),
            )
        )
    uniform = 1.0 / len(worlds)
    for world in worlds:
        world.uniform_weight = uniform
    return worlds


def edited_region_ids(world: World) -> set[str]:
    return {
        assignment.region_id
        for assignment in world.assignments
        if assignment.state not in NON_EDIT_STATES
    }
