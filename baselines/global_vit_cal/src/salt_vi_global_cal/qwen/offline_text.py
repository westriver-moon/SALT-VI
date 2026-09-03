"""Convert retained Qwen regional worlds into deterministic offline text views."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import Tensor


@dataclass(frozen=True)
class OfflineTextWorld:
    """One retained regional world for consumption by a Stage-B text encoder."""

    world_id: str
    text: str
    weight: float


def retained_world_texts(payload: Mapping[str, object]) -> list[OfflineTextWorld]:
    """Read QwenRegionalPipeline output without inventing new semantic content.

    The strings preserve each retained atom's category, state and original
    atomic description. They can be tokenized by a Stage-B text encoder
    offline; this function never calls Qwen during training.
    """

    worlds_container = payload.get("worlds")
    if not isinstance(worlds_container, Mapping):
        raise ValueError("Qwen annotation omits the worlds mapping")
    worlds = worlds_container.get("worlds")
    if not isinstance(worlds, Sequence) or isinstance(worlds, (str, bytes)) or not worlds:
        raise ValueError("Qwen annotation has no retained worlds")
    result: list[OfflineTextWorld] = []
    for raw_world in worlds:
        if not isinstance(raw_world, Mapping):
            raise ValueError("Qwen world must be a mapping")
        world_id = raw_world.get("world_id")
        assignments = raw_world.get("assignments")
        weight = raw_world.get("selected_weight")
        if not isinstance(world_id, str) or not isinstance(assignments, Sequence):
            raise ValueError("Qwen world is missing its id or assignments")
        if not isinstance(weight, (int, float)) or float(weight) < 0.0:
            raise ValueError("Qwen world has an invalid selected weight")
        fragments = []
        for assignment in assignments:
            if not isinstance(assignment, Mapping):
                raise ValueError("Qwen assignment must be a mapping")
            category = assignment.get("category")
            state = assignment.get("state")
            description = assignment.get("description")
            if not all(isinstance(value, str) and value for value in (category, state, description)):
                raise ValueError("Qwen assignment is missing category, state or description")
            fragments.append(f"{category}: {state}; {description}")
        result.append(OfflineTextWorld(
            world_id=world_id,
            text="Regional appearance. " + " ".join(fragments),
            weight=float(weight),
        ))
    total = sum(item.weight for item in result)
    if total <= 0.0:
        raise ValueError("Qwen retained-world weights must have positive mass")
    return [OfflineTextWorld(item.world_id, item.text, item.weight / total) for item in result]


def weighted_world_feature_mean(world_features: Tensor, world_weights: Tensor) -> Tensor:
    """Aggregate tokenized offline worlds to one Stage-B text feature per image."""

    if world_features.ndim != 3:
        raise ValueError("world_features must have shape [B,K,D]")
    if world_weights.shape != world_features.shape[:2]:
        raise ValueError("world_weights must have shape [B,K]")
    if not torch.isfinite(world_features).all() or not torch.isfinite(world_weights).all():
        raise ValueError("Qwen world features and weights must be finite")
    if bool((world_weights < 0).any()):
        raise ValueError("Qwen world weights must be non-negative")
    mass = world_weights.sum(dim=1, keepdim=True)
    if bool((mass <= 0).any()):
        raise ValueError("each image needs at least one retained Qwen world")
    normalized = world_weights / mass
    return (world_features * normalized[:, :, None].to(world_features.dtype)).sum(dim=1)
