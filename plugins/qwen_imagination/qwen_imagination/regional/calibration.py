from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .schema import World


@dataclass(frozen=True)
class CalibrationWeights:
    alpha: float = 1.0
    beta: float = 8.0
    gamma: float = 4.0
    delta: float = 16.0
    epsilon: float = 1e-8


def cosine_identity_energy(reference: np.ndarray, candidate: np.ndarray) -> float:
    reference = np.asarray(reference, dtype=np.float64).reshape(-1)
    candidate = np.asarray(candidate, dtype=np.float64).reshape(-1)
    denominator = float(np.linalg.norm(reference) * np.linalg.norm(candidate))
    if denominator <= 0:
        raise ValueError("identity features must be non-zero")
    return float(max(0.0, 1.0 - np.dot(reference, candidate) / denominator))


def calibrate_world_weights(
    worlds: list[World], coefficients: CalibrationWeights = CalibrationWeights()
) -> list[World]:
    if not worlds:
        raise ValueError("cannot calibrate an empty world set")
    logits = []
    for world in worlds:
        q = max(float(world.proposal_weight), coefficients.epsilon)
        values = (world.e_lr, world.e_id, world.e_edit)
        if any(not np.isfinite(value) or value < 0 for value in values):
            raise ValueError(f"world {world.world_id} has invalid calibration energies: {values}")
        logits.append(
            coefficients.alpha * math.log(q + coefficients.epsilon)
            - coefficients.beta * float(world.e_lr)
            - coefficients.gamma * float(world.e_id)
            - coefficients.delta * float(world.e_edit)
        )
    maximum = max(logits)
    masses = [math.exp(value - maximum) for value in logits]
    total = sum(masses)
    for world, mass in zip(worlds, masses):
        world.posterior_weight = float(mass / total)
    return worlds
