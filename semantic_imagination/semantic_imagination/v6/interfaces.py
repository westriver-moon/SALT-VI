from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from .contracts import (
    BackendDescriptor,
    BackendResult,
    JointWorld,
    Observation,
    RewriteRequest,
    SourceSpec,
)


class VLMBackend(Protocol):
    descriptor: BackendDescriptor

    def observe(self, source: SourceSpec) -> BackendResult[Observation]: ...

    def generate_joint_worlds(
        self,
        source: SourceSpec,
        observation: Observation,
        count: int,
        seed: int,
    ) -> BackendResult[Sequence[JointWorld]]:
        """Generate up to count complete worlds in one request, without repeated draws.

        Each candidate must preserve the observation's visible facts and contain
        one coherent assignment per ROI. Do not independently combine ROI
        alternatives or supply model-estimated probabilities.
        """
        ...


class SemanticEncoder(Protocol):
    descriptor: BackendDescriptor

    def encode(
        self, texts: Sequence[str]
    ) -> BackendResult[Sequence[Sequence[float]]]: ...


class RewriteBackend(Protocol):
    descriptor: BackendDescriptor

    def rewrite(self, request: RewriteRequest) -> BackendResult[str]: ...
