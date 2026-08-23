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

    def sample_joint_world(
        self,
        source: SourceSpec,
        observation: Observation,
        seed: int,
    ) -> BackendResult[JointWorld]: ...


class SemanticEncoder(Protocol):
    descriptor: BackendDescriptor

    def encode(
        self, texts: Sequence[str]
    ) -> BackendResult[Sequence[Sequence[float]]]: ...


class RewriteBackend(Protocol):
    descriptor: BackendDescriptor

    def rewrite(self, request: RewriteRequest) -> BackendResult[str]: ...
