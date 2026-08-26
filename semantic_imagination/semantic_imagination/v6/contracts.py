from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Generic, Mapping, Sequence, TypeVar


QRI_V6 = "qri-v6"
T = TypeVar("T")


@dataclass(frozen=True)
class BackendDescriptor:
    backend_id: str
    model_id: str
    revision: str
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "backend_id": self.backend_id,
            "model_id": self.model_id,
            "revision": self.revision,
            "parameters": dict(self.parameters),
        }


@dataclass(frozen=True)
class BackendResult(Generic[T]):
    value: T
    usage: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class RegionSpec:
    region_id: str
    category: str
    bbox_xyxy: tuple[int, int, int, int]

    def manifest(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "category": self.category,
            "bbox_xyxy": list(self.bbox_xyxy),
        }


@dataclass(frozen=True)
class SourceSpec:
    source_key: str
    image: Path
    modality: str
    regions: tuple[RegionSpec, ...]

    def __post_init__(self) -> None:
        if not self.source_key or "\\" in self.source_key:
            raise ValueError("source_key must be a non-empty POSIX path")
        key = PurePosixPath(self.source_key)
        if (
            key.is_absolute()
            or ".." in key.parts
            or (key.parts and key.parts[0].endswith(":"))
        ):
            raise ValueError("source_key must be a relative POSIX path")
        if not self.regions:
            raise ValueError("qri-v6 requires at least one region")
        identifiers = [region.region_id for region in self.regions]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("qri-v6 region ids must be unique")


@dataclass(frozen=True)
class RegionObservation:
    region_id: str
    facts: tuple[str, ...] = ()

    def manifest(self) -> dict[str, Any]:
        return {"region_id": self.region_id, "facts": list(self.facts)}


@dataclass(frozen=True)
class Observation:
    caption: str
    regions: tuple[RegionObservation, ...]

    def manifest(self) -> dict[str, Any]:
        return {
            "caption": self.caption,
            "regions": [region.manifest() for region in self.regions],
        }


@dataclass(frozen=True)
class AtomicDetail:
    region_id: str
    category: str
    state: str
    value: str
    location: str

    def canonical_text(self) -> str:
        return (
            f"region={self.region_id}; category={self.category}; state={self.state}; "
            f"value={self.value}; location={self.location}"
        )

    def manifest(self) -> dict[str, str]:
        return {
            "region_id": self.region_id,
            "category": self.category,
            "state": self.state,
            "value": self.value,
            "location": self.location,
        }


@dataclass(frozen=True)
class JointWorld:
    assignments: tuple[AtomicDetail, ...]

    def canonical_text(self) -> str:
        return " || ".join(item.canonical_text() for item in self.assignments)

    def state_signature(self) -> tuple[tuple[str, str, str], ...]:
        return tuple(
            (item.region_id, item.category, item.state) for item in self.assignments
        )

    def manifest(self) -> list[dict[str, str]]:
        return [item.manifest() for item in self.assignments]


@dataclass(frozen=True)
class RewriteRequest:
    source: SourceSpec
    observation: Observation
    world: JointWorld
    seed: int


def ordered_world(world: JointWorld, regions: Sequence[RegionSpec]) -> JointWorld:
    by_region = {item.region_id: item for item in world.assignments}
    expected = {region.region_id: region for region in regions}
    if len(by_region) != len(world.assignments):
        raise ValueError("joint draw must not repeat a region assignment")
    if set(by_region) != set(expected):
        raise ValueError("joint draw must contain exactly one assignment per region")
    ordered = tuple(by_region[region.region_id] for region in regions)
    for item in ordered:
        if item.category != expected[item.region_id].category:
            raise ValueError(
                f"joint draw category mismatch for region {item.region_id}: "
                f"{item.category} != {expected[item.region_id].category}"
            )
        if not all((item.state.strip(), item.value.strip(), item.location.strip())):
            raise ValueError("joint draw fields must be non-empty")
    return JointWorld(ordered)
