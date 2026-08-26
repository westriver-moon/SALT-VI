from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SourceItem:
    source_key: str
    image: Path
    identity: str
    camera: str
    modality: str
    split: str = "train"


@dataclass
class Candidate:
    state: str
    value: str
    evidence: str = ""
    evidence_source: str = "compatible_prior_only"

    def key(self) -> tuple[str, str]:
        return self.state, self.value


@dataclass(frozen=True)
class JointSample:
    assignments: dict[str, Candidate]
    origin: str

    def __post_init__(self):
        if self.origin not in {"coverage", "free"}:
            raise ValueError("joint sample origin must be coverage or free")


@dataclass
class Region:
    region_id: str
    category: str
    bbox_xyxy: tuple[int, int, int, int]
    mask: Any = field(repr=False)
    side: str | None = None
    mask_path: str | None = None
    mask_sha256: str | None = None
    u_swin: float = 0.0
    u_swin_normalized: float = 0.0
    u_blur: float = 0.0
    u_qwen: float = 0.0
    u_qwen_proposal: float = 0.0
    u_qwen_compatible: float = 0.0
    candidates: list[Candidate] = field(default_factory=list)
    critic: list[dict[str, Any]] = field(default_factory=list)

    def manifest(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("mask", None)
        payload["bbox_xyxy"] = list(self.bbox_xyxy)
        return payload


@dataclass
class Assignment:
    region_id: str
    category: str
    state: str
    value: str
    critic_label: str = "compatible_prior_only"
    critic_score: float = 0.0

    def key(self) -> tuple[str, str, str]:
        return self.region_id, self.state, self.value


@dataclass
class World:
    world_id: str
    assignments: list[Assignment]
    sample_count: int
    proposal_mass: float
    coverage_sample_count: int = 0
    free_sample_count: int = 0
    caption: str = ""
    seed: int = 0
    mask_path: str | None = None
    mask_sha256: str | None = None
    pasd_output: str | None = None
    pasd_output_sha256: str | None = None
    realizations: list[dict[str, Any]] = field(default_factory=list)
    output: str | None = None
    output_sha256: str | None = None
    output_bytes: int | None = None
    e_lr: float = 0.0
    e_id: float = 0.0
    e_edit: float = 0.0
    uniform_weight: float = 0.0
    proposal_weight: float = 0.0
    posterior_weight: float = 0.0

    def assignment_key(self) -> tuple[tuple[str, str, str], ...]:
        return tuple(sorted(assignment.key() for assignment in self.assignments))

    def manifest(self) -> dict[str, Any]:
        return asdict(self)


def fallback_world() -> World:
    return World(
        world_id="w00_swin_fallback",
        assignments=[],
        sample_count=1,
        proposal_mass=1.0,
        caption="no additional uncertain detail",
        uniform_weight=1.0,
        proposal_weight=1.0,
        posterior_weight=1.0,
    )
