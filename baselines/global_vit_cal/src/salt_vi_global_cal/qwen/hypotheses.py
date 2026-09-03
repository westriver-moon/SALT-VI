from __future__ import annotations

import hashlib
import math
import re
from dataclasses import asdict, dataclass
from typing import Protocol, Sequence

from PIL import Image

from .schema import Region
from .zoom import swin_roi_board


@dataclass(frozen=True)
class Atom:
    category: str
    state: str
    value: str
    location: str

    def text(self) -> str:
        return f"ATOM | {self.category} | {self.state} | {self.value} | {self.location}"


@dataclass
class AtomicSample:
    seed: int
    status: str
    text: str | None = None
    cluster_id: int | None = None
    error: str | None = None


@dataclass
class HypothesisCluster:
    cluster_id: int
    category: str
    state: str
    representative: str
    sample_count: int
    empirical_mass: float
    weight: float
    weight_interval_95: tuple[float, float]
    member_indices: list[int]


@dataclass
class RegionHypotheses:
    region_id: str
    category: str
    scheduled_sample_count: int
    valid_sample_count: int
    clusters: list[HypothesisCluster]
    samples: list[AtomicSample]

    @property
    def valid_mass(self) -> float:
        return self.valid_sample_count / max(1, self.scheduled_sample_count)

    def to_dict(self) -> dict[str, object]:
        return {
            "region_id": self.region_id,
            "category": self.category,
            "scheduled_sample_count": self.scheduled_sample_count,
            "valid_sample_count": self.valid_sample_count,
            "valid_mass": self.valid_mass,
            "clusters": [asdict(cluster) for cluster in self.clusters],
            "samples": [asdict(sample) for sample in self.samples],
        }


class AtomSampler(Protocol):
    model_id: str

    def sample_atomic(
        self,
        full_swin: Image.Image,
        roi_board: Image.Image,
        region: Region,
        *,
        modality: str,
        observed: str,
        system_prompt: str,
        seed: int,
        temperature: float,
        thinking: bool,
    ) -> str: ...


_PROHIBITED = re.compile(
    r"\b(probability|confidence|confident|score|likelihood|percent|percentage)\b",
    re.IGNORECASE,
)


def atomic_system_prompt(region: Region, modality: str) -> str:
    color_rule = (
        "Use only colors visible in the SwinIR image."
        if modality.lower() == "rgb"
        else "This is infrared input; do not invent visible-spectrum colors."
    )
    allowed = ", ".join(sorted(region.allowed_states))
    return (
        "Perform one independent semantic imagination draw for a single ROI. "
        f"Target category: {region.category}. Allowed states: {allowed}. "
        "Return exactly: ATOM | category | state | short value | location. "
        "Use one allowed state and one concrete value/location. For absent or "
        "no_additional_detail use none for both. Never return prose, multiple "
        "details, a second candidate, a score, probability or confidence. "
        f"{color_rule} No reasoning."
    )


def parse_atom(
    text: str,
    *,
    expected_category: str,
    allowed_states: Sequence[str],
) -> Atom:
    raw = str(text).strip()
    if not raw or "\n" in raw or ";" in raw or _PROHIBITED.search(raw):
        raise ValueError("atomic response contains prose, multiple fields or confidence")
    parts = [part.strip() for part in raw.split("|")]
    if len(parts) != 5 or parts[0].casefold() != "atom":
        raise ValueError("atomic response must contain exactly five pipe fields")
    category, state, value, location = parts[1:]
    if category != expected_category:
        raise ValueError("atomic category does not match the target ROI")
    allowed_lookup = {item.casefold(): item for item in allowed_states}
    if state.casefold() not in allowed_lookup:
        raise ValueError("atomic state is not allowed for this category")
    if not value or not location or any("|" in field for field in parts[1:]):
        raise ValueError("atomic value and location are required")
    state = allowed_lookup[state.casefold()]
    if state.casefold() in {"absent", "no_additional_detail"} and (
        value.casefold() != "none" or location.casefold() != "none"
    ):
        raise ValueError("empty states require none value and location")
    return Atom(category=category, state=state, value=value, location=location)


def hashed_embeddings(atoms: Sequence[Atom], dimensions: int = 128) -> list[list[float]]:
    vectors: list[list[float]] = []
    for atom in atoms:
        vector = [0.0] * dimensions
        source = f"{atom.value} {atom.location}".casefold()
        for token in re.findall(r"[a-z0-9_]+", source):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % dimensions
            vector[index] += 1.0
        if not any(vector):
            vector[0] = 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        vectors.append([value / norm for value in vector])
    return vectors


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _complete_link_clusters(
    vectors: Sequence[Sequence[float]], threshold: float
) -> list[list[int]]:
    clusters: list[list[int]] = []
    for index, vector in enumerate(vectors):
        candidates = []
        for cluster_index, members in enumerate(clusters):
            minimum = min(_cosine(vector, vectors[member]) for member in members)
            if minimum >= threshold:
                candidates.append((minimum, -cluster_index, cluster_index))
        if candidates:
            clusters[max(candidates)[2]].append(index)
        else:
            clusters.append([index])
    return clusters


def _medoid(members: Sequence[int], vectors: Sequence[Sequence[float]]) -> int:
    return max(
        members,
        key=lambda index: (
            sum(_cosine(vectors[index], vectors[other]) for other in members),
            -index,
        ),
    )


def _wilson_interval(count: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    z = 1.959963984540054
    proportion = count / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    spread = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    ) / denominator
    return max(0.0, center - spread), min(1.0, center + spread)


def cluster_atoms(
    atoms: Sequence[Atom],
    *,
    scheduled_count: int,
    similarity_threshold: float,
) -> tuple[list[HypothesisCluster], list[int]]:
    if not atoms:
        raise ValueError("no valid atomic hypotheses remain")
    vectors = hashed_embeddings(atoms)
    members_by_cluster = _complete_link_clusters(vectors, similarity_threshold)
    assignments = [-1] * len(atoms)
    clusters = []
    valid = len(atoms)
    for cluster_id, members in enumerate(members_by_cluster):
        for member in members:
            assignments[member] = cluster_id
        representative = atoms[_medoid(members, vectors)]
        count = len(members)
        clusters.append(
            HypothesisCluster(
                cluster_id=cluster_id,
                category=representative.category,
                state=representative.state,
                representative=representative.text(),
                sample_count=count,
                empirical_mass=count / max(1, scheduled_count),
                weight=count / valid,
                weight_interval_95=_wilson_interval(count, valid),
                member_indices=list(members),
            )
        )
    return clusters, assignments


def sample_region_hypotheses(
    backend: AtomSampler,
    swin: Image.Image,
    region: Region,
    *,
    modality: str,
    observed: str,
    sample_count: int = 8,
    seed: int = 0,
    similarity_threshold: float = 0.85,
    max_attempts: int = 4,
    board_size_px: int = 512,
) -> RegionHypotheses:
    region.validate()
    if sample_count < 1 or max_attempts < 1:
        raise ValueError("sample_count and max_attempts must be positive")
    board = swin_roi_board(swin, region, size_px=board_size_px)
    prompt = atomic_system_prompt(region, modality)
    atoms: list[Atom] = []
    samples: list[AtomicSample] = []
    valid_sample_indices: list[int] = []
    for sample_index in range(sample_count):
        last_error = None
        accepted = None
        accepted_seed = None
        for attempt in range(max_attempts):
            draw_seed = (int(seed) + sample_index * 104729 + attempt) % (2**31)
            try:
                text = backend.sample_atomic(
                    swin,
                    board,
                    region,
                    modality=modality,
                    observed=observed,
                    system_prompt=prompt,
                    seed=draw_seed,
                    temperature=0.75,
                    thinking=False,
                )
                atom = parse_atom(
                    text,
                    expected_category=region.category,
                    allowed_states=region.allowed_states,
                )
                accepted = atom
                accepted_seed = draw_seed
                break
            except (ValueError, RuntimeError) as error:
                last_error = f"{type(error).__name__}: {error}"
        if accepted is None:
            samples.append(
                AtomicSample(
                    seed=(int(seed) + sample_index * 104729) % (2**31),
                    status="excluded_invalid",
                    error=last_error,
                )
            )
        else:
            valid_sample_indices.append(len(samples))
            atoms.append(accepted)
            samples.append(
                AtomicSample(seed=int(accepted_seed), status="valid", text=accepted.text())
            )
    clusters, assignments = cluster_atoms(
        atoms,
        scheduled_count=sample_count,
        similarity_threshold=similarity_threshold,
    )
    for sample_index, cluster_id in zip(valid_sample_indices, assignments):
        samples[sample_index].cluster_id = cluster_id
    return RegionHypotheses(
        region_id=region.region_id,
        category=region.category,
        scheduled_sample_count=sample_count,
        valid_sample_count=len(atoms),
        clusters=clusters,
        samples=samples,
    )


def build_regional_hypotheses(
    backend: AtomSampler,
    swin: Image.Image,
    regions: Sequence[Region],
    *,
    modality: str,
    observed: str,
    atomic_sample_count: int = 8,
    seed: int = 0,
    similarity_threshold: float = 0.85,
    max_attempts: int = 4,
    board_size_px: int = 512,
) -> list[RegionHypotheses]:
    if not regions:
        raise ValueError("at least one selected ROI is required")
    return [
        sample_region_hypotheses(
            backend,
            swin,
            region,
            modality=modality,
            observed=observed,
            sample_count=atomic_sample_count,
            seed=(int(seed) + index * 1000003) % (2**31),
            similarity_threshold=similarity_threshold,
            max_attempts=max_attempts,
            board_size_px=board_size_px,
        )
        for index, region in enumerate(regions)
    ]
