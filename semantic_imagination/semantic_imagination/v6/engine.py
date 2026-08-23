from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TypeVar

from .clustering import cluster_joint_worlds, wilson_interval
from .config import V6Config
from .contracts import (
    BackendResult,
    JointWorld,
    Observation,
    QRI_V6,
    RewriteRequest,
    SourceSpec,
    ordered_world,
)
from .interfaces import RewriteBackend, SemanticEncoder, VLMBackend


T = TypeVar("T")


class BackendRequestError(RuntimeError):
    pass


@dataclass(frozen=True)
class PipelineResult:
    record: dict[str, Any]
    path: Path
    cached: bool


def _json_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _derived_seed(base: int, source_key: str, phase: str, index: int) -> int:
    digest = hashlib.sha256(
        f"{base}:{source_key}:{phase}:{index}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


class _Telemetry:
    def __init__(self) -> None:
        self.operations: Counter[str] = Counter()
        self.attempts: Counter[str] = Counter()
        self.successes: Counter[str] = Counter()
        self.elapsed: defaultdict[str, float] = defaultdict(float)
        self.usage: defaultdict[str, defaultdict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )

    def operation(self, phase: str) -> None:
        self.operations[phase] += 1

    def attempt(self, phase: str, elapsed: float) -> None:
        self.attempts[phase] += 1
        self.elapsed[phase] += elapsed

    def success(self, phase: str, result: BackendResult[Any]) -> None:
        self.successes[phase] += 1
        for name, value in result.usage.items():
            self.usage[phase][str(name)] += float(value)

    def manifest(self) -> dict[str, Any]:
        phases = sorted(self.attempts)
        return {
            "request_operation_count": sum(self.operations.values()),
            "request_attempt_count": sum(self.attempts.values()),
            "request_success_count": sum(self.successes.values()),
            "request_retry_count": sum(
                self.attempts[phase] - self.operations[phase] for phase in phases
            ),
            "request_failure_count": sum(
                self.operations[phase] - self.successes[phase] for phase in phases
            ),
            "by_phase": {
                phase: {
                    "operations": self.operations[phase],
                    "attempts": self.attempts[phase],
                    "successes": self.successes[phase],
                    "retries": self.attempts[phase] - self.operations[phase],
                    "failures": self.operations[phase] - self.successes[phase],
                    "elapsed_seconds": self.elapsed[phase],
                    "usage": dict(sorted(self.usage[phase].items())),
                }
                for phase in phases
            },
        }


def _run_signature_payload(
    config: V6Config,
    vlm: VLMBackend,
    encoder: SemanticEncoder,
    rewriter: RewriteBackend,
) -> dict[str, Any]:
    return {
        "plugin_version": QRI_V6,
        "algorithm": config.algorithm_contract(),
        "backends": {
            "vlm": vlm.descriptor.manifest(),
            "semantic_encoder": encoder.descriptor.manifest(),
            "llm_rewriter": rewriter.descriptor.manifest(),
        },
    }


class V6Engine:
    def __init__(
        self,
        config: V6Config,
        *,
        vlm: VLMBackend,
        encoder: SemanticEncoder,
        rewriter: RewriteBackend,
    ) -> None:
        self.config = config.validate()
        self.vlm = vlm
        self.encoder = encoder
        self.rewriter = rewriter

    def run_signature(self) -> dict[str, Any]:
        payload = _run_signature_payload(
            self.config, self.vlm, self.encoder, self.rewriter
        )
        return {"payload": payload, "sha256": _json_hash(payload)}

    def sampling_contract(
        self,
        source: SourceSpec,
        *,
        source_image_sha256: str | None = None,
    ) -> dict[str, Any]:
        source_image_sha256 = source_image_sha256 or _file_sha256(source.image)
        return {
            **self.run_signature()["payload"],
            "source": {
                "source_key": source.source_key,
                "image_sha256": source_image_sha256,
                "modality": source.modality,
                "regions": [region.manifest() for region in source.regions],
            },
        }

    def _invoke(
        self,
        phase: str,
        call: Callable[[], BackendResult[T]],
        telemetry: _Telemetry,
    ) -> BackendResult[T]:
        telemetry.operation(phase)
        last_error: Exception | None = None
        for _ in range(self.config.request_max_attempts):
            started = time.perf_counter()
            try:
                result = call()
            except Exception as error:
                last_error = error
                telemetry.attempt(phase, time.perf_counter() - started)
                continue
            telemetry.attempt(phase, time.perf_counter() - started)
            telemetry.success(phase, result)
            return result
        assert last_error is not None
        raise BackendRequestError(
            f"{phase} failed after {self.config.request_max_attempts} attempts: "
            f"{type(last_error).__name__}: {last_error}"
        ) from last_error

    @staticmethod
    def _observation(
        result: BackendResult[Observation], source: SourceSpec
    ) -> BackendResult[Observation]:
        observation = result.value
        if not observation.caption.strip():
            raise ValueError("VLM observation caption must be non-empty")
        observed_ids = [region.region_id for region in observation.regions]
        expected_ids = [region.region_id for region in source.regions]
        if observed_ids != expected_ids:
            raise ValueError("VLM observation regions must match source region order")
        return result

    @staticmethod
    def _joint_draw(
        result: BackendResult[JointWorld], source: SourceSpec
    ) -> BackendResult[JointWorld]:
        return BackendResult(
            ordered_world(result.value, source.regions),
            usage=result.usage,
        )

    def build_manifest(
        self,
        source: SourceSpec,
        *,
        source_image_sha256: str | None = None,
    ) -> dict[str, Any]:
        source_image_sha256 = source_image_sha256 or _file_sha256(source.image)
        signature = self.run_signature()
        sampling_contract = self.sampling_contract(
            source, source_image_sha256=source_image_sha256
        )
        telemetry = _Telemetry()
        observation_result = self._invoke(
            "vlm_observation",
            lambda: self._observation(self.vlm.observe(source), source),
            telemetry,
        )
        observation = observation_result.value

        samples: list[dict[str, Any]] = []
        valid_worlds: list[JointWorld] = []
        valid_sample_indices: list[int] = []
        for sample_index in range(self.config.joint_sample_count):
            seed = _derived_seed(
                self.config.seed, source.source_key, "joint-vlm-draw", sample_index
            )
            try:
                result = self._invoke(
                    "vlm_joint_sample",
                    lambda seed=seed: self._joint_draw(
                        self.vlm.sample_joint_world(source, observation, seed), source
                    ),
                    telemetry,
                )
            except BackendRequestError as error:
                samples.append(
                    {
                        "sample_index": sample_index,
                        "seed": seed,
                        "status": "request_failed",
                        "error": str(error),
                    }
                )
                continue
            valid_sample_indices.append(sample_index)
            valid_worlds.append(result.value)
            samples.append(
                {
                    "sample_index": sample_index,
                    "seed": seed,
                    "status": "valid",
                    "assignments": result.value.manifest(),
                }
            )
        if not valid_worlds:
            raise BackendRequestError("all qri-v6 joint VLM draws failed")

        encoded = self._invoke(
            "semantic_encoding",
            lambda: self.encoder.encode(
                [world.canonical_text() for world in valid_worlds]
            ),
            telemetry,
        )
        clusters = cluster_joint_worlds(
            valid_worlds,
            encoded.value,
            self.config.similarity_threshold,
        )
        ranked = sorted(
            clusters,
            key=lambda cluster: (
                -len(cluster.member_indices),
                valid_worlds[cluster.representative_index].canonical_text(),
            ),
        )[: self.config.max_worlds]
        retained_count = sum(len(cluster.member_indices) for cluster in ranked)
        worlds = []
        for cluster_id, cluster in enumerate(ranked):
            representative = valid_worlds[cluster.representative_index]
            rewrite_seed = _derived_seed(
                self.config.seed, source.source_key, "llm-rewrite", cluster_id
            )
            rewritten = self._invoke(
                "llm_rewrite",
                lambda representative=representative,
                rewrite_seed=rewrite_seed: self.rewriter.rewrite(
                    RewriteRequest(source, observation, representative, rewrite_seed)
                ),
                telemetry,
            )
            caption = str(rewritten.value).strip()
            if not caption:
                raise ValueError("LLM rewrite caption must be non-empty")
            member_sample_indices = [
                valid_sample_indices[index] for index in cluster.member_indices
            ]
            count = len(cluster.member_indices)
            worlds.append(
                {
                    "world_id": f"w{cluster_id:02d}",
                    "representative_sample_index": valid_sample_indices[
                        cluster.representative_index
                    ],
                    "member_sample_indices": member_sample_indices,
                    "assignments": representative.manifest(),
                    "caption": caption,
                    "sample_count": count,
                    "empirical_mass": count / self.config.joint_sample_count,
                    "valid_weight": count / len(valid_worlds),
                    "selected_weight": count / retained_count,
                    "valid_weight_interval_95": wilson_interval(
                        count, len(valid_worlds)
                    ),
                }
            )

        cluster_membership = {
            sample_index: cluster_id
            for cluster_id, cluster in enumerate(ranked)
            for sample_index in (
                valid_sample_indices[index] for index in cluster.member_indices
            )
        }
        for sample in samples:
            if sample["sample_index"] in cluster_membership:
                sample["retained_cluster_id"] = cluster_membership[
                    sample["sample_index"]
                ]

        valid_count = len(valid_worlds)
        return {
            "schema_version": 6,
            "plugin_version": QRI_V6,
            "status": "complete",
            "source_key": source.source_key,
            "image": str(source.image),
            "source_image_sha256": source_image_sha256,
            "run_signature": signature,
            "sampling_contract": sampling_contract,
            "sampling_contract_sha256": _json_hash(sampling_contract),
            "observation": observation.manifest(),
            "samples": samples,
            "worlds": worlds,
            "sampling_diagnostics": {
                "scheduled": self.config.joint_sample_count,
                "valid": valid_count,
                "request_failed": self.config.joint_sample_count - valid_count,
                "valid_mass": valid_count / self.config.joint_sample_count,
                "cluster_count": len(clusters),
                "retained_world_count": len(worlds),
                "retained_valid_mass": retained_count / valid_count,
                "retained_empirical_mass": retained_count
                / self.config.joint_sample_count,
            },
            "telemetry": telemetry.manifest(),
        }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class V6Pipeline:
    def __init__(
        self,
        config: V6Config,
        *,
        vlm: VLMBackend,
        encoder: SemanticEncoder,
        rewriter: RewriteBackend,
    ) -> None:
        self.config = config.validate()
        self.engine = V6Engine(
            self.config,
            vlm=vlm,
            encoder=encoder,
            rewriter=rewriter,
        )

    def record_path(self, source: SourceSpec) -> Path:
        key = Path(source.source_key)
        return self.config.output_root / "metadata" / key.parent / f"{key.stem}.json"

    def run(self, source: SourceSpec, *, overwrite: bool = False) -> PipelineResult:
        path = self.record_path(source)
        source_hash = _file_sha256(source.image)
        signature = self.engine.run_signature()
        sampling_contract_sha256 = _json_hash(
            self.engine.sampling_contract(source, source_image_sha256=source_hash)
        )
        if path.is_file() and not overwrite:
            cached = json.loads(path.read_text(encoding="utf-8"))
            if (
                cached.get("status") == "complete"
                and cached.get("plugin_version") == QRI_V6
                and cached.get("source_image_sha256") == source_hash
                and cached.get("run_signature") == signature
                and cached.get("sampling_contract_sha256") == sampling_contract_sha256
            ):
                return PipelineResult(copy.deepcopy(cached), path, True)
        record = self.engine.build_manifest(source, source_image_sha256=source_hash)
        _atomic_json(path, record)
        return PipelineResult(record, path, False)
