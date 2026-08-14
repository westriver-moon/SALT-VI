from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Protocol

from .clustering import cluster_hypothesis_samples, wilson_interval
from .schema import ValidationResult
from .validator import build_retry_instruction


class ImaginationBackend(Protocol):
    model_id: str

    def observe(self, image: Path) -> str: ...

    def perturb(self, image: Path, seed: int) -> object: ...

    def imagine(self, image: object, observed: str, instruction: str, seed: int) -> str: ...

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...


AtomicValidator = Callable[[str, str, str], ValidationResult]


def _default_compose(observed: str, hypothesis: str) -> str:
    return " ".join(part.strip() for part in (observed, hypothesis) if part.strip())


def _validation_attempt(
    result: ValidationResult,
    seed: int,
    instruction: str,
    retry_feedback_applied: bool,
) -> dict:
    return {
        "seed": seed,
        "instruction": instruction,
        "retry_feedback_applied": retry_feedback_applied,
        "raw": result.raw,
        "valid": result.valid,
        "repairs": list(result.repairs),
        "issues": [
            {"code": issue.code, "message": issue.message} for issue in result.issues
        ],
    }


def _apply_stratified_weights(
    hypotheses: list[dict],
    samples: list[dict],
    valid_sample_indices: list[int],
) -> None:
    scheduled: dict[str, int] = {}
    valid: dict[str, int] = {}
    for sample in samples:
        category = str(sample.get("stratum", "__unstructured__"))
        scheduled[category] = scheduled.get(category, 0) + 1
        if sample["status"] == "valid":
            valid[category] = valid.get(category, 0) + 1
    total = len(samples)
    valid_total = len(valid_sample_indices)
    stratified = any("stratum" in sample for sample in samples)
    for hypothesis in hypotheses:
        category = str(hypothesis["category"])
        count = int(hypothesis["count"])
        if stratified:
            category_prior = scheduled.get(category, 0) / total
            effective_n = valid.get(category, 0)
            conditional = count / effective_n
            interval = wilson_interval(count, effective_n)
            hypothesis["category_weight"] = category_prior
            hypothesis["conditional_weight"] = conditional
            hypothesis["conditional_weight_interval_95"] = interval
            hypothesis["weight"] = category_prior * conditional
            hypothesis["weight_interval_95"] = {
                "method": "stratified-wilson-95-conditional-on-valid-samples",
                "lower": category_prior * float(interval["lower"]),
                "upper": category_prior * float(interval["upper"]),
            }
        else:
            hypothesis["weight"] = count / valid_total
            hypothesis["weight_interval_95"] = wilson_interval(count, valid_total)


def build_hypothesis_manifest(
    image: str | Path,
    source_key: str,
    backend: ImaginationBackend,
    instruction: str,
    sample_count: int,
    seed: int,
    similarity_threshold: float,
    compose: Callable[[str, str], str] = _default_compose,
    contract: Mapping[str, object] | None = None,
    cluster_linkage: str = "complete",
    sampling_strata: Sequence[str] | None = None,
    validator: AtomicValidator | None = None,
    max_attempts: int = 1,
    validation_failure_policy: str = "raise",
) -> dict:
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    if validation_failure_policy not in {"raise", "exclude"}:
        raise ValueError("validation_failure_policy must be 'raise' or 'exclude'")

    image = Path(image).expanduser().resolve()
    observed = str(backend.observe(image)).strip()
    rng = random.Random(seed)
    strata = tuple(
        str(value).strip() for value in (sampling_strata or ()) if str(value).strip()
    )
    samples: list[dict] = []
    valid_indices: list[int] = []
    valid_texts: list[str] = []
    for sample_index in range(sample_count):
        sample_seed = rng.randrange(2**31)
        perturbed = backend.perturb(image, sample_seed)
        stratum = strata[sample_index % len(strata)] if strata else None
        sampled_instruction = instruction
        if stratum is not None:
            sampled_instruction += (
                f"\nTarget category: {stratum}. The hypothesis must concern only this category."
            )
        sample: dict = {"seed": sample_seed, "status": "validation_failed"}
        if stratum is not None:
            sample["stratum"] = stratum
        attempts = []
        last_result: ValidationResult | None = None
        for attempt_index in range(max_attempts if validator else 1):
            attempt_seed = (sample_seed + attempt_index * 1000003) % (2**31)
            attempt_instruction = (
                sampled_instruction
                if last_result is None
                else build_retry_instruction(sampled_instruction, last_result)
            )
            raw = str(
                backend.imagine(perturbed, observed, attempt_instruction, attempt_seed)
            ).strip()
            if validator is None:
                if not raw:
                    raise ValueError("imagination samples must be non-empty")
                text = raw
                break
            result = validator(raw, stratum or "", observed)
            attempts.append(
                _validation_attempt(
                    result,
                    attempt_seed,
                    attempt_instruction,
                    retry_feedback_applied=attempt_index > 0,
                )
            )
            if result.valid:
                assert result.hypothesis is not None
                text = result.hypothesis.to_text()
                break
            last_result = result
        else:
            text = ""

        if attempts:
            sample["attempts"] = attempts
        if text:
            sample["status"] = "valid"
            sample["text"] = text
            valid_indices.append(sample_index)
            valid_texts.append(text)
        elif validation_failure_policy == "raise":
            raise ValueError(
                f"sample {sample_index} failed validation after {max_attempts} attempts"
            )
        samples.append(sample)

    if not valid_texts:
        raise ValueError("all imagination samples failed validation")
    vectors = backend.embed(valid_texts)
    hypotheses = cluster_hypothesis_samples(
        valid_texts, vectors, similarity_threshold, cluster_linkage
    )
    for hypothesis in hypotheses:
        hypothesis["representative_index"] = valid_indices[
            int(hypothesis["representative_index"])
        ]
        hypothesis["member_indices"] = [
            valid_indices[index] for index in hypothesis["member_indices"]
        ]
        hypothesis["caption"] = compose(observed, hypothesis["representative"])
        for member in hypothesis["member_indices"]:
            samples[member]["cluster_id"] = hypothesis["cluster_id"]
    _apply_stratified_weights(hypotheses, samples, valid_indices)

    category_diagnostics = {}
    for category in sorted({str(sample.get("stratum", "__unstructured__")) for sample in samples}):
        category_samples = [sample for sample in samples if sample.get("stratum", "__unstructured__") == category]
        valid_count = sum(sample["status"] == "valid" for sample in category_samples)
        category_diagnostics[category] = {
            "scheduled": len(category_samples),
            "valid": valid_count,
            "validation_failed": len(category_samples) - valid_count,
            "valid_rate": valid_count / len(category_samples),
        }
    issue_counts: dict[str, int] = {}
    for sample in samples:
        for attempt in sample.get("attempts", []):
            for issue in attempt["issues"]:
                code = str(issue["code"])
                issue_counts[code] = issue_counts.get(code, 0) + 1
    diagnostics = {
        "scheduled": len(samples),
        "valid": len(valid_indices),
        "validation_failed": len(samples) - len(valid_indices),
        "valid_rate": len(valid_indices) / len(samples),
        "attempts": sum(len(sample.get("attempts", [])) for sample in samples),
        "retries": sum(max(0, len(sample.get("attempts", [])) - 1) for sample in samples),
        "repaired": sum(
            bool(sample["attempts"][-1].get("repairs"))
            for sample in samples
            if sample["status"] == "valid" and sample.get("attempts")
        ),
        "validation_issue_counts": issue_counts,
        "active_abstentions": sum(
            sample.get("text", "").find("state=no_additional_detail") >= 0
            for sample in samples
            if sample["status"] == "valid"
        ),
        "by_category": category_diagnostics,
    }

    sampling_contract = {
        "schema_version": 2,
        "backend_id": str(getattr(backend, "model_id", type(backend).__name__)),
        "instruction": instruction,
        "sample_count": sample_count,
        "seed": int(seed),
        "similarity_threshold": float(similarity_threshold),
        "cluster_linkage": cluster_linkage,
        "sampling_strata": list(strata),
        "validation": {
            "enabled": validator is not None,
            "max_attempts": max_attempts,
            "failure_policy": validation_failure_policy,
            "retry_feedback": "validator_issue_codes" if validator is not None else "disabled",
        },
        "backend_contract": dict(contract or {}),
    }
    encoded = json.dumps(
        sampling_contract, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "schema_version": 2,
        "source_key": source_key,
        "image": str(image),
        "observed": observed,
        "sampling_contract": sampling_contract,
        "sampling_contract_sha256": hashlib.sha256(encoded).hexdigest(),
        "samples": samples,
        "hypotheses": hypotheses,
        "cluster_linkage": cluster_linkage,
        "sampling_diagnostics": diagnostics,
    }
