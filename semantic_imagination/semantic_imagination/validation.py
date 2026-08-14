from __future__ import annotations

import re
from collections.abc import Mapping

from .schema import AtomicHypothesis, ValidationIssue, ValidationResult
from .taxonomy import (
    CATEGORY_STATES,
    SENTINEL_STATES,
    normalize_symbol,
    state_value_compatible,
)


_ATOM = re.compile(
    r"\s*ATOM\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\.?\s*",
    re.IGNORECASE,
)
_FORBIDDEN_COMPOUND = re.compile(r"\b(and|or)\b|[,;|]", re.IGNORECASE)


def validate_atomic_response(
    raw: str,
    target_category: str,
    observed: str = "",
    *,
    category_states: Mapping[str, frozenset[str]] = CATEGORY_STATES,
) -> ValidationResult:
    match = _ATOM.fullmatch(str(raw).strip())
    if not match:
        return ValidationResult(
            raw=str(raw),
            issues=(ValidationIssue("invalid_format", "expected one ATOM record"),),
        )

    category, state, value, location = (part.strip().casefold() for part in match.groups())
    category = normalize_symbol(category)
    state = normalize_symbol(state)
    target = normalize_symbol(target_category)
    issues: list[ValidationIssue] = []
    if target not in category_states:
        issues.append(ValidationIssue("invalid_target_category", target))
    if category != target:
        issues.append(ValidationIssue("category_mismatch", f"{category} != {target}"))
    elif state not in category_states[target]:
        issues.append(ValidationIssue("invalid_state", state))
    elif not state_value_compatible(state, value):
        issues.append(ValidationIssue("state_value_mismatch", f"{state} -> {value}"))
    if not value.strip() or normalize_symbol(value) in {"unknown", "unclear", "not_visible", "not_sure"}:
        issues.append(ValidationIssue("empty_or_unknown_value", value))
    if not location.strip():
        issues.append(ValidationIssue("empty_location", location))
    if _FORBIDDEN_COMPOUND.search(value) or _FORBIDDEN_COMPOUND.search(location):
        issues.append(ValidationIssue("compound_field", f"{value} | {location}"))
    value_tokens = set(re.findall(r"[a-z0-9]+", value.casefold()))
    observed_tokens = set(re.findall(r"[a-z0-9]+", observed.casefold()))
    if observed_tokens and value_tokens and value_tokens <= observed_tokens:
        issues.append(ValidationIssue("repeats_observation", value))

    hypothesis = None
    if not issues:
        if state in SENTINEL_STATES:
            value = state
        if state == "no_additional_detail":
            location = "none"
        hypothesis = AtomicHypothesis(category, state, value, location)
    return ValidationResult(raw=str(raw), hypothesis=hypothesis, issues=tuple(issues))
