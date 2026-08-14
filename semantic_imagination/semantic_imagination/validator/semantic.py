from __future__ import annotations

import re
from collections.abc import Mapping

from ..schema import AtomicHypothesis, ValidationIssue, ValidationResult
from ..taxonomy import (
    CATEGORY_STATES,
    COLOR_TOKENS,
    SENTINEL_STATES,
    STATE_VALUE_EVIDENCE,
    evidence_for,
    normalize_symbol,
)
from .parser import ParsedAtomicResponse, parse_atomic_response


_NO_DETAIL_ALIASES = frozenset(
    {
        "none",
        "no",
        "no_detail",
        "no_additional",
        "no_additional_detail",
        "no_additional_details",
        "no_additional_detail_visible",
        "no_additional_details_visible",
        "absent",
        "not_present",
        "not_visible",
        "unclear",
    }
)
_ABSENT_ALIASES = frozenset(
    {"none", "no", "absent", "not_present", "not_visible"}
)
_UNKNOWN_VALUES = frozenset(
    {"", "unknown", "unclear", "not_visible", "not_sure", "unsure"}
)
_LOCATION_TOKENS = frozenset(
    {
        "arm",
        "back",
        "chest",
        "face",
        "feet",
        "foot",
        "hand",
        "head",
        "hip",
        "left",
        "leg",
        "neck",
        "none",
        "pocket",
        "right",
        "shoulder",
        "side",
        "torso",
        "waist",
        "wrist",
    }
)
_GENERIC_QUALIFIERS = COLOR_TOKENS | _LOCATION_TOKENS | frozenset(
    {
        "big",
        "detail",
        "item",
        "large",
        "medium",
        "object",
        "possible",
        "small",
        "thick",
        "thin",
        "tiny",
        "visible",
    }
)


def _tokens(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", value.casefold()))


def _canonicalize_sentinel(
    parsed: ParsedAtomicResponse,
) -> tuple[str, str, list[str]]:
    value = parsed.value
    location = parsed.location
    repairs = list(parsed.repairs)
    normalized_value = normalize_symbol(value)
    aliases = (
        _NO_DETAIL_ALIASES
        if parsed.state == "no_additional_detail"
        else _ABSENT_ALIASES
    )
    value_tokens = _tokens(value)
    if normalized_value in aliases:
        if value != parsed.state:
            repairs.append(f"canonicalized_sentinel_value:{normalized_value}->{parsed.state}")
        value = parsed.state
    elif value_tokens and value_tokens <= _LOCATION_TOKENS:
        repairs.append(f"reclassified_sentinel_location_as_value:{value}->{parsed.state}")
        if location == "none":
            location = value
        value = parsed.state
    elif (
        parsed.state == "no_additional_detail"
        and value_tokens
        and value_tokens <= _GENERIC_QUALIFIERS
    ):
        repairs.append(f"canonicalized_generic_sentinel_value:{value}->{parsed.state}")
        value = parsed.state
    if parsed.state == "no_additional_detail" and location != "none":
        repairs.append(f"canonicalized_sentinel_location:{location}->none")
        location = "none"
    return value, location, repairs


def _specific_value_conflicts(category: str, state: str, value: str) -> list[str]:
    value_tokens = _tokens(value)
    expected = evidence_for(category, state)
    conflicts: list[str] = []
    for other_category, states in STATE_VALUE_EVIDENCE.items():
        for other_state, evidence in states.items():
            if (other_category, other_state) == (category, state):
                continue
            if other_state == "color_detail":
                continue
            matched = (evidence & value_tokens) - expected
            if matched:
                conflicts.append(
                    f"{other_category}.{other_state} via {','.join(sorted(matched))}"
                )
    return conflicts


def _semantic_issues(
    category: str,
    state: str,
    value: str,
    location: str,
    observed: str,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    normalized_value = normalize_symbol(value)
    if state in SENTINEL_STATES:
        aliases = _NO_DETAIL_ALIASES if state == "no_additional_detail" else _ABSENT_ALIASES
        if normalized_value not in aliases and normalized_value != state:
            issues.append(
                ValidationIssue(
                    "state_value_mismatch",
                    f"sentinel state {state} conflicts with positive value {value}",
                )
            )
        return issues

    if normalized_value in (_NO_DETAIL_ALIASES | _ABSENT_ALIASES | _UNKNOWN_VALUES):
        issues.append(
            ValidationIssue(
                "state_value_mismatch",
                f"positive state {state} cannot use sentinel value {value}",
            )
        )
        return issues

    value_tokens = _tokens(value)
    if state.startswith("other_") and not (value_tokens - _GENERIC_QUALIFIERS):
        issues.append(
            ValidationIssue(
                "underspecified_other_value",
                f"{state} requires a concrete object or detail noun, not only {value}",
            )
        )

    conflicts = _specific_value_conflicts(category, state, value)
    if conflicts:
        issues.append(
            ValidationIssue(
                "state_value_mismatch",
                f"{category}.{state} conflicts with {'; '.join(conflicts)}",
            )
        )

    expected = evidence_for(category, state)
    observed_tokens = _tokens(observed)
    location_tokens = _tokens(location) - {"none"}
    state_already_observed = bool(expected & observed_tokens)
    value_already_observed = bool(value_tokens) and value_tokens <= observed_tokens
    location_already_observed = not location_tokens or location_tokens <= observed_tokens
    if (
        observed_tokens
        and state_already_observed
        and value_already_observed
        and location_already_observed
    ):
        issues.append(ValidationIssue("repeats_observation", value))
    return issues


def validate_atomic_response(
    raw: str,
    target_category: str,
    observed: str = "",
    *,
    category_states: Mapping[str, frozenset[str]] = CATEGORY_STATES,
) -> ValidationResult:
    parsed, parse_issues = parse_atomic_response(raw)
    if parsed is None:
        return ValidationResult(raw=str(raw), issues=parse_issues)

    category = parsed.category
    state = parsed.state
    target = normalize_symbol(target_category)
    issues: list[ValidationIssue] = []
    if target not in category_states:
        issues.append(ValidationIssue("invalid_target_category", target))
    if category != target:
        issues.append(ValidationIssue("category_mismatch", f"{category} != {target}"))
    elif state not in category_states[target]:
        issues.append(ValidationIssue("invalid_state", state))

    value = parsed.value
    location = parsed.location
    repairs = list(parsed.repairs)
    if state in SENTINEL_STATES:
        value, location, repairs = _canonicalize_sentinel(parsed)
    elif normalize_symbol(value) in _UNKNOWN_VALUES:
        issues.append(ValidationIssue("empty_or_unknown_value", value))
    if not location.strip():
        issues.append(ValidationIssue("empty_location", location))

    if category == target and target in category_states and state in category_states[target]:
        issues.extend(_semantic_issues(category, state, value, location, observed))

    hypothesis = None
    if not issues:
        hypothesis = AtomicHypothesis(category, state, value, location)
    return ValidationResult(
        raw=str(raw),
        hypothesis=hypothesis,
        issues=tuple(issues),
        repairs=tuple(repairs),
    )
