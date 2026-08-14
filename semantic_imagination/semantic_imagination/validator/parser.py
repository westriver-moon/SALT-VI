from __future__ import annotations

import re
from dataclasses import dataclass

from ..schema import ValidationIssue
from ..taxonomy import SENTINEL_STATES, normalize_symbol


_CANONICAL = re.compile(
    r"\s*category\s*=\s*([^;]+?)\s*;\s*"
    r"state\s*=\s*([^;]+?)\s*;\s*"
    r"value\s*=\s*([^;]+?)\s*;\s*"
    r"location\s*=\s*([^;]+?)\s*\.?\s*",
    re.IGNORECASE,
)

_STATE_ALIASES = {
    "no_detail": "no_additional_detail",
    "no_additional": "no_additional_detail",
    "no_additional_details": "no_additional_detail",
    "not_present": "absent",
}


@dataclass(frozen=True)
class ParsedAtomicResponse:
    category: str
    state: str
    value: str
    location: str
    repairs: tuple[str, ...] = ()


def _clean_field(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().casefold())


def _canonical_state(value: str) -> tuple[str, str | None]:
    state = normalize_symbol(value)
    canonical = _STATE_ALIASES.get(state, state)
    if canonical == state:
        return state, None
    return canonical, f"canonicalized_state:{state}->{canonical}"


def parse_atomic_response(
    raw: str,
) -> tuple[ParsedAtomicResponse | None, tuple[ValidationIssue, ...]]:
    """Parse one atomic record and repair only unambiguous surface defects."""

    text = str(raw).strip()
    canonical = _CANONICAL.fullmatch(text)
    if canonical:
        fields = list(canonical.groups())
        missing_location = False
    else:
        if text.endswith("."):
            text = text[:-1].rstrip()
        parts = [part.strip() for part in text.split("|")]
        if not parts or parts[0].casefold() != "atom" or len(parts) not in {4, 5}:
            return None, (
                ValidationIssue(
                    "invalid_format",
                    "expected ATOM | category | state | value | location",
                ),
            )
        fields = parts[1:]
        missing_location = len(fields) == 3

    category = normalize_symbol(fields[0])
    state, state_repair = _canonical_state(fields[1])
    value = _clean_field(fields[2])
    repairs: list[str] = []
    if state_repair:
        repairs.append(state_repair)

    if missing_location:
        if state not in SENTINEL_STATES:
            return None, (
                ValidationIssue(
                    "invalid_format",
                    "a positive state requires an explicit location field",
                ),
            )
        location = "none"
        repairs.append("inferred_sentinel_location:none")
    else:
        location = _clean_field(fields[3])

    return (
        ParsedAtomicResponse(category, state, value, location, tuple(repairs)),
        (),
    )
