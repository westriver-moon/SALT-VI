from __future__ import annotations

from ..schema import ValidationResult


_GUIDANCE = {
    "invalid_format": "Return exactly one five-field ATOM record in the requested order.",
    "invalid_target_category": "Use the supplied target category exactly.",
    "category_mismatch": "Replace the category with the supplied target category.",
    "invalid_state": "Choose one controlled state allowed for that target category.",
    "state_value_mismatch": (
        "Keep state and value semantically consistent. The value may be a short "
        "color, size, or appearance qualifier and need not repeat the state name."
    ),
    "empty_or_unknown_value": "Give a concrete short value or use a valid sentinel state.",
    "empty_location": "Give a short body or object location.",
    "underspecified_other_value": (
        "For an other_* state, name one concrete object or detail noun; a color or size alone is insufficient."
    ),
    "repeats_observation": "Propose a detail not already present in the observation.",
}


def build_retry_instruction(base_instruction: str, result: ValidationResult) -> str:
    codes = tuple(dict.fromkeys(issue.code for issue in result.issues))
    guidance = " ".join(_GUIDANCE.get(code, "Correct this validation failure.") for code in codes)
    return (
        f"{base_instruction}\n"
        f"Previous response failed validation. Failure codes: {', '.join(codes)}. "
        f"Correction: {guidance}"
    )
