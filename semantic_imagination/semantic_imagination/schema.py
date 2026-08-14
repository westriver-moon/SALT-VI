from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AtomicHypothesis:
    category: str
    state: str
    value: str
    location: str

    def to_text(self) -> str:
        return (
            f"category={self.category}; state={self.state}; "
            f"value={self.value}; location={self.location}"
        )


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str


@dataclass(frozen=True)
class ValidationResult:
    raw: str
    hypothesis: AtomicHypothesis | None = None
    issues: tuple[ValidationIssue, ...] = field(default_factory=tuple)

    @property
    def valid(self) -> bool:
        return self.hypothesis is not None and not self.issues
