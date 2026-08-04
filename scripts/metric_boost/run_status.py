"""Shared terminal-status semantics for experiment schedulers."""

from __future__ import annotations

import signal


TERMINAL_STATUSES = frozenset(
    {"succeeded", "failed", "stopped_by_user", "cancelled", "blocked", "skipped"}
)
_USER_STOP_RETURN_CODES = frozenset(
    {-signal.SIGINT, -signal.SIGTERM, 128 + signal.SIGINT, 128 + signal.SIGTERM}
)


def classify_terminal_status(
    return_code: int,
    has_metrics: bool,
    *,
    user_stop_requested: bool = False,
) -> str:
    """Classify a finished trainer without treating an authorized stop as failure."""
    if return_code == 0 and has_metrics:
        return "succeeded"
    if user_stop_requested and return_code in _USER_STOP_RETURN_CODES:
        return "stopped_by_user"
    return "failed"
