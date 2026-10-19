"""Package-wide settings (replaces R's ``options(SpaNorm.*)``)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Settings:
    """Mutable package settings.

    Attributes
    ----------
    polish_mem_budget
        Bytes available to ONE polish worker for the batched Newton's working set
        (R ``options(SpaNorm.polish.mem.budget)``, default ``1e9``).
    """

    polish_mem_budget: float = 1e9


settings = Settings()


def _match_arg(value, choices, name: str) -> str:
    """R ``match.arg`` for a single string (exact match only)."""
    if not isinstance(value, str) or value not in choices:
        raise ValueError(
            f"'{name}' should be one of: " + ", ".join(f"'{c}'" for c in choices)
        )
    return value
