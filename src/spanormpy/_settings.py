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
