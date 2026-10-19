"""The ``SpaNormFit`` container (R class ``SpaNormFit``, ``R/AllClasses.R``)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SpaNormFit:
    """A fitted SpaNorm model. Matrices keep R's orientation (alpha: genes x p, W: cells x p)."""

    ngenes: int
    ncells: int
    gene_model: str
    df_tps: np.ndarray
    sample_p: float
    lambda_a: float | np.ndarray
    batch: np.ndarray | None
    W: np.ndarray
    alpha: np.ndarray
    gmean: np.ndarray
    psi: np.ndarray
    wtype: np.ndarray
    loglik: np.ndarray
    sampling: np.ndarray
    polish: dict = field(default_factory=dict)
