"""Covariate surface plots (R ``plotCovariate``, ``R/plotCovariates.R``)."""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes

from .._adjust import normalise_meanbatch, normalise_meanbio, normalise_meanls
from .._fit import get_spanorm_fit, load_fit, subset_genes
from ._spatial import _draw

_FUNS = {"biology": normalise_meanbio, "ls": normalise_meanls, "batch": normalise_meanbatch}


def covariate_values(adata, covariate="biology", genes=None, name="spanorm") -> np.ndarray:
    """log2 fitted mean of one covariate, spots x genes (R ``normaliseMeanBio/LS/Batch`` at scale 1).

    ``covariate`` is ``"biology"``, ``"ls"`` (library-size effect at the median library size)
    or ``"batch"``. ``genes`` is a list of ``var_names`` (default all).
    """
    if covariate not in _FUNS:
        raise ValueError("'covariate' must be one of: biology, ls, batch")
    if load_fit(adata, name, validate=False) is None:
        raise ValueError("SpaNorm fit not found in 'adata'")
    fit = get_spanorm_fit(adata) if name == "spanorm" else load_fit(adata, name)
    if covariate == "batch" and "batch" not in set(np.atleast_1d(fit.wtype).tolist()):
        raise ValueError("no batch covariates found in the SpaNorm fit")
    if genes is not None:
        idx = adata.var_names.get_indexer(list(genes))
        if np.any(idx < 0):
            raise ValueError("gene not found in 'adata.var_names'")
        fit = subset_genes(fit, idx)
    return _FUNS[covariate](None, 1, fit).T


def plot_covariate(adata, covariate="biology", color=None, name="spanorm", **kwargs) -> Axes:
    """Spatially plot the library-size, biology or batch effect modelled for a gene (R ``plotCovariate``).

    Parameters
    ----------
    adata
        Object holding a SpaNorm fit (see :func:`spanormpy.spanorm`).
    covariate
        ``"biology"`` (default), ``"ls"`` or ``"batch"``.
    color
        The gene (in ``adata.var_names``) to plot; required, as in R.
    name
        Name of the stored fit.
    **kwargs
        Passed to :func:`spanormpy.pl.plot_spatial` (``img``, ``crop``, ``ax``, ``show``, ...).
    """
    for bad in ("what", "layer", "basis"):
        if bad in kwargs:
            raise TypeError(f"'{bad}' is not accepted by plot_covariate")
    if color is None:
        raise ValueError("no genes selected for the plot")
    if not isinstance(color, str):
        raise TypeError("'color' must be a single gene name (str)")
    if color not in adata.var_names:
        raise ValueError("no genes selected for the plot")
    if (adata.var_names == color).sum() != 1:
        raise ValueError(f"gene '{color}' is duplicated in 'adata.var_names'")
    kwargs.setdefault("title", covariate)
    vals = covariate_values(adata, covariate, [color], name)[:, 0]
    return _draw(adata, vals, str(color), **kwargs)
