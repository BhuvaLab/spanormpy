"""GLM-based (SpaNorm) PCA (R ``SpaNormPCA``, ``R/mainSpaNormPCA.R``).

The technical (null) fit holds the library-size effects, batch effects and the gene mean.
Regressing it out of the counts leaves deviance or Pearson residuals, and PCA of the residuals
of the top spatially variable genes approximates GLM-PCA (Townes et al., Genome Biology 2019).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ._adjust import normalise_pearson, subset_fit_genes
from ._fit import _as_actual, get_spanorm_fit
from ._fitnb import calculate_mu
from ._io import get_counts, resolve_input
from ._nbdist import winsorise_psi
from .spanorm import _match_arg
from .svg import top_svgs

_RESIDUALS = ("deviance", "pearson")
_SOLVERS = ("exact", "randomized")
_BLOCK_ELEMENTS = 4_000_000  # genes x spots per residual block (residuals are per gene, so exact)


def _dense(Y) -> np.ndarray:
    return Y.toarray() if sp.issparse(Y) else np.asarray(Y)


def _deviance_block(Y, fit) -> np.ndarray:
    mu = calculate_mu(fit.gmean, fit.alpha, fit.W)
    psi = winsorise_psi(fit.psi, psi_max=getattr(fit, "psi_max", None))[:, None]
    Y = _dense(Y).astype(float)
    with np.errstate(all="ignore"):
        dev = Y * np.log(Y / mu)
        dev[np.isnan(dev)] = 0.0  # 0 * log(0) (R: is.nan)
        dev = dev - (Y + 1.0 / psi) * np.log((1.0 + Y * psi) / (1.0 + mu * psi))
        dev = np.maximum(dev, 0.0)  # correction for numerical errors
        return np.sign(Y - mu) * np.sqrt(2.0 * dev)


def deviance_residuals(Y, fit_technical) -> np.ndarray:
    """NB deviance residuals of ``Y`` (genes x spots) under a fit (R ``devianceResiduals``)."""
    return _blocked(_deviance_block, Y, fit_technical)


def _pearson_block(Y, fit) -> np.ndarray:
    return normalise_pearson(Y, 1, fit)


def _blocked(fun, Y, fit) -> np.ndarray:
    ng, nc = Y.shape
    step = max(1, _BLOCK_ELEMENTS // max(nc, 1))
    out = np.empty((ng, nc))
    for s in range(0, ng, step):
        sl = slice(s, min(s + step, ng))
        # the block keeps the whole fit's dispersion-winsorisation threshold
        out[sl] = fun(Y[sl], subset_fit_genes(fit, sl))
    return out


def get_residuals(Y, fit_technical, type: str = "deviance") -> np.ndarray:
    """Deviance or Pearson residuals of ``Y`` (genes x spots) (R ``getResiduals``).

    Pearson residuals are R's ``normalisePearson(Y, 1, fit)``: the baseline excludes the
    biology terms, which for a technical fit are absent anyway.
    """
    type = _match_arg(type, _RESIDUALS, "residuals")
    if type == "pearson":
        return _blocked(_pearson_block, Y, fit_technical)
    return deviance_residuals(Y, fit_technical)


def _randomized_svd(x: np.ndarray, k: int, n_iter: int = 7, oversample: int = 10, seed: int = 0):
    rng = np.random.default_rng(seed)
    q = x @ rng.standard_normal((x.shape[1], min(k + oversample, min(x.shape))))
    q, _ = np.linalg.qr(q)
    for _ in range(n_iter):
        z, _ = np.linalg.qr(x.T @ q)
        q, _ = np.linalg.qr(x @ z)
    u_b, s, vt = np.linalg.svd(q.T @ x, full_matrices=False)
    return q @ u_b[:, :k], s[:k], vt[:k]


def calculate_pca(emat: np.ndarray, ncomponents: int, svd_solver: str = "exact") -> dict:
    """PCA of the (genes x spots) residuals ``emat``, centred per gene (R ``calculatePCA``).

    Returns ``x`` (spots x components), ``rotation`` (genes x components), ``variance``
    (squared sdev) and ``percent_var`` (percentages of the variance of the computed components,
    as R's ``percentVar``, which sums over the returned components only).
    """
    x = np.asarray(emat, dtype=float).T
    x = x - x.mean(axis=0, keepdims=True)
    k = int(ncomponents)
    if svd_solver == "exact":
        u, s, vt = np.linalg.svd(x, full_matrices=False)
        u, s, vt = u[:, :k], s[:k], vt[:k]
    else:
        u, s, vt = _randomized_svd(x, k)
    variance = (s / np.sqrt(x.shape[0] - 1)) ** 2
    return {
        "x": u * s,
        "rotation": vt.T,
        "variance": variance,
        "percent_var": variance / variance.sum() * 100,
    }


def spanorm_pca(
    data,
    nsvgs: int = 3000,
    ncomponents: int = 50,
    svg_fdr: float = 1.0,
    residuals: str = "deviance",
    key_added: str = "X_spanorm_pca",
    svd_solver: str = "exact",
    n_jobs: int = 1,
    layer: str | None = None,
    table_key: str | None = None,
    copy: bool = False,
):
    """GLM-based (SpaNorm) PCA (R ``SpaNormPCA``).

    Deviance or Pearson residuals of the stored technical fit (``spanorm_null``, created by
    :func:`~spanormpy.spanorm_svg`) are computed for the top ``nsvgs`` SVGs (see
    :func:`~spanormpy.top_svgs`), centred per gene (not scaled), and decomposed.

    Parameters
    ----------
    data
        An ``AnnData`` or ``SpatialData`` on which ``spanorm`` and ``spanorm_svg`` have run.
    nsvgs, ncomponents, svg_fdr
        Number of SVGs used, number of components, and the SVG FDR threshold.
    residuals
        ``"deviance"`` (default) or ``"pearson"``.
    key_added
        Scores go to ``obsm[key_added]``; ``uns[key_added]`` holds ``variance`` (squared
        standard deviations), ``percent_var`` (R ``percentVar``, percent of the variance of the
        computed components), ``svgs`` (the genes used) and the settings; ``varm[key_added +
        "_loadings"]`` holds the rotation (R ``rotation``), zero for genes not used.
    svd_solver
        ``"exact"`` (``numpy.linalg.svd``, R ``ExactParam``) or ``"randomized"``
        (randomised SVD for large data; approximate, fixed seed).
    n_jobs
        Accepted for R's ``BPPARAM``; the decomposition uses BLAS threads, so it has no effect.
    layer, table_key
        Counts layer (default ``X``) and, for ``SpatialData``, the table.
    copy
        Return a modified copy instead of modifying ``data`` in place.

    Returns
    -------
    ``None``, or with ``copy=True`` a modified copy of the same container type.
    """
    adata, write_back = resolve_input(data, layer, "spatial", table_key)
    if not nsvgs > 0:
        raise ValueError("nsvgs > 0 is not TRUE")
    if not ncomponents > 0:
        raise ValueError("ncomponents > 0 is not TRUE")
    residuals = _match_arg(residuals, _RESIDUALS, "residuals")
    svd_solver = _match_arg(svd_solver, _SOLVERS, "svd_solver")
    if int(n_jobs) < 1:
        raise ValueError("'n_jobs' must be a positive integer")
    Y = get_counts(adata, layer)
    if not ncomponents <= adata.n_obs:
        raise ValueError("ncomponents <= ncol(spe) is not TRUE")

    get_spanorm_fit(adata)  # the full fit must exist and match the spots
    fit_technical = get_spanorm_fit(adata, null=True)
    top = top_svgs(adata, nsvgs, svg_fdr, layer=layer)
    if top["svg_fdr"].isna().any():
        # R: emat[svgs, ] with the "NA" row names topSVGs gives genes without an FDR
        raise ValueError("subscript out of bounds: some top SVGs have no FDR (NaN)")
    svgs = list(top.index)
    var_names = np.asarray(adata.var_names)
    rows = pd.Index(var_names).get_indexer(svgs)  # SVG order
    if ncomponents > min(adata.n_obs, len(rows)):
        raise ValueError(
            f"ncomponents ({ncomponents}) exceeds the number of SVGs used ({len(rows)}) "
            f"or spots ({adata.n_obs})"
        )

    emat = get_residuals(Y[rows], subset_fit_genes(fit_technical, rows), residuals)
    pca = calculate_pca(emat, int(ncomponents), svd_solver)

    if copy:
        adata = adata.copy()
    _as_actual(adata)
    adata.obsm[key_added] = pca["x"]
    loadings = np.zeros((adata.n_vars, int(ncomponents)))
    loadings[rows] = pca["rotation"]
    adata.varm[key_added + "_loadings"] = loadings
    adata.uns[key_added] = {
        "variance": pca["variance"],
        "percent_var": pca["percent_var"],
        "svgs": np.asarray(var_names[rows], dtype=str),
        "params": {
            "residuals": residuals,
            "nsvgs": len(rows),
            "ncomponents": int(ncomponents),
            "svg_fdr": float(svg_fdr),
            "svd_solver": svd_solver,
        },
    }
    result = write_back(adata, copy=copy)
    return result if copy else None


__all__ = ["deviance_residuals", "get_residuals", "spanorm_pca"]
