"""Model-based spatially variable gene (SVG) calling (R ``SpaNormSVG``, ``R/mainSpaNormSVG.R``).

The SpaNorm fit used for normalisation is the full model. A nested model without the
biology splines (the technical, or null, fit) is fitted to the same spots, and a likelihood
ratio F-test per gene compares the two. When the full fit is polished
(:func:`~spanormpy.polish_spanorm`) the null is polished the same way, because the likelihood
ratio only compares nested models when both sides are estimated alike.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats

from ._backend import DEFAULT_WINSOR
from ._fit import SpaNormFit, _as_actual, get_spanorm_fit, is_polished, load_fit, store_fit
from ._fitnb import _message, _quiet, calculate_mu, fit_spanorm_nb
from ._io import get_counts, resolve_input
from ._nbdist import dnbinom, winsorise_psi
from ._polish_model import polish_psi_range
from .polish_spanorm import SVG_COLUMNS, _stored_unpolished, polish_spanorm_fit
from .spanorm import _BACKENDS, _match_arg

_TABLE_COLUMNS = ("svg_F", "svg_p", "svg_fdr")
_BLOCK_ELEMENTS = 4_000_000  # genes x spots per likelihood block (rowSums are per gene, so exact)


# ------------------------------------------------------------------ polished alike


def _settings(fit: SpaNormFit) -> dict:
    return fit.polish.get("settings", {}) if is_polished(fit) else {}


def _polished_alike(fit_spanorm: SpaNormFit, fit_technical: SpaNormFit) -> bool:
    """Are a full and a null fit polished alike (R ``.polishedAlike``)?

    True iff both are unpolished, or both are polished and agree on ``psi_method``, ``ls`` and
    ``cells`` (and under ``psi_method="profile"`` on ``psi_range``, a missing one being
    polish_nb's default): the settings that define the fitted objective. ``maxit`` and ``tol``
    only control how tightly it is converged and ``a1`` legitimately differs between the two
    fits, so neither is compared.
    """
    full_polished, null_polished = is_polished(fit_spanorm), is_polished(fit_technical)
    if full_polished != null_polished:
        return False
    if not full_polished:
        return True
    sf, sn = _settings(fit_spanorm), _settings(fit_technical)
    same = (
        sf.get("psi_method") == sn.get("psi_method")
        and sf.get("ls") == sn.get("ls")
        and sf.get("cells") == sn.get("cells")
    )
    if not same:
        return False
    return sf.get("psi_method") != "profile" or np.array_equal(
        polish_psi_range(sf), polish_psi_range(sn)
    )


def _polish_alike_diff(fit_spanorm: SpaNormFit, fit_technical: SpaNormFit) -> str:
    """What :func:`_polished_alike` found different, for an error or message."""
    full_polished, null_polished = is_polished(fit_spanorm), is_polished(fit_technical)
    if full_polished != null_polished:
        return (
            "the polish state differs (full: "
            f"{'' if full_polished else 'not '}polished, null: "
            f"{'' if null_polished else 'not '}polished)"
        )
    sf, sn = _settings(fit_spanorm), _settings(fit_technical)
    for field in ("psi_method", "ls", "cells"):
        if sf.get(field) != sn.get(field):
            return f"'{field}' differs (full: {sf.get(field)!r}, null: {sn.get(field)!r})"
    if sf.get("psi_method") == "profile" and not np.array_equal(
        polish_psi_range(sf), polish_psi_range(sn)
    ):
        return (
            f"'psi_range' differs (full: {polish_psi_range(sf).tolist()}, "
            f"null: {polish_psi_range(sn).tolist()})"
        )
    return "no difference found"  # unreachable when _polished_alike is False


# ------------------------------------------------------------------ technical fit


def fit_spanorm_technical(Y, fit: SpaNormFit, msgfun=_message, rng=None, **kwargs) -> SpaNormFit:
    """Fit the nested model without the biology terms (R ``fitSpaNormTechnical``).

    The null is fitted to the cells the full model was fitted to, with its dispersion subset
    redrawn. The library-size penalty is scaled by the number of cells as in the full fit.
    ``kwargs`` (e.g. ``backend``) go to :func:`spanormpy._fitnb.fit_spanorm_nb`.
    """
    ncells = Y.shape[1]
    if ncells != fit.ncells:
        raise ValueError(
            f"number of cells in data ({ncells}) differs from the SpaNorm fit ({fit.ncells})"
        )
    sampling = np.asarray(fit.sampling).astype(str)
    msgfun(f"{int(np.sum(sampling != 'all'))} cells/spots sampled to fit model")

    wtype = np.asarray(fit.wtype).astype(str)
    keep = wtype != "biology"  # technical covariates
    W = np.asarray(fit.W)[:, keep]
    wtype = wtype[keep]

    lam = np.atleast_1d(np.asarray(fit.lambda_a, dtype=float))
    lambda_vec = np.zeros(len(wtype))
    lambda_vec[wtype == "biology"] = lam[0]
    lambda_vec[wtype == "ls"] = lam[1] if len(lam) > 1 else lam[0]
    lambda_vec = lambda_vec[1:]

    if rng is None:
        rng = np.random.default_rng(0)
    res = fit_spanorm_nb(
        Y,
        W,
        sampling != "all",
        maxn_psi=int(np.sum(sampling == "dispersion")),
        lambda_a=lambda_vec * ncells,
        msgfun=msgfun,
        rng=rng,
        is_spanorm=True,
        **kwargs,
    )
    return SpaNormFit(
        ngenes=Y.shape[0],
        ncells=ncells,
        gene_model=fit.gene_model,
        df_tps=fit.df_tps,
        sample_p=fit.sample_p,
        lambda_a=fit.lambda_a,
        batch=fit.batch,
        W=W,
        alpha=res["alpha"],
        gmean=res["gmean"],
        psi=res["psi"],
        wtype=wtype,
        loglik=np.atleast_1d(res["loglik"]),
        sampling=res["sampling"],
        colnames=None if fit.colnames is None else np.asarray(fit.colnames)[keep],
    )


# ------------------------------------------------------------------ the test


def _p_adjust_fdr(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjustment (R ``p.adjust(method = "fdr")``; NaN are left out of n)."""
    p = np.asarray(p, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = ~np.isnan(p)
    n = int(ok.sum())
    if n == 0:
        return out
    pv = p[ok]
    o = np.argsort(-pv, kind="stable")
    ro = np.argsort(o, kind="stable")
    q = np.minimum.accumulate(n / (n - np.arange(n)) * pv[o])
    out[ok] = np.minimum(1.0, q)[ro]
    return out


def _fit_loglik(Y, fit: SpaNormFit) -> np.ndarray:
    """Per-gene NB log-likelihood of ``Y`` at the fit's (winsorised) mean and dispersion."""
    psi = winsorise_psi(fit.psi, DEFAULT_WINSOR, psi_max=getattr(fit, "psi_max", None))
    ng, nc = Y.shape
    step = max(1, _BLOCK_ELEMENTS // max(nc, 1))
    out = np.empty(ng)
    alpha, gmean, W = np.asarray(fit.alpha), np.asarray(fit.gmean), np.asarray(fit.W)
    for s in range(0, ng, step):
        sl = slice(s, min(s + step, ng))
        mu = calculate_mu(gmean[sl], alpha[sl], W)
        Yb = Y[sl].toarray() if sp.issparse(Y) else np.asarray(Y[sl])
        ll = dnbinom(Yb, mu, 1.0 / psi[sl, None], log=True)
        out[sl] = ll.detach().cpu().numpy().astype(float).sum(axis=1)
    return out


def svg_test(Y, fit_spanorm: SpaNormFit, fit_technical: SpaNormFit) -> pd.DataFrame:
    """Likelihood-ratio F-test of the full against the nested technical fit (R ``svgTest``).

    Returns a frame with ``svg_F`` (clamped at 0), ``svg_p`` and ``svg_fdr``; the unclamped
    statistic is in ``result.attrs["F_raw"]``.
    """
    if not (isinstance(Y, np.ndarray) or sp.issparse(Y)):
        raise TypeError("Y must be a matrix or Matrix object")
    if not isinstance(fit_spanorm, SpaNormFit):
        raise TypeError("fit_spanorm must be a SpaNormFit object")
    if not isinstance(fit_technical, SpaNormFit):
        raise TypeError("fit_technical must be a SpaNormFit object")
    if not _polished_alike(fit_spanorm, fit_technical):
        raise ValueError(
            "the full and null SpaNorm fits must be polished alike: "
            + _polish_alike_diff(fit_spanorm, fit_technical)
        )
    if len({Y.shape[0], fit_spanorm.ngenes, fit_technical.ngenes}) != 1:
        raise ValueError("number of genes differ between SpaNorm fits and/or data")
    if len({Y.shape[1], fit_spanorm.ncells, fit_technical.ncells}) != 1:
        raise ValueError("number of cells differ between SpaNorm fits and/or data")
    ncol_full, ncol_tech = np.shape(fit_spanorm.W)[1], np.shape(fit_technical.W)[1]
    names_full, names_tech = fit_spanorm.colnames, fit_technical.colnames
    # R: colnames(NULL) %in% colnames(NULL) is empty, so unnamed fits pass the name check
    names_ok = names_full is None or names_tech is None or set(names_tech) <= set(names_full)
    if not names_ok or ncol_tech == ncol_full:
        raise ValueError("technical model is not nested in the full model")

    loglik_full = _fit_loglik(Y, fit_spanorm)
    loglik_tech = _fit_loglik(Y, fit_technical)

    df1 = ncol_full - ncol_tech
    df2 = Y.shape[1] - ncol_full
    with np.errstate(invalid="ignore"):
        f_raw = 2 * (loglik_full - loglik_tech) / df1
        f_lrt = np.maximum(f_raw, 0)  # threshold to 0 due to convergence issues
    p_val = stats.f.sf(f_lrt, df1, df2)
    res = pd.DataFrame({"svg_F": f_lrt, "svg_p": p_val, "svg_fdr": _p_adjust_fdr(p_val)})
    res.attrs["F_raw"] = f_raw  # the unclamped statistic (additive; svg_F is unchanged)
    return res


# ------------------------------------------------------------------ paired null


def _svg_paired_null(
    fit_spanorm: SpaNormFit,
    fit_technical: SpaNormFit | None,
    unpolished_stored: SpaNormFit | None,
    Y,
    msgfun,
    report_progress,
    backend: str,
    rng,
    gene_names=None,
) -> tuple[SpaNormFit, SpaNormFit | None]:
    """Fit, retrieve or re-polish the null so it is paired with the full fit's polish state.

    Returns ``(null, unpolished)``; ``unpolished`` is the pre-polish null when one was polished
    here (else ``None``), for the caller to store as ``spanorm_null_unpolished``. Cases:

    * no stored null: fit it, and polish it too if the full fit is polished, with all of the
      full fit's recorded polish settings;
    * a stored null polished alike: used as is;
    * the polish state differs: a polished full fit with an unpolished null polishes the null
      here; an unpolished full fit with a polished null is an error (polishing is
      :func:`~spanormpy.polish_spanorm`'s job);
    * both polished with different settings: the null is re-polished with the full fit's
      settings, from the stored ``spanorm_null_unpolished`` when there is one, else refitted.
    """
    full_polished = is_polished(fit_spanorm)

    def polish_null(nul: SpaNormFit) -> SpaNormFit:
        s = fit_spanorm.polish["settings"]
        return polish_spanorm_fit(
            nul,
            Y,
            psi_method=s["psi_method"],
            ls=s["ls"],
            cells=s["cells"],
            maxit=s["maxit"],
            tol=s["tol"],
            verbose=False,
            name="spanorm_null",
            gene_names=gene_names,
            psi_range=polish_psi_range(s),
        )

    def fit_null() -> SpaNormFit:
        return fit_spanorm_technical(Y, fit_spanorm, msgfun, rng=rng, backend=backend)

    if fit_technical is None:
        report_progress(
            "Fitting and polishing Null SpaNorm model"
            if full_polished
            else "Fitting Null SpaNorm model"
        )
        nul = fit_null()
        if not full_polished:
            return nul, None
        return polish_null(nul), nul

    if _polished_alike(fit_spanorm, fit_technical):
        report_progress("Retrieving Null SpaNorm model")
        return fit_technical, None

    null_polished = is_polished(fit_technical)
    if full_polished and not null_polished:
        _message(
            "the stored null SpaNorm model is not polished; polishing it to match the "
            "polished full model"
        )
        report_progress("Polishing Null SpaNorm model")
        return polish_null(fit_technical), fit_technical
    if not full_polished:
        raise ValueError(
            "the full SpaNorm model is not polished but the stored null SpaNorm model is; "
            "call 'polish_spanorm()' to polish both (or re-fit an unpolished null) before "
            "rerunning 'spanorm_svg()'"
        )

    _message(
        "the stored null SpaNorm model is polished with different settings than the full "
        f"model ({_polish_alike_diff(fit_spanorm, fit_technical)}); re-polishing it to match"
    )
    report_progress("Polishing Null SpaNorm model")
    source = (
        unpolished_stored
        if unpolished_stored is not None and not is_polished(unpolished_stored)
        else fit_null()
    )
    return polish_null(source), source


# ------------------------------------------------------------------ public API


def spanorm_svg(
    data,
    backend: str = "auto",
    verbose: bool = True,
    layer: str | None = None,
    table_key: str | None = None,
    copy: bool = False,
    random_state: int = 0,
):
    """Model-based spatially variable gene (SVG) calling (R ``SpaNormSVG``).

    Uses the stored SpaNorm fit as the full model, fits (or reuses) a nested model without the
    splines representing biology, and compares them per gene by a likelihood ratio F-test. The
    statistics are written to ``var["svg_F"]``, ``var["svg_p"]`` and ``var["svg_fdr"]``, the
    unclamped statistic to ``var["svg_F_raw"]``, and the null fit is stored as
    ``spanorm_null``. When the full fit is polished, the null is polished with the same
    settings (and the pre-polish null is kept as ``spanorm_null_unpolished``).

    Parameters
    ----------
    data
        An ``AnnData`` or ``SpatialData`` on which :func:`~spanormpy.spanorm` has been run,
        with the raw counts in ``X`` or ``layer``.
    backend
        ``"auto"``, ``"cpu"`` or ``"gpu"``, used to fit the null model.
    verbose
        Print progress messages (to stderr, like R's ``message``).
    layer, table_key
        Counts layer (default ``X``) and, for ``SpatialData``, the table.
    copy
        Return a modified copy instead of modifying ``data`` in place.
    random_state
        Seed of the generator behind the null model's dispersion subsampling.

    Returns
    -------
    ``None``, or with ``copy=True`` a modified copy of the same container type.
    """
    adata, write_back = resolve_input(data, layer, "spatial", table_key)
    if copy:
        adata = adata.copy()
    backend = _match_arg(backend, _BACKENDS, "backend")
    msgfun = _message if verbose else _quiet
    total_steps = 3
    step = 0

    def report_progress(msg):
        nonlocal step
        if verbose:
            step += 1
            msgfun(f"({step}/{total_steps}) {msg}")

    Y = get_counts(adata, layer)

    if all(c in adata.var.columns for c in _TABLE_COLUMNS):
        warnings.warn("SVG results exist in 'adata' and will be overwritten", stacklevel=2)

    report_progress("Retrieving SpaNorm model")
    fit_spanorm = get_spanorm_fit(adata)
    fit_technical = get_spanorm_fit(adata, null=True, validate=False)
    unpolished_stored = _stored_unpolished(
        adata, "spanorm_null", load_fit(adata, "spanorm_null_unpolished", validate=False)
    )
    nul, unpolished = _svg_paired_null(
        fit_spanorm,
        fit_technical,
        unpolished_stored,
        Y,
        msgfun,
        report_progress,
        backend,
        np.random.default_rng(random_state),
        gene_names=np.asarray(adata.var_names),
    )

    report_progress("Finding SVGs")
    df_svg = svg_test(Y, fit_spanorm, nul)
    msgfun(f"{int(np.sum(df_svg['svg_fdr'].to_numpy() < 0.05))} SVGs found (FDR < 0.05)")

    # ---- write results (a view first becomes an actual AnnData, no warning)
    _as_actual(adata)
    if nul is not fit_technical:
        store_fit(adata, nul, "spanorm_null")
    if unpolished is not None:
        store_fit(adata, unpolished, "spanorm_null_unpolished")
    stale = [c for c in SVG_COLUMNS if c in adata.var.columns]
    if stale:
        adata.var.drop(columns=stale, inplace=True)
    for col in _TABLE_COLUMNS:
        adata.var[col] = df_svg[col].to_numpy()
    adata.var["svg_F_raw"] = df_svg.attrs["F_raw"]
    result = write_back(adata, copy=copy)
    return result if copy else None


def top_svgs(data, n: int = 10, fdr: float = 1.0, table_key: str | None = None) -> pd.DataFrame:
    """The top SVGs from :func:`spanorm_svg` (R ``topSVGs``).

    Parameters
    ----------
    data
        An ``AnnData`` or ``SpatialData`` carrying SVG results.
    n
        Maximum number of genes returned.
    fdr
        False discovery rate threshold in ``[0, 1]``; genes with ``svg_fdr <= fdr`` are kept.
    table_key
        The table to read, for ``SpatialData``.

    Returns
    -------
    A frame with columns ``svg_F``, ``svg_p`` and ``svg_fdr``, indexed by gene name and sorted
    by FDR (ties in gene order).
    """
    if not n > 0:
        raise ValueError("n > 0 is not TRUE")
    if not (0 <= fdr <= 1):
        raise ValueError("fdr >= 0 && fdr <= 1 is not TRUE")
    adata, _ = resolve_input(data, None, "spatial", table_key)
    if not all(c in adata.var.columns for c in _TABLE_COLUMNS):
        raise ValueError("SVG results not found in 'adata'. Please run 'spanorm_svg' first.")
    res = adata.var[list(_TABLE_COLUMNS)].copy()
    res = res[res["svg_fdr"] <= fdr]
    res = res.iloc[np.argsort(res["svg_fdr"].to_numpy(), kind="stable")]
    return res.iloc[: min(int(n), len(res))]


__all__ = ["fit_spanorm_technical", "spanorm_svg", "svg_test", "top_svgs"]
