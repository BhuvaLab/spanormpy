"""``polish_spanorm()``: converge each gene of a SpaNorm fit to its own optimum.

Port of ``R/polishSpaNorm.R`` (SpaNorm c12e5e5), including ``.polishSpaNormCore`` and
``.polishSpaNormFit``. The model mapping and the joint library-size step are in
:mod:`spanormpy._polish_model`.
"""

from __future__ import annotations

import warnings
from dataclasses import replace

import numpy as np
import pandas as pd
import scipy.sparse as sp

from . import _polish_model as _pm
from ._adjust import get_adjustment_fun, normalise_block_size, normalise_blocked
from ._fit import (
    SpaNormFit,
    _as_actual,
    fit_is_stale_for_spots,
    get_spanorm_fit,
    is_polished,
    load_fit,
    store_fit,
)
from ._fitnb import _message, _quiet
from ._io import get_counts, resolve_input
from ._polish_model import (
    ls_uninformed,
    polish_hold_out,
    polish_psi_range,
    polish_shared_ls,
    spanorm_polish_problem,
    unpenalised_rank_deficient,
)
from .spanorm import _ADJ_METHODS, _BACKENDS, _match_arg

_PSI_METHODS = ("fixed", "profile")
_LS = ("fixed", "joint")
_CELLS = ("all", "fit")
_ENGINES = ("batch", "gene")
_POLISH_PASSTHROUGH = {"block_size", "psi_range"}  # R `...` keys forwarded to polish_nb
SVG_COLUMNS = ("svg_F", "svg_p", "svg_fdr", "svg_F_raw")

_GENE_COLS = (
    "iterations", "psi_fitnb", "restarted", "capped", "singular", "psi_bound", "polished",
    "loglik", "held_out",
)


def _polish_source(fit: SpaNormFit, unpolished: SpaNormFit | None, name: str, overwrite: bool):
    """The fit to polish from (R ``.polishSource``).

    An unpolished fit is its own input. A polished one is refused without ``overwrite``; with
    it the stored unpolished fit is the input, so ``<name>_unpolished`` always holds the
    unpolished fit.
    """
    if not is_polished(fit):
        return fit
    if not overwrite:
        raise ValueError(
            f"the '{name}' fit is already polished; pass overwrite=True to polish it again "
            f"from the unpolished fit ('{name}_unpolished')"
        )
    if (
        unpolished is None
        or is_polished(unpolished)
        or unpolished.ngenes != fit.ngenes
        or unpolished.ncells != fit.ncells
    ):
        raise ValueError(
            f"the '{name}' fit is polished but no matching unpolished fit is stored as "
            f"'{name}_unpolished', so there is nothing to polish from; rerun "
            f"'{'spanorm_svg' if name == 'spanorm_null' else 'spanorm'}'"
        )
    return unpolished


def polish_spanorm_fit(
    fit: SpaNormFit,
    Y,
    psi_method: str = "fixed",
    ls: str = "fixed",
    cells: str = "all",
    maxit: int = 50,
    tol: float = 1e-8,
    verbose: bool = False,
    name: str = "spanorm",
    gene_names=None,
    **polish_kw,
) -> SpaNormFit:
    """Polish one SpaNorm-model fit (R ``.polishSpaNormFit``).

    Maps the fit onto polish_nb's generic problem, converges every gene that has a finite
    optimum, and writes the result into a copy of the fit. A gene with no counts in the polished
    cells, or none in one batch level of them, is held out (kept bit for bit, ``polished=False``,
    reason in ``held_out``). With ``ls="joint"`` the shared ``a1`` is then moved to the joint
    optimum over the polished genes, and every gene, held out or not, takes the new ``a1``.

    ``Y`` is genes x cells and matches the fit. ``polish_kw`` go to
    :func:`~spanormpy.polish.polish_nb`.
    """
    # late-bound: tests wrap spanormpy._polish_model.polish_nb
    polish_nb = _pm.polish_nb

    if Y.shape[0] != fit.ngenes or Y.shape[1] != fit.ncells:
        raise ValueError(
            f"the counts ({Y.shape[0]} x {Y.shape[1]}) do not match the SpaNorm fit "
            f"({fit.ngenes} genes x {fit.ncells} cells/spots)"
        )
    prob = spanorm_polish_problem(fit, cells)
    Yc = Y if prob["cells_idx"].all() else Y[:, prob["cells_idx"]]
    held = polish_hold_out(Yc, prob["X"], prob["pen"])
    keep = np.flatnonzero(pd.isna(held))
    nzero = int(np.sum(held == "all-zero"))
    if verbose and nzero > 0:
        _message(
            f"  {nzero} gene{'' if nzero == 1 else 's'} with no counts in the polished "
            f"cells/spots {'is' if nzero == 1 else 'are'} left unpolished"
        )
    nlevel = int(np.sum(held == "zero-batch-level"))
    if nlevel > 0:
        warnings.warn(
            f"{nlevel} gene{'' if nlevel == 1 else 's'} with no counts in some batch level of "
            f"the polished cells/spots {'is' if nlevel == 1 else 'are'} held out of the "
            f"'{name}' polish: that level's coefficient has no finite optimum, so "
            f"{'it keeps' if nlevel == 1 else 'they keep'} the input fit "
            '(polished = False, held_out = "zero-batch-level")',
            stacklevel=3,
        )

    # updated in place, so an unpolished gene keeps its input values bit for bit
    gmean = np.array(fit.gmean, dtype=float)
    alpha = np.array(fit.alpha, dtype=float)
    psi = np.array(fit.psi, dtype=float)
    a1 = prob["a1"]
    ng = fit.ngenes
    genes = {
        "iterations": np.zeros(ng, dtype=np.int64),
        "psi_fitnb": np.array(fit.psi, dtype=float),
        "restarted": np.zeros(ng, dtype=bool),
        "capped": np.zeros(ng, dtype=bool),
        "singular": np.zeros(ng, dtype=bool),
        "psi_bound": np.zeros(ng, dtype=bool),
        "polished": np.zeros(ng, dtype=bool),
        "loglik": np.full(ng, np.nan),
    }
    # an unpenalised block that is collinear on the polished cells makes every gene's
    # information exactly singular. Caught once, by a rank-revealing pass, so the verdict does
    # not turn on each gene's rounding.
    collinear = len(keep) > 0 and unpenalised_rank_deficient(prob["X"], prob["pen"])
    if collinear:
        genes["singular"][keep] = True
    j = None  # the joint step's record (ls = "joint" only)
    if len(keep) and not collinear:
        Yk = Yc[keep]
        probk = dict(prob)
        probk["A0"] = prob["A0"][keep]
        probk["psi"] = prob["psi"][keep]
        p = prob["X"].shape[1]
        pol = polish_nb(
            Yk, probk["X"], probk["A0"], probk["psi"], lambda_a=probk["pen"],
            offset=probk["offset"], start_cols=np.arange(p) == 0, psi_method=psi_method,
            maxit=maxit, tol=tol, verbose=verbose, **polish_kw,
        )
        if ls == "joint":
            j = polish_shared_ls(
                Yk, probk, pol, psi_method=psi_method, verbose=verbose,
                maxit=maxit, tol=tol, **polish_kw,
            )
            pol = j["pol"]
            a1 = j["a1"]
        pa = np.asarray(pol["alpha"])
        gmean[keep] = pa[:, 0]
        alpha[keep, 1:] = pa[:, 1:]
        psi[keep] = pol["psi"]
        for c in pol["polish"].columns:
            genes[c][keep] = pol["polish"][c].to_numpy()
        genes["loglik"][keep] = pol["loglik"]
    elif ls == "joint":
        # nothing informs a1 (no gene to polish, or a collinear block): warn and keep it
        j = ls_uninformed(None, a1)
    # a gene passed to the polish that the engine could not converge keeps its input fit
    # (polish_nb's fallback); the fit is still marked polished, so say how many
    failed = keep[~genes["polished"][keep]]
    if len(failed):
        warnings.warn(
            f"{len(failed)} of the {len(keep)} gene{'' if len(keep) == 1 else 's'} passed to "
            f"the '{name}' polish could not be polished "
            f"({int(genes['singular'][failed].sum())} with a singular information matrix) and "
            f"keep{'s' if len(failed) == 1 else ''} the input fit (polished = False)"
            + (
                ": the unpenalised columns ((gmean) and batch) are collinear on the polished "
                "cells/spots"
                if collinear
                else ""
            ),
            stacklevel=3,
        )
    # the library-size coefficient is one value shared by every gene
    alpha[:, 0] = a1

    from . import __version__

    settings: dict = {
        "psi_method": psi_method,
        "ls": ls,
        "cells": cells,
        "maxit": int(maxit),
        "tol": float(tol),
        "pen": prob["pen"],
        "a1_input": prob["a1"],
        "a1": float(a1),
    }
    if psi_method == "profile":
        # the profile's search interval is part of its objective (a gene whose optimum is on a
        # bound keeps its input psi), so the SVG null polish needs it and "polished alike"
        # compares it; "fixed" never reads it
        settings["psi_range"] = polish_psi_range({"psi_range": polish_kw.get("psi_range")})
    if ls == "joint":
        settings.update(
            ls_iterations=int(j["iterations"]),
            ls_maxit=int(j["maxit_ls"]),
            ls_tol=float(j["tol_ls"]),
            ls_score=float(j["score"]),
            ls_se=float(j["se"]),
            ls_singular=int(j["singular"]),
            ls_converged=bool(j["converged"]),
        )
    settings["spanorm"] = __version__

    genes_df = pd.DataFrame(
        {**genes, "held_out": held},
        index=pd.Index(gene_names) if gene_names is not None else None,
        columns=list(_GENE_COLS),
    )
    out = replace(
        fit, gmean=gmean, alpha=alpha, psi=psi, polish={"settings": settings, "genes": genes_df}
    )
    out.validate()
    return out


def _stored_fits(adata, null: bool) -> dict:
    """The fits ``polish_spanorm`` reads (R ``.polishStoredFits``), by the stored names.

    The full fit must exist and match the data (:func:`get_spanorm_fit` validates both); a null
    is validated only when it is going to be polished.
    """
    nul = None
    if null and get_spanorm_fit(adata, null=True, validate=False) is not None:
        nul = get_spanorm_fit(adata, null=True)
    return {
        "spanorm": get_spanorm_fit(adata),
        "spanorm_null": nul,
        "spanorm_unpolished": load_fit(adata, "spanorm_unpolished", validate=False),
        "spanorm_null_unpolished": load_fit(adata, "spanorm_null_unpolished", validate=False),
    }


def _stored_unpolished(adata, name: str, fit: SpaNormFit | None):
    """The stored ``<name>_unpolished`` fit, or None when absent or stale for the spots."""
    if fit is None or fit_is_stale_for_spots(adata, f"{name}_unpolished"):
        return None
    return fit


def polish_spanorm(
    data,
    adj_method: str = "auto",
    scale_factor: float = 1.0,
    psi_method: str = "fixed",
    ls: str = "fixed",
    cells: str = "all",
    null: bool = True,
    maxit: int = 50,
    tol: float = 1e-8,
    engine: str = "batch",
    batch_size: int | None = None,
    backend: str = "cpu",
    n_jobs: int = 1,
    overwrite: bool = False,
    verbose: bool = True,
    layer: str | None = None,
    table_key: str | None = None,
    copy: bool = False,
    **kwargs,
):
    r"""Converge each gene of a SpaNorm fit to its own optimum (R ``polishSpaNorm``).

    :func:`~spanormpy.spanorm` fits every gene in one shared IRLS loop, with a gene-averaged
    cell weight vector and an aggregate convergence criterion, so a gene can be left short of its
    own optimum (bright, spatially restricted genes most of all). This takes the stored fit,
    converges each gene separately under SpaNorm's own model, stores the polished fit and
    rewrites ``layers["logcounts"]`` from it.

    SpaNorm's mean for gene :math:`g` in spot :math:`i` is
    :math:`\log\mu_{gi} = \bar\mu_g + a_1 w_{i1} + \sum_{j\ge2} W_{ij}\alpha_{gj}`, where
    :math:`w_{i1}` is the log library size and :math:`a_1` is shared by every gene. The polish
    maximises each gene's penalised NB log-likelihood by damped Newton (see
    :func:`~spanormpy.polish.polish_nb`), with :math:`a_1` held at the fit's value
    (``ls="fixed"``), :math:`\bar\mu_g` an unpenalised per-gene intercept, and the other
    coefficients ridge-penalised as SpaNorm penalises them. The polished coefficients are the
    optimum of the unwinsorised penalised likelihood; the normalisation still applies SpaNorm's
    usual winsorisation, as for an unpolished fit.

    A gene with no counts in the polished spots, or none in every polished spot of one batch
    level, has no finite optimum and is not polished: it keeps its input coefficients and
    dispersion, and its diagnostics row reads ``polished=False`` with ``held_out`` naming the
    reason (``"all-zero"`` or ``"zero-batch-level"``). Genes held out for a batch level are
    counted in a warning. A gene the Newton engine cannot converge also keeps its input fit,
    and a warning counts these. A design whose unpenalised columns are collinear is detected up
    front and no gene is passed to the engine; the fit is marked polished either way.

    With ``ls="joint"`` the shared coefficient moves to the optimum of the total penalised
    log-likelihood over the polished genes: Newton steps on its profile log-likelihood alternate
    with a warm re-polish of every gene at the candidate value. It stops when the standardised
    score :math:`|U|/\sqrt{I}` is below ``1e-3`` or after 10 steps; ``maxit`` and ``tol`` are the
    per-gene Newton's. The joint estimate weights genes by their information, so bright genes
    dominate it. Genes that are not polished take the new shared :math:`a_1` but keep their
    other coefficients and dispersion.

    The polished fit replaces ``spanorm`` and the input fit is kept as ``spanorm_unpolished``
    (see :func:`~spanormpy._fit.store_fit` for the layout). With ``null=True`` a stored
    ``spanorm_null`` (from the SVG test) is polished with the same settings and the input kept as
    ``spanorm_null_unpolished``. SVG results in ``var`` were computed from the unpolished fit, so
    they are removed with a warning; rerun ``spanorm_svg``. A fit stored for a different number
    of spots (a spot subset of the object) is an error, as in R; a gene subset reuses the stored
    fit (see ``docs/r-discrepancies.md``).

    The NB likelihood is defined on counts only, so a non-integer counts matrix is refused.

    Parameters
    ----------
    data
        An ``AnnData`` or ``SpatialData`` that :func:`~spanormpy.spanorm` has been run on, with
        the raw counts in ``X`` or ``layer``.
    adj_method
        ``"auto"`` (= ``"logpac"``), ``"logpac"``, ``"pearson"``, ``"medbio"`` or ``"meanbio"``:
        the adjustment used to rewrite the normalised data.
    scale_factor
        Scaling factor for the adjusted counts.
    psi_method
        ``"fixed"`` keeps the fit's dispersion and converges the mean under it; ``"profile"``
        re-estimates it by profile ML at the converged mean and re-polishes.
    ls
        The library-size coefficient shared by every gene: ``"fixed"`` holds it at the fit's
        value, ``"joint"`` re-estimates it jointly with the per-gene coefficients.
    cells
        The spots each gene is converged over: ``"all"``, or ``"fit"`` for the ones
        :func:`~spanormpy.spanorm` sampled to fit the model.
    null
        Also polish the null model stored by ``spanorm_svg``, when there is one. The SVG test
        compares the two fits, so they must be estimated the same way.
    maxit, tol
        The Newton iteration cap and relative log-likelihood tolerance, per gene.
    engine, batch_size, backend, n_jobs
        As in :func:`~spanormpy.polish.polish_nb`; ``backend`` and ``n_jobs`` also drive the
        normalisation.
    overwrite
        Polish an already polished fit again, starting from the unpolished fit kept as
        ``spanorm_unpolished`` (and ``spanorm_null_unpolished``), never from the polished one.
    verbose
        Print progress messages (to stderr, like R's ``message``).
    layer, table_key
        Counts layer (default ``X``) and, for ``SpatialData``, the table.
    copy
        Return a modified copy instead of modifying ``data`` in place.
    **kwargs
        R's ``...``: ``block_size`` and ``psi_range`` are passed to ``polish_nb``; any other key
        is ignored with a ``UserWarning``.

    Returns
    -------
    ``None``, or with ``copy=True`` a modified copy of the same container type.
    """
    unknown = sorted(set(kwargs) - _POLISH_PASSTHROUGH)
    if unknown:
        warnings.warn(f"unrecognised arguments ignored: {', '.join(unknown)}", stacklevel=2)
    passthrough = {k: v for k, v in kwargs.items() if k in _POLISH_PASSTHROUGH}
    adata, write_back = resolve_input(data, layer, "spatial", table_key)
    if copy:
        adata = adata.copy()
    try:
        Y = get_counts(adata, layer)
    except ValueError as e:
        if "non-integer" not in str(e):
            raise
        raise ValueError(
            "polish_spanorm() needs integer counts: the negative binomial likelihood is "
            "undefined otherwise.\n  The counts are not integer-valued (e.g. a back-transform "
            "such as 2**logcounts - 1 written over the raw counts).\n  Use the raw counts."
        ) from None
    adj_method = _match_arg(adj_method, _ADJ_METHODS, "adj_method")
    psi_method = _match_arg(psi_method, _PSI_METHODS, "psi_method")
    ls = _match_arg(ls, _LS, "ls")
    cells = _match_arg(cells, _CELLS, "cells")
    engine = _match_arg(engine, _ENGINES, "engine")
    backend = _match_arg(backend, _BACKENDS, "backend")
    msgfun = _message if verbose else _quiet

    stored = _stored_fits(adata, bool(null))
    # every refusal happens before any polishing
    full_in = _polish_source(
        stored["spanorm"], _stored_unpolished(adata, "spanorm", stored["spanorm_unpolished"]),
        "spanorm", overwrite,
    )
    null_in = None
    if null and stored["spanorm_null"] is not None:
        null_in = _polish_source(
            stored["spanorm_null"],
            _stored_unpolished(adata, "spanorm_null", stored["spanorm_null_unpolished"]),
            "spanorm_null", overwrite,
        )
    nsteps = 2 + (null_in is not None)
    step = 0

    def progress(msg):
        nonlocal step
        step += 1
        msgfun(f"({step}/{nsteps}) {msg}")

    kw = dict(
        engine=engine, batch_size=batch_size, backend=backend, n_jobs=n_jobs, **passthrough
    )
    names = np.asarray(adata.var_names)
    progress("Polishing spanorm model")
    full = polish_spanorm_fit(
        full_in, Y, psi_method=psi_method, ls=ls, cells=cells, maxit=maxit, tol=tol,
        verbose=verbose, name="spanorm", gene_names=names, **kw,
    )
    nul = None
    if null_in is not None:
        progress("Polishing null spanorm model")
        nul = polish_spanorm_fit(
            null_in, Y, psi_method=psi_method, ls=ls, cells=cells, maxit=maxit, tol=tol,
            verbose=verbose, name="spanorm_null", gene_names=names, **kw,
        )

    progress("Normalising data")
    adj_fun = get_adjustment_fun(full.gene_model, adj_method)
    normmat = normalise_blocked(
        adj_fun, Y, scale_factor, full, n_jobs=n_jobs, block_size=normalise_block_size(Y),
        backend=backend,
    )

    # ---- write results (a view first becomes an actual AnnData, no warning)
    _as_actual(adata)
    store_fit(adata, full_in, "spanorm_unpolished")
    store_fit(adata, full, "spanorm")
    if nul is not None:
        store_fit(adata, null_in, "spanorm_null_unpolished")
        store_fit(adata, nul, "spanorm_null")
    # the SVG statistics compare the fits this call has just replaced
    stale = [c for c in SVG_COLUMNS if c in adata.var.columns]
    if stale:
        warnings.warn(
            "SVG results in 'adata' were computed from the unpolished fit and have been "
            "removed; rerun 'spanorm_svg'",
            stacklevel=2,
        )
        adata.var.drop(columns=stale, inplace=True)
    out = normmat.T
    adata.layers["logcounts"] = sp.csr_matrix(out) if sp.issparse(Y) else np.ascontiguousarray(out)
    adata.uns.setdefault("spanorm", {})["adj_method"] = adj_method
    result = write_back(adata, copy=copy)
    return result if copy else None
