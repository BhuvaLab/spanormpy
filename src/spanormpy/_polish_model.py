"""Maps a SpaNorm-model fit onto the generic polish problem, and the joint library-size step.

Ports ``R/polishSpaNormModel.R`` and ``R/polishSpaNormJoint.R`` (SpaNorm c12e5e5).

SpaNorm's mean is ``log mu_gi = gmean_g + a1 * W_i1 + sum_{j>=2} W_ij alpha_gj``: ``W[:, 0]``
is the log library size and its coefficient ``a1`` is SHARED by every gene (all rows of
``alpha[:, 0]`` are equal); ``gmean`` is a per-gene intercept held outside ``W``; columns 1..p-1
are ridge-penalised by ``wtype`` (biology -> ``lambda_a[0]``, ls -> ``lambda_a[1]``, batch -> 0),
times the number of cells. All arrays keep R's orientation (genes x cells for counts).

``ls = "joint"`` moves ``a1`` to the optimum of the TOTAL penalised log-likelihood over the
polished genes by Newton steps on the profile log-likelihood
``p(a1) = sum_g max_{A_g} l_g(a1, A_g)``::

    U =  p'(a1) = sum_g w1' r_g
    I = -p''(a1) = sum_g [ (w1^2)' D_g - c_g' H_g^-1 c_g ]

with ``D_g = mu / (1 + psi mu)``, ``r_g = (y - mu) / (1 + psi mu)``, ``c_g = X'(D_g * w1)`` and
``H_g = X' D_g X + diag(pen)``. ``I`` is the Fisher information; each gene's term is the Schur
complement of ``H_g`` in the augmented information of ``(A_g, a1)``, so it is >= 0.
"""

from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from ._backend import as_tensor
from ._polish_batch import mu_batch, nb_loglik_batch, offset_rows
from ._settings import _match_arg
from ._solver import nb_newton_solver
from .polish import (  # looked up by name so tests can wrap them
    _dense_rows,
    chunk_genes,
    nb_profile_psi,
    polish_nb,
)

LS_MAXIT = 10
LS_TOL = 1e-3
"""The joint loop's own stop: cap on steps and the standardised score ``|U| / sqrt(I)``.

The standardised score is ``a1``'s distance from its optimum in units of its own profiled SE.
1e-3 asks for ``a1`` within a thousandth of its SE, still far inside its sampling error and
what the inner per-gene polish can resolve (R measured 1e-6 hitting the cap on real data).
"""

DEFAULT_PSI_RANGE = (1e-3, 1e3)


def _msg(text: str) -> None:
    print(text, file=sys.stderr)


# ----------------------------------------------------------------------------- the problem


def spanorm_penalty(fit) -> np.ndarray:
    """The penalty vector SpaNorm's fitters use (R ``.spaNormPenalty``).

    ``biology -> lambda_a[0]``, ``ls -> lambda_a[1]``, ``batch -> 0``, times ``ncells``.
    Column 1 (the shared log library size) is not in the per-gene problem, hence the leading 0
    for the unpenalised ``gmean`` column.
    """
    lam = np.resize(np.atleast_1d(np.asarray(fit.lambda_a, dtype=float)), 2)
    wtype = np.asarray(fit.wtype).astype(str)
    v = np.zeros(len(wtype))
    v[wtype == "biology"] = lam[0]
    v[wtype == "ls"] = lam[1]
    return np.concatenate([[0.0], v[1:] * fit.ncells])


def spanorm_polish_problem(fit, cells: str = "all") -> dict:
    """Map a SpaNorm-model fit onto polish_nb's generic problem (R ``.spaNormPolishProblem``).

    The shared library-size coefficient (``alpha[:, 0]``) is pulled out of ``W`` into a per-cell
    offset, and the per-gene intercept ``gmean`` becomes an explicit unpenalised first column of
    ``X``. ``cells="fit"`` keeps the cells the model was fitted on (``sampling != "all"``).

    Returns a dict with ``X``, ``pen``, ``a1``, ``offset``, ``w1`` (the selected cells' log
    library size), ``A0`` (starting coefficients, genes x ncol(X)), ``psi`` and ``cells_idx``
    (boolean, one per cell).
    """
    cells = _match_arg(cells, ("all", "fit"), "cells")
    alpha = np.asarray(fit.alpha, dtype=float)
    a1s = alpha[:, 0]
    if np.ptp(a1s) > 1e-8 * max(1.0, abs(a1s[0])):
        raise ValueError(
            "column 1 of the fit's alpha (the library-size coefficient) is not shared across "
            "genes, so this is not a SpaNorm-model fit; use polish_nb() for a generic "
            "fit_nb() fit"
        )
    W = np.asarray(fit.W, dtype=float)
    idx = np.ones(W.shape[0], dtype=bool) if cells == "all" else np.asarray(fit.sampling) != "all"
    W = W[idx]
    a1 = float(a1s[0])
    return {
        "X": np.column_stack([np.ones(W.shape[0]), W[:, 1:]]),
        "pen": spanorm_penalty(fit),
        "a1": a1,
        "offset": a1 * W[:, 0],
        "w1": W[:, 0],
        "A0": np.column_stack([np.asarray(fit.gmean, dtype=float), alpha[:, 1:]]),
        "psi": np.asarray(fit.psi, dtype=float),
        "cells_idx": idx,
    }


def unpenalised_rank_deficient(X, pen) -> bool:
    """Is the unpenalised block of the polish design rank-deficient? (R ``.unpenalisedRankDeficient``)

    A gene's information ``X'WX + diag(pen)`` is positive definite exactly when the unpenalised
    columns (``pen == 0``) have full column rank on the polished cells, so one rank-revealing
    pass over those few columns decides singularity for every gene at once. The rank follows
    R's ``qr()`` (LINPACK ``dqrdc2``, tolerance 1e-7): a column is dependent when its norm after
    projecting out the earlier independent columns is below ``1e-7`` times its original norm.
    """
    U = np.asarray(X, dtype=float)[:, np.asarray(pen) == 0]
    if U.shape[1] <= 1:
        return False
    tol = 1e-7
    basis: list[np.ndarray] = []
    for j in range(U.shape[1]):
        v = U[:, j].copy()
        orig = np.linalg.norm(v)
        if orig == 0:
            orig = 1.0
        for _ in range(2):  # re-orthogonalise: classical Gram-Schmidt loses digits when nearly dependent
            for q in basis:
                v -= q * (q @ v)
        nrm = np.linalg.norm(v)
        if nrm < tol * orig:
            continue
        basis.append(v / nrm)
    return len(basis) < U.shape[1]


def unpenalised_groups(X, pen) -> np.ndarray | None:
    """The groups of cells the unpenalised indicator columns separate (R ``.unpenalisedGroups``).

    Column 0 of the polish design is the unpenalised intercept; the other unpenalised columns
    are SpaNorm's batch columns (and any spline column ``lambda_a`` leaves unpenalised). When
    each of those is a 0/1 indicator, cells sharing a row of the unpenalised block form one
    group, numbered from 1 by first appearance. Returns all 1 when only the intercept is
    unpenalised, and ``None`` when an unpenalised column other than the intercept is not 0/1.
    """
    X = np.asarray(X, dtype=float)
    cols = [j for j in np.flatnonzero(np.asarray(pen) == 0) if j != 0]
    g = np.ones(X.shape[0], dtype=np.int64)
    if not cols:
        return g
    Xu = X[:, cols]
    if not np.all((Xu == 0) | (Xu == 1)):
        return None
    for j in range(Xu.shape[1]):
        key = 2 * g + Xu[:, j].astype(np.int64)
        g = pd.factorize(key)[0] + 1  # first-appearance numbering, re-compacted each column
    return g


def _row_sums(Y) -> np.ndarray:
    if sp.issparse(Y):
        return np.asarray(Y.sum(axis=1)).reshape(-1).astype(float)
    return np.asarray(Y, dtype=float).sum(axis=1)


def polish_hold_out(Y, X, pen) -> np.ndarray:
    """Which genes the polish holds out, and why (R ``.polishHoldOut``).

    A gene has no finite optimum when it has no counts on the whole support of an unpenalised
    direction of the design: in every polished cell (the intercept runs to -Inf), or in every
    cell of one group the unpenalised indicator columns separate.

    Returns an object array, one per gene: ``"all-zero"``, ``"zero-batch-level"`` or ``None``
    for a gene to polish.
    """
    total = _row_sums(Y)
    why = np.full(len(total), None, dtype=object)
    grp = unpenalised_groups(X, pen)
    if grp is not None and np.any(grp != 1):
        zero = np.zeros(len(total), dtype=bool)
        for k in np.unique(grp):
            zero |= _row_sums(Y[:, grp == k]) <= 0
        why[zero] = "zero-batch-level"
    why[np.isnan(total) | (total <= 0)] = "all-zero"
    return why


def polish_psi_range(settings) -> np.ndarray:
    """The profile dispersion's search interval a polish used (R ``.polishPsiRange``).

    A missing element means polish_nb's default, which is what a profile polish used before
    the interval was recorded.
    """
    r = settings.get("psi_range") if settings is not None else None
    if r is None:
        r = DEFAULT_PSI_RANGE
    return np.asarray(r, dtype=float)


# ----------------------------------------------------------------------------- joint a1


def ls_score_info(Y, A, psi, a1, X, w1, pen, solver, block: int = 500) -> dict:
    """The pooled score and profiled information for the shared ``a1`` (R ``.lsScoreInfo``).

    Blocked over genes (``block`` at a time), so the dense genes x cells matrices are bounded by
    the block. A gene whose penalised information is singular (the solver returns ``None``, or a
    non-finite solution) is left out of BOTH sums and reported in ``excluded`` (0-based row
    indices): a zero correction would add its unprofiled ``(w1^2)' D_g`` to ``I``.

    Returns ``U``, ``I``, ``excluded`` and ``n`` (the genes summed).
    """
    dev = torch.device("cpu")
    A = np.asarray(A, dtype=float)
    psi_all = np.asarray(psi, dtype=float).reshape(-1)
    Xt = as_tensor(X, dev, torch.float64)
    w1t = as_tensor(w1, dev, torch.float64)
    Xw = Xt * w1t[:, None]
    w1sq = w1t**2
    U = 0.0
    I = 0.0
    excluded: list[int] = []
    for b in chunk_genes(A.shape[0], block):
        Ab = as_tensor(A[b], dev, torch.float64)
        Mu = mu_batch(Ab, Xt, offset_rows(a1 * w1t, np.arange(len(b)), Ab))
        pb = as_tensor(psi_all[b], dev, torch.float64)[:, None]
        den = 1 + pb * Mu
        Yb = as_tensor(_dense_rows(Y, b), dev, torch.float64)
        R = (Yb - Mu) / den
        D = Mu / den
        del Mu, den, Yb
        u = (R @ w1t).numpy()
        del R
        d11 = (D @ w1sq).numpy()
        C = D @ Xw
        for k in range(len(b)):
            h = solver.solve(solver.factor(D[k]), C[k])
            if h is None or not bool(torch.isfinite(h).all()):
                excluded.append(int(b[k]))
                continue
            U += float(u[k])
            I += float(d11[k]) - float((C[k] * h).sum())
    return {"U": U, "I": I, "excluded": excluded, "n": A.shape[0] - len(excluded)}


def ls_loglik(Y, A, psi, a1, X, w1, pen, block: int = 500) -> np.ndarray:
    """Per-gene penalised log-likelihood at a shared ``a1``, blocked over genes (R ``.lsLoglik``)."""
    dev = torch.device("cpu")
    A = np.asarray(A, dtype=float)
    psi_all = np.asarray(psi, dtype=float).reshape(-1)
    Xt = as_tensor(X, dev, torch.float64)
    w1t = as_tensor(w1, dev, torch.float64)
    out = np.empty(A.shape[0])
    for b in chunk_genes(A.shape[0], block):
        Ab = as_tensor(A[b], dev, torch.float64)
        Mu = mu_batch(Ab, Xt, offset_rows(a1 * w1t, np.arange(len(b)), Ab))
        Yb = as_tensor(_dense_rows(Y, b), dev, torch.float64)
        out[b] = nb_loglik_batch(Yb, Mu, psi_all[b], Ab, pen).numpy()
    return out


def ls_uninformed(
    pol,
    a1,
    why: str = "no gene was polished",
    what: str = "nothing informs the shared library-size coefficient",
    iterations: int = 0,
    singular: int = 0,
    maxit_ls: int = LS_MAXIT,
    tol_ls: float = LS_TOL,
) -> dict:
    """The joint step's result when it cannot inform ``a1`` (R ``.lsUninformed``).

    Warns and returns the cold pass ``pol`` and the fit's ``a1`` unchanged, so the fit is the
    fixed-``a1`` polish with the joint attempt recorded.
    """
    warnings.warn(
        f'ls = "joint": {why}, so {what}; it is left at the fit\'s value and the genes at their '
        "fixed-a1 polish",
        stacklevel=3,
    )
    return {
        "pol": pol,
        "a1": a1,
        "iterations": int(iterations),
        "score": np.nan,
        "se": np.nan,
        "singular": int(singular),
        "converged": False,
        "maxit_ls": int(maxit_ls),
        "tol_ls": tol_ls,
    }


def polish_shared_ls(
    Y,
    prob: dict,
    pol: dict,
    psi_method: str = "fixed",
    maxit_ls: int = LS_MAXIT,
    tol_ls: float | None = None,
    block: int = 500,
    verbose: bool = False,
    **polish_kw,
) -> dict:
    """The joint polish: alternate the per-gene polish with a pooled step on ``a1`` (R ``.polishSharedLS``).

    Starts from a converged per-gene polish at the fit's ``a1`` (``pol``, from ``polish_nb``) and
    takes Newton steps on the profile log-likelihood of ``a1``, each with a line search on the
    total penalised log-likelihood after a warm re-polish at the candidate ``a1``. It stops when
    ``|U| / sqrt(I) <= tol_ls``, at ``maxit_ls`` steps, or when no halving of the step keeps the
    total from falling. A non-positive ``I`` returns the cold pass with a warning.

    Only the genes ``pol`` polished take part. Within an iteration the step, the objective it is
    judged on and the stop use the genes that formed ``U`` and ``I``. Under
    ``psi_method="profile"`` a candidate is a warm re-polish, a profile of the dispersion at that
    mean and a warm re-polish at it.

    ``polish_kw`` go to every warm :func:`~spanormpy.polish.polish_nb` (``maxit``, ``tol``,
    ``engine``, ``batch_size``, ``block_size``, ``backend``, ``n_jobs``); ``psi_range``,
    ``block_size`` and ``n_jobs`` also reach :func:`~spanormpy.polish.nb_profile_psi`.

    Returns a dict with ``pol`` (as ``polish_nb`` returns it, at the joint ``a1``), ``a1``,
    ``iterations`` (accepted steps), ``score`` (the final ``U``), ``se`` (``1 / sqrt(I)``),
    ``singular`` (genes left out of ``U`` and ``I`` at some step), ``converged``, ``maxit_ls``
    and ``tol_ls``.
    """
    if tol_ls is None:
        tol_ls = LS_TOL
    X = prob["X"]
    w1 = prob["w1"]
    pen = prob["pen"]
    a1 = float(prob["a1"])
    polished = np.asarray(pol["polish"]["polished"], dtype=bool)
    ok = np.flatnonzero(polished)

    def uninformed(**kw):
        return ls_uninformed(pol, prob["a1"], maxit_ls=maxit_ls, tol_ls=tol_ls, **kw)

    if len(ok) == 0:
        return uninformed()

    Yo = Y[ok]
    A = np.array(pol["alpha"])[ok]
    psi = np.asarray(pol["psi"], dtype=float)[ok]
    solver = nb_newton_solver(as_tensor(X, "cpu", torch.float64), as_tensor(pen, "cpu", torch.float64))
    psi_range = polish_kw.get("psi_range", DEFAULT_PSI_RANGE)

    def repolish(A, psi, a1):
        off = a1 * w1

        def warm(A, psi):
            return polish_nb(
                Yo, X, A, psi, lambda_a=pen, offset=off, psi_method="fixed", warm=True,
                **polish_kw,
            )

        p = warm(A, psi)
        it = np.asarray(p["polish"]["iterations"]).copy()
        capped = np.asarray(p["polish"]["capped"], dtype=bool).copy()
        # a warm pass that fails keeps the gene's previous coefficients
        singular = np.asarray(p["polish"]["singular"], dtype=bool) | ~np.asarray(
            p["polish"]["polished"], dtype=bool
        )
        if psi_method == "profile":
            # the profile is a fixed-iteration bisection: it takes no maxit or tol
            psi = nb_profile_psi(
                Yo, X, p["alpha"], p["psi"], psi_range=psi_range,
                block_size=polish_kw.get("block_size"), n_jobs=polish_kw.get("n_jobs", 1),
                offset=off,
            )
            p2 = warm(p["alpha"], psi)
            it = it + np.asarray(p2["polish"]["iterations"])
            capped = capped | np.asarray(p2["polish"]["capped"], dtype=bool)
            singular = (
                singular
                | np.asarray(p2["polish"]["singular"], dtype=bool)
                | ~np.asarray(p2["polish"]["polished"], dtype=bool)
            )
            p = p2
        return {
            "alpha": np.array(p["alpha"]),
            "psi": np.asarray(p["psi"], dtype=float),
            "iterations": it,
            "capped": capped,
            "singular": singular,
        }

    def loglik(A, psi, a1):
        return ls_loglik(Yo, A, psi, a1, X, w1, pen, block)

    def score(A, psi, a1):
        return ls_score_info(Yo, A, psi, a1, X, w1, pen, solver, block)

    def zscore(si):
        return abs(si["U"]) / np.sqrt(si["I"]) if np.isfinite(si["I"]) and si["I"] > 0 else np.nan

    ll_g = loglik(A, psi, a1)
    si = score(A, psi, a1)
    excluded = list(si["excluded"])
    extra_it = np.zeros(len(ok), dtype=np.int64)
    extra_capped = np.zeros(len(ok), dtype=bool)
    extra_singular = np.zeros(len(ok), dtype=bool)
    it = 0
    stop_why = None
    if verbose:
        _msg(f"  joint library size: a1 = {a1:.6g}, |U|/sqrt(I) = {zscore(si):.3g}")
    while True:
        if si["n"] == 0:
            # every polished gene's information is singular: nothing informs a1
            if it == 0:
                return uninformed(
                    why="every polished gene's information is singular", singular=len(excluded)
                )
            stop_why = "every polished gene's information became singular"
            break
        if not np.isfinite(si["I"]) or si["I"] <= 0:
            # the profiled information is a sum of Schur complements, so >= 0; a non-positive
            # value means w1 is (numerically) in the span of the per-gene design and a1 is
            # not identified: return the fixed-a1 polish.
            return uninformed(
                why=f"the profiled information for a1 is not positive ({si['I']:g})",
                what=(
                    "a1 is not identified (the log library size may be collinear with the "
                    "model's other terms)"
                ),
                iterations=it,
                singular=len(excluded),
            )
        if zscore(si) <= tol_ls:
            break
        if it >= maxit_ls:
            stop_why = f"maxit_ls = {maxit_ls} reached"
            break
        # The step, the objective it is judged on and the stop refer to ONE gene set: the
        # genes that formed U and I this iteration.
        in_set = np.setdiff1d(np.arange(len(ok)), np.asarray(si["excluded"], dtype=int))
        ll = float(ll_g[in_set].sum())
        step = si["U"] / si["I"]
        accepted = False
        for h in range(11):
            a1n = a1 + step / 2**h
            cand = repolish(A, psi, a1n)
            lln_g = loglik(cand["alpha"], cand["psi"], a1n)
            lln = float(lln_g[in_set].sum())
            if np.isfinite(lln) and lln >= ll - 1e-9 * abs(ll):
                accepted = True
                break
        if not accepted:
            stop_why = "no step along the Newton direction raised the total likelihood"
            break
        it += 1
        a1 = a1n
        A = cand["alpha"]
        psi = cand["psi"]
        ll_g = lln_g
        extra_it = extra_it + cand["iterations"]
        extra_capped = extra_capped | cand["capped"]
        extra_singular = extra_singular | cand["singular"]
        si = score(A, psi, a1)
        excluded = sorted(set(excluded) | set(si["excluded"]))
        if verbose:
            _msg(
                f"  joint library size: step {it}, a1 = {a1:.6g}, "
                f"|U|/sqrt(I) = {zscore(si):.3g}"
            )
    converged = stop_why is None
    if not converged:
        warnings.warn(
            'ls = "joint": the step on the shared library-size coefficient stopped with '
            f"|U|/sqrt(I) = {zscore(si):.3g}, above tol_ls = {tol_ls:g} ({stop_why})",
            stacklevel=3,
        )

    out = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in pol.items() if k != "polish"}
    out["alpha"] = np.array(pol["alpha"], dtype=float)
    out["psi"] = np.array(pol["psi"], dtype=float)
    out["loglik"] = np.array(pol["loglik"], dtype=float)
    out["alpha"][ok] = A
    out["psi"][ok] = psi
    out["loglik"][ok] = ll_g
    pdf = pol["polish"].copy()
    pdf.iloc[ok, pdf.columns.get_loc("iterations")] = (
        pdf["iterations"].to_numpy()[ok] + extra_it
    )
    pdf.iloc[ok, pdf.columns.get_loc("capped")] = pdf["capped"].to_numpy()[ok] | extra_capped
    pdf.iloc[ok, pdf.columns.get_loc("singular")] = pdf["singular"].to_numpy()[ok] | extra_singular
    out["polish"] = pdf
    return {
        "pol": out,
        "a1": a1,
        "iterations": it,
        "score": si["U"],
        "se": 1 / np.sqrt(si["I"]) if np.isfinite(si["I"]) and si["I"] > 0 else np.nan,
        "singular": len(excluded),
        "converged": converged,
        "maxit_ls": int(maxit_ls),
        "tol_ls": tol_ls,
    }
