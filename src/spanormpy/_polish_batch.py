"""The batched polish engine (port of ``R/polishEngineBatch.R``).

The per-gene Newton of :mod:`spanormpy._polish_engine`, restructured so a block of
genes shares every read of the design: the linear predictor and the score become one
GEMM each. Nothing here is a new estimator. The gram stays a per-gene call into the
solver (each gene has its own working weights); the dispersion search is a batched
fixed-iteration bisection on the score instead of Brent's method.

Everything is torch on the device of the counts. Row sets (``rows``, ``act``, ``kk``)
are 0-based numpy integer arrays; per-gene bookkeeping (``ll``, ``stale``, flags) lives
on the host. Float64 only (see :func:`require_float64`).
"""

from __future__ import annotations

import math

import numpy as np
import torch

from ._backend import as_tensor, get_backend_dtype
from ._block_design import (
    _block_col_cells,
    _block_eta_batch,
    _block_score_batch,
    _is_block_design,
)
from ._nbdist import dnbinom
from ._polish_engine import MU_FLOOR
from ._settings import settings
from ._solver import _subset_state, nb_newton_solver_batch

# Live gene x cell matrices inside one batched Newton iteration: the counts slice, mu,
# the residual/weight matrix, the candidate coefficients' mean, the accepted mean, plus
# headroom. Sized by inspection, so it errs high.
POLISH_GENE_CELL_MATS = 6


def require_float64(dtype=None) -> None:
    """Refuse a single-precision device (R ``.requireFloat64``).

    ``get_backend_dtype()`` is float64 on CUDA but float32 on MPS. A penalised Newton
    solve over hundreds of columns in single precision is not defensible, and the
    failure would look like non-convergence rather than lost precision.
    """
    dtype = get_backend_dtype() if dtype is None else dtype
    nm = str(dtype).lower()
    if "float64" not in nm and "double" not in nm:
        raise ValueError(
            f"the polish stage refuses a single precision device (dtype {nm}).\n"
            "  A penalised Newton solve over hundreds of columns in float32 is not "
            "defensible, and the failure would look like non-convergence rather than "
            'lost precision.\n  Use backend = "cpu", or a device with float64 (CUDA has '
            "it; MPS does not)."
        )


def offset_rows(offset, kk, like: torch.Tensor):
    """The offset for the genes in ``kk`` (row indices into the batch), shaped like ``like``.

    ``None``, a per-cell vector (identical for every gene) or a genes x cells matrix.
    Every linear predictor in the batch engine goes through this one function so a
    mis-sliced offset cannot hide at one site.
    """
    if offset is None:
        return None
    kk = np.asarray(kk, dtype=np.int64)
    off = as_tensor(offset, like.device, like.dtype)
    if off.ndim == 1:
        return off[None, :].expand(len(kk), off.numel())
    return off[torch.as_tensor(kk, device=off.device)]


def mu_batch(A: torch.Tensor, W, offset=None) -> torch.Tensor:
    """``exp(A W' + offset)`` floored at the mean floor, for a block of genes at once.

    ``A`` is ``genes x p``; ``W`` the ``cells x p`` design or an ``NBBlockDesign``;
    ``offset`` is ``None`` or already shaped like ``A W'`` (use :func:`offset_rows`).
    The floor is not cosmetic: a degenerate start can drive a linear predictor to -50
    and the log-likelihood to -Inf.
    """
    eta = _block_eta_batch(W, A) if _is_block_design(W) else A @ W.T
    if offset is not None:
        if offset.ndim != 2:
            raise ValueError(
                "mu_batch() takes the offset shaped like A @ W.T; shape it with offset_rows()"
            )
        eta = eta + offset
    return torch.clamp(torch.exp(eta), min=MU_FLOOR)


def nb_loglik_batch(Y: torch.Tensor, M: torch.Tensor, psi, A: torch.Tensor, pen) -> torch.Tensor:
    """Penalised NB log-likelihood per gene: ``sum_cells dnbinom(y; 1/psi, mu) - a' diag(pen) a / 2``.

    ``psi = 0`` is ``size = Inf``, which ``dnbinom`` treats as Poisson, so an all-Poisson
    batch stays finite. ``psi`` and ``pen`` may be host arrays.
    """
    psi_t = as_tensor(psi, M.device, M.dtype).reshape(-1)
    pen_t = as_tensor(pen, M.device, M.dtype).reshape(-1)
    Yt = as_tensor(Y, M.device, M.dtype)
    ll = dnbinom(Yt, mu=M, size=1.0 / psi_t, log=True).sum(1)
    return ll - 0.5 * ((A * A) @ pen_t)


def psi_profile_batch(Y, Mu: torch.Tensor, psi_range=(1e-3, 1e3), maxit: int = 50):
    """Profile dispersion by fixed-iteration bisection on the score (R ``.psiProfileBatch``).

    Replaces a per-gene ``optimize()``: its default tolerance is ~1.2e-4 absolute on a
    log-interval of width 13.8; fifty halvings take that to ~1e-14, with no divergent
    control flow across a batch. With ``r = 1/psi`` the score is

        dl/dr = sum_i digamma(y_i+r) - digamma(r) + log(r/(r+mu_i)) + 1 - (r+y_i)/(r+mu_i)

    and ``dl/d log psi = -r dl/dr`` is positive at the small-psi end for overdispersed
    counts and negative at the large-psi end, so the root is bracketed. A gene whose
    optimum lies outside the range is clamped to the endpoint it ran past and flagged
    ``at_bound``.

    Returns
    -------
    (psi, at_bound)
        ``psi`` a tensor like ``Mu``'s rows; ``at_bound`` a host bool array.
    """
    lo, hi = math.log(psi_range[0]), math.log(psi_range[1])
    edge = 1e-3 * (hi - lo)
    Yt = as_tensor(Y, Mu.device, Mu.dtype)
    b = Mu.shape[0]

    def score(lp):
        r = torch.exp(-lp)[:, None]
        rm = r + Mu
        g = (torch.digamma(Yt + r) - torch.digamma(r) + torch.log(r / rm) + 1 - (r + Yt) / rm).sum(
            1
        )
        return -torch.exp(-lp) * g

    a = torch.full((b,), lo, dtype=Mu.dtype, device=Mu.device)
    z = torch.full((b,), hi, dtype=Mu.dtype, device=Mu.device)
    Sa, Sz = score(a), score(z)
    for _ in range(maxit):
        m = (a + z) / 2
        pos = score(m) > 0
        a = torch.where(pos, m, a)
        z = torch.where(pos, z, m)
    lp = (a + z) / 2
    lp = torch.where(Sa <= 0, torch.full_like(lp, lo), lp)
    lp = torch.where(Sz >= 0, torch.full_like(lp, hi), lp)
    lpr = lp.cpu().numpy()
    at_bound = ((lpr - lo) < edge) | ((hi - lpr) < edge)
    return torch.exp(lp), at_bound


def polish_batch_size(ncells: int, budget: float | None = None, mats: int = POLISH_GENE_CELL_MATS):
    """Genes per batched Newton, from a per-worker memory budget (R ``.polishBatchSize``).

    A gene block bounds densification of the counts; the batched Newton's working set is
    gene x cell and is bounded separately. ``budget`` is bytes for ONE worker (default
    ``spanormpy.settings.polish_mem_budget``). At least 1.
    """
    budget = settings.polish_mem_budget if budget is None else budget
    per_gene = 8.0 * float(ncells) * mats
    return max(1, math.floor(budget / per_gene))


def _rows_finite(M: torch.Tensor) -> np.ndarray:
    return torch.isfinite(M).all(1).cpu().numpy()


def _idx(i, like: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(np.asarray(i, dtype=np.int64), device=like.device)


def _empty_result(
    alpha,
    psi,
    loglik,
    iters,
    restarted,
    capped,
    singular,
    psi_bound,
    polished,
    n_fac=0,
    n_fac_sync=0,
):
    return {
        "alpha": alpha,
        "psi": psi,
        "loglik": loglik,
        "iterations": iters,
        "restarted": restarted,
        "capped": capped,
        "singular": singular,
        "psi_bound": psi_bound,
        "polished": polished,
        "factorisations": {"pergene": int(n_fac), "sync": int(n_fac_sync)},
    }


def polish_batch(
    Yb,
    W,
    A0,
    psi0,
    pen,
    solver,
    maxit: int = 50,
    tol: float = 1e-8,
    start_cols=None,
    psi_range=(1e-3, 1e3),
    psi_method: str = "profile",
    warm: bool = False,
    shared_factor: bool = False,
    nested=None,
    offset=None,
) -> dict:
    """Converge a block of genes to their own penalised NB optima (R ``.polishBatch``).

    Parameters
    ----------
    Yb
        counts for the block, ``genes x cells``, dense (tensor, on the device).
    W
        the design (cells x columns): a tensor, or an ``NBBlockDesign`` (CPU only).
    A0
        the starting coefficients of the block, ``genes x columns``.
    psi0
        starting dispersions (length ``nrow(Yb)``, or one value recycled).
    pen
        per-column ridge penalty (tensor).
    solver
        a per-gene ``NewtonSolver`` (used unless ``shared_factor``).
    shared_factor
        one factorisation for the active set (the state a device can hold), built from
        ``nb_newton_solver_batch(W, pen, nested)``; refreshed when ANY active gene is
        stale.
    offset
        ``None``, a per-cell log-scale offset (length ``ncol(Yb)``) or a
        ``nrow(Yb) x ncol(Yb)`` matrix.

    Returns
    -------
    dict
        host arrays ``alpha`` (genes x columns), ``psi``, ``loglik``, ``iterations``,
        ``restarted``, ``capped``, ``singular``, ``psi_bound``, ``polished``, and
        ``factorisations`` (``{"pergene", "sync"}`` counts).
    """
    if psi_method not in ("profile", "fixed"):
        raise ValueError("'psi_method' should be one of 'profile', 'fixed'")
    Yb = as_tensor(Yb, _device_of(W, Yb), torch.float64)
    dev, dt = Yb.device, Yb.dtype
    B, ncell = Yb.shape
    p = int(W.shape[1])
    pen = as_tensor(pen, dev, dt).reshape(-1)
    # the two shapes offset_rows() slices correctly; anything else is refused here
    # rather than recycled into a wrong linear predictor
    if offset is not None:
        osh = (
            tuple(np.shape(offset)) if not isinstance(offset, torch.Tensor) else tuple(offset.shape)
        )
        ok = (len(osh) == 1 and osh[0] == ncell) or (
            len(osh) == 2 and osh[0] == B and osh[1] == ncell
        )
        if not ok:
            raise ValueError("'offset' must have one value per cell, or be a genes x cells matrix")
        offset = as_tensor(offset, dev, dt)
    psi0 = np.resize(np.asarray(psi0, dtype=np.float64).reshape(-1), B)  # rep_len
    compact = _is_block_design(W)
    A0 = as_tensor(A0, dev, dt)
    n_fac = 0
    n_fac_sync = 0
    # one factorisation for the active set instead of a list of per-gene ones
    solverB = nb_newton_solver_batch(W, pen, nested) if shared_factor else None

    def mu_of(A, kk):
        return mu_batch(A, W, offset_rows(offset, kk, A))

    def ll_of(Y, M, ps, A):
        return nb_loglik_batch(Y, M, ps, A, pen)

    def host(x):
        return x.cpu().numpy().astype(np.float64)

    def newton(rows, A, ps, maxit):
        """Damped Newton over ``rows``; results indexed by position within ``rows``."""
        nonlocal n_fac, n_fac_sync
        rows = np.asarray(rows, dtype=np.int64)
        m = len(rows)
        Y = Yb[_idx(rows, Yb)]
        Ai = A[_idx(rows, A)].clone()
        pi_ = np.asarray(ps, dtype=np.float64)[rows]
        Mu = mu_of(Ai, rows)
        ll = host(ll_of(Y, Mu, pi_, Ai))  # length-genes bookkeeping lives on the host
        it = np.zeros(m, dtype=np.int64)
        conv = np.zeros(m, dtype=bool)
        sing = np.zeros(m, dtype=bool)
        stale = np.zeros(m, dtype=np.int64)
        fac = [None] * m
        st = None
        st_rows = np.zeros(0, dtype=np.int64)
        act = np.arange(m)
        while len(act):
            it[act] += 1
            ia = _idx(act, Y)
            Ya, Ma, Aa = Y[ia], Mu[ia], Ai[ia]
            pa = as_tensor(pi_[act], dev, dt)[:, None]
            R = (Ya - Ma) / (1 + pa * Ma)
            S = (_block_score_batch(R, W) if compact else R @ W) - Aa * pen

            if shared_factor:
                # the stack is aligned to `act`; genes only ever LEAVE the active set
                if st is not None and not np.array_equal(st_rows, act):
                    st = _subset_state(st, np.searchsorted(st_rows, act))
                    st_rows = act
                if st is None or (stale[act] >= 3).any():
                    Wt = Ma / (1 + pa * Ma)
                    st = solverB.factor(Wt)
                    st_rows = act
                    stale[act] = 0
                    n_fac += len(act)
                    n_fac_sync += len(act)
                D = solverB.solve(st, S)
                # same verdict as the per-gene path's non-finite test: NaN rows from a
                # singular slice, but Inf and NaN too
                bad = ~_rows_finite(D)
                if bad.any():
                    D = D.clone()
                    D[_idx(np.nonzero(bad)[0], D)] = 0.0
            else:
                refresh = np.array([fac[k] is None for k in act], dtype=bool) | (stale[act] >= 3)
                if refresh.any():
                    kk = act[refresh]
                    ir = _idx(np.nonzero(refresh)[0], Ma)
                    Mr = Ma[ir]
                    Wt = Mr / (1 + pa[ir] * Mr)
                    for j, k in enumerate(kk):
                        fac[k] = solver.factor(Wt[j])  # a failed factorisation is None
                    stale[kk] = 0
                    n_fac += len(kk)
                    n_fac_sync += len(act)
                # the step is per gene: its own information, its own right-hand side
                D = torch.zeros(len(act), p, dtype=dt, device=dev)
                bad = np.zeros(len(act), dtype=bool)
                for j, k in enumerate(act):
                    d = solver.solve(fac[k], S[j])
                    if d is None or not bool(torch.isfinite(d).all()):
                        bad[j] = True
                        continue
                    D[j] = d
            if bad.any():
                sing[act[bad]] = True
                act = act[~bad]
                if not len(act):
                    break
                keep = np.nonzero(~bad)[0]
                D = D[_idx(keep, D)]
                if shared_factor:
                    st = _subset_state(st, keep)
                    st_rows = act

            # --- the line search, one trial round for every pending gene ---------------
            na = len(act)
            step = np.ones(na)
            halv = np.zeros(na, dtype=np.int64)
            ok = np.zeros(na, dtype=bool)
            ia = _idx(act, Ai)
            A1 = Ai[ia].clone()
            M1 = Mu[ia].clone()
            L1 = ll[act].copy()
            pend = np.arange(na)
            while len(pend):
                ip = _idx(act[pend], Ai)
                cand = Ai[ip] + as_tensor(step[pend], dev, dt)[:, None] * D[_idx(pend, D)]
                mu_c = mu_of(cand, rows[act[pend]])
                # the one deliberate transfer per trial round: the accept test is
                # host-side control flow over length-genes numbers
                ll_c = host(ll_of(Y[ip], mu_c, pi_[act[pend]], cand))
                ref = ll[act[pend]]
                acc = np.isfinite(ll_c) & (ll_c >= ref - 1e-9 * np.abs(ref))
                if acc.any():
                    take = pend[acc]
                    ia_ = _idx(np.nonzero(acc)[0], cand)
                    it_ = _idx(take, A1)
                    A1[it_] = cand[ia_]
                    M1[it_] = mu_c[ia_]
                    L1[take] = ll_c[acc]
                    ok[take] = True
                fail = pend[~acc]
                step[fail] /= 2
                halv[fail] += 1
                pend = fail[step[fail] > 1e-6]

            # --- what each gene does next ---------------------------------------------
            # a stale matrix can give a bad direction: rebuild once and retry, which costs
            # an iteration, exactly as the per-gene loop does via `continue`
            retry = ~ok & (stale[act] > 0)
            if retry.any():
                kk = act[retry]
                if shared_factor:
                    # one stack: a retry for any gene rebuilds it for all of them
                    Ma2 = Mu[_idx(act, Mu)]
                    pa2 = as_tensor(pi_[act], dev, dt)[:, None]
                    st = solverB.factor(Ma2 / (1 + pa2 * Ma2))
                    st_rows = act
                    stale[act] = 0
                else:
                    ik = _idx(kk, Mu)
                    Mk = Mu[ik]
                    Wt = Mk / (1 + as_tensor(pi_[kk], dev, dt)[:, None] * Mk)
                    for j, k in enumerate(kk):
                        fac[k] = solver.factor(Wt[j])
                    stale[kk] = 0
                n_fac += len(kk)
                n_fac_sync += len(act)
            stop_now = ~ok & ~retry  # line search exhausted

            leave_converged = np.zeros(0, dtype=np.int64)
            if ok.any():
                kk = act[ok]
                iok = _idx(np.nonzero(ok)[0], A1)
                ik = _idx(kk, Ai)
                gain = L1[ok] - ll[kk]
                Ai[ik] = A1[iok]
                Mu[ik] = M1[iok]
                ll[kk] = L1[ok]
                stale[kk] = np.where(halv[ok] > 2, 3, stale[kk] + 1)
                # the convergence test reads the NEW log-likelihood, as the per-gene loop
                # does (`ll = ll1` before `if gain < tol * abs(ll)`)
                done = gain < tol * np.abs(ll[kk])
                conv[kk[done]] = True
                leave_converged = kk[done]

            remove = np.concatenate([act[stop_now], leave_converged])
            act = act[~np.isin(act, remove)]
            act = act[it[act] < maxit]
            if shared_factor and len(act) and not np.array_equal(st_rows, act):
                st = _subset_state(st, np.searchsorted(st_rows, act))
                st_rows = act
        return {"A": Ai, "Mu": Mu, "ll": ll, "it": it, "converged": conv, "singular": sing}

    # --- per-gene helpers, vectorised where they are shared ---------------------------
    def sane_start(rows):
        rows = np.asarray(rows, dtype=np.int64)
        A = torch.zeros(len(rows), p, dtype=dt, device=dev)
        ct = [] if start_cols is None else list(np.nonzero(np.asarray(start_cols, bool))[0])
        # each log mean is net of the offset's row mean over the same cells
        o = offset_rows(offset, rows, A)
        Yr = Yb[_idx(rows, Yb)]
        if ct:
            for j in ct:
                if compact:
                    cells = _block_col_cells(W, int(j))
                else:
                    cells = torch.nonzero(W[:, int(j)] != 0).reshape(-1)
                if len(cells):
                    A[:, j] = torch.log(Yr[:, cells].mean(1) + 1e-3) - (
                        0 if o is None else o[:, cells].mean(1)
                    )
        else:
            A[:, 0] = torch.log(Yr.mean(1) + 1e-3) - (0 if o is None else o.mean(1))
        return A

    def degenerate(rows, A):
        rows = np.asarray(rows, dtype=np.int64)
        out = np.zeros(len(rows), dtype=bool)
        fin = _rows_finite(A)
        out[~fin] = True
        if fin.any():
            kk = np.nonzero(fin)[0]
            Ak = A[_idx(kk, A)]
            Eta = _block_eta_batch(W, Ak) if compact else Ak @ W.T
            o = offset_rows(offset, rows[kk], Eta)
            if o is not None:
                Eta = Eta + o
            pos = Yb[_idx(rows[kk], Yb)] > 0
            # a gene with no positive count has no such cell: Inf, hence not degenerate
            mn = torch.where(pos, Eta, torch.full_like(Eta, math.inf)).min(1).values
            out[kk] = (mn < -10).cpu().numpy()
        return out

    def psi_ml(rows, Mu):
        return psi_profile_batch(Yb[_idx(rows, Yb)], Mu, psi_range)

    # --- the flow, mirroring polish_gene() --------------------------------------------
    alpha = A0.cpu().numpy().astype(np.float64).copy()
    psi = psi0.copy()
    loglik = np.full(B, np.nan)
    iters = np.zeros(B, dtype=np.int64)
    restarted = np.zeros(B, dtype=bool)
    capped = np.zeros(B, dtype=bool)
    singular = np.zeros(B, dtype=bool)
    psi_bound = np.zeros(B, dtype=bool)
    polished = np.zeros(B, dtype=bool)

    def result():
        return _empty_result(
            alpha,
            psi,
            loglik,
            iters,
            restarted,
            capped,
            singular,
            psi_bound,
            polished,
            n_fac,
            n_fac_sync,
        )

    if warm:
        fin = _rows_finite(A0)
        if fin.any():
            rows = np.nonzero(fin)[0]
            f = newton(rows, A0, psi0, maxit)
            good = ~f["singular"] & _rows_finite(f["A"]) & np.isfinite(f["ll"])
            kk = rows[good]
            alpha[kk] = host(f["A"][_idx(np.nonzero(good)[0], f["A"])])
            loglik[kk] = f["ll"][good]
            iters[kk] = f["it"][good]
            capped[kk] = ~f["converged"][good]
            polished[kk] = True
            singular[rows[~good]] = f["singular"][~good]
        return result()

    A = A0.clone()
    deg = degenerate(np.arange(B), A)
    if deg.any():
        rr = np.nonzero(deg)[0]
        A[_idx(rr, A)] = sane_start(rr)
        restarted[deg] = True
    f = newton(np.arange(B), A, psi0, maxit)
    fin = _rows_finite(f["A"])
    mx = f["Mu"].max(1).values.cpu().numpy()
    need = ~restarted & (f["singular"] | ~fin | (mx > 1e10))
    if need.any():
        rows = np.nonzero(need)[0]
        A[_idx(rows, A)] = sane_start(rows)
        restarted[rows] = True
        f2 = newton(rows, A, psi0, maxit)
        # newton() indexes its result by POSITION within `rows`, not by gene id
        ir = _idx(rows, f["A"])
        f["A"][ir] = f2["A"]
        f["Mu"][ir] = f2["Mu"]
        f["ll"][rows] = f2["ll"]
        f["it"][rows] = f2["it"]
        f["converged"][rows] = f2["converged"]
        f["singular"][rows] = f2["singular"]
    fin = _rows_finite(f["A"])
    fell = f["singular"] | ~fin | ~np.isfinite(f["ll"])
    singular[fell] = f["singular"][fell]
    keep = np.nonzero(~fell)[0]
    if not len(keep):
        return result()

    A, Mu, ll, it_total, conv = f["A"], f["Mu"], f["ll"], f["it"], f["converged"]
    ps = psi0.copy()
    if psi_method == "profile":
        live = keep
        for _ in range(2):
            if not len(live):
                break
            pm_psi, at_bound = psi_ml(live, Mu[_idx(live, Mu)])
            psi_bound[live[at_bound]] = True
            hit = at_bound
            live = live[~hit]
            if not len(live):
                break
            ps[live] = host(pm_psi)[~hit]
            f2 = newton(live, A, ps, 20)
            il = _idx(live, A)
            A[il] = f2["A"]
            Mu[il] = f2["Mu"]
            ll[live] = f2["ll"]
            it_total[live] = it_total[live] + f2["it"]
            conv[live] = conv[live] & f2["converged"]
    alpha[keep] = host(A[_idx(keep, A)])
    psi[keep] = ps[keep]
    loglik[keep] = ll[keep]
    iters[keep] = it_total[keep]
    capped[keep] = ~conv[keep]
    polished[keep] = True
    return result()


def _device_of(W, Yb):
    """Device the counts live on: that of ``Yb`` if it is a tensor, else of the design."""
    if isinstance(Yb, torch.Tensor):
        return Yb.device
    return W.device
