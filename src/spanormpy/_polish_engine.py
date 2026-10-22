"""The per-gene polish engine (port of ``R/polishEngine.R``).

Damped Newton on one gene's own penalised negative binomial log-likelihood, with a
profile-ML dispersion. It is the reference implementation (``engine="gene"``) and
the oracle for the batched engine in :mod:`spanormpy._polish_batch`.

Everything here is torch on the device of ``W`` (the CPU: the per-gene engine keeps
one factorisation object per gene). Column indices are 0-based; ``start_cols`` is a
boolean vector over the columns of ``W``.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from ._backend import as_tensor
from ._block_design import _design_col_nonzero, _design_cross, _design_eta
from ._nbdist import dnbinom
from ._settings import _match_arg

# The fitted mean is floored here and at inference, at the same value. A linear
# predictor below about -745 underflows exp() to exactly 0 and the NB quantities
# built from it divide by zero; exp(-30) is far below any mean the model can
# meaningfully estimate, so this is a numerical guard, not a statistical clamp.
MU_FLOOR = math.exp(-30)

_DBL_EPS = float(np.finfo(np.float64).eps)
_DBL_MAX = float(np.finfo(np.float64).max)


def nb_mu_floor() -> float:
    """The floor on a fitted negative binomial mean, ``exp(-30)`` (R ``nbMuFloor``).

    The polish engine floors every fitted mean at this value; a caller that rebuilds
    the mean from polished coefficients should floor at the same value.

    Examples
    --------
    >>> import spanormpy as sn
    >>> sn.nb_mu_floor()  # smallest fitted mean used by the polishing engine
    """
    return MU_FLOOR


def _size_of(psi: float) -> float:
    """``1 / psi`` as R computes it: ``psi = 0`` is ``Inf`` (the Poisson limit)."""
    return math.inf if psi == 0 else 1.0 / psi


def nb_pen_loglik(y: torch.Tensor, mu: torch.Tensor, psi: float, a: torch.Tensor, pen) -> float:
    """Penalised NB log-likelihood of one gene (R ``.nbPenLoglik``)."""
    ll = dnbinom(y, mu=mu, size=_size_of(psi), log=True).sum()
    return float(ll - 0.5 * (pen * a * a).sum())


def brent_fmin(f, ax: float, bx: float, tol: float = _DBL_EPS**0.25) -> float:
    """Brent's one-dimensional minimiser, a line-for-line port of R's ``Brent_fmin``.

    R's ``optimize()`` is this routine; it is reproduced rather than replaced by
    ``scipy.optimize.fminbound`` because the per-gene engine's dispersion is only
    determined to this routine's tolerance (~1e-4 on ``log psi``), and matching R to
    ``1e-6`` needs the same sequence of evaluations. A non-finite objective is
    replaced by the largest double, as R's ``fmin`` does.
    """
    c = (3.0 - math.sqrt(5.0)) * 0.5
    eps = math.sqrt(_DBL_EPS)
    a, b = ax, bx
    v = a + c * (b - a)
    w = v
    x = v
    d = 0.0
    e = 0.0

    def fn(t):
        val = f(t)
        return val if math.isfinite(val) else _DBL_MAX

    fx = fn(x)
    fv = fx
    fw = fx
    tol3 = tol / 3.0
    while True:
        xm = (a + b) * 0.5
        tol1 = eps * abs(x) + tol3
        t2 = tol1 * 2.0
        if abs(x - xm) <= t2 - (b - a) * 0.5:
            break
        p = q = r = 0.0
        if abs(e) > tol1:  # fit parabola
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p = (x - v) * q - (x - w) * r
            q = (q - r) * 2.0
            if q > 0.0:
                p = -p
            else:
                q = -q
            r = e
            e = d
        if abs(p) >= abs(q * 0.5 * r) or p <= q * (a - x) or p >= q * (b - x):
            e = (b - x) if x < xm else (a - x)  # golden-section step
            d = c * e
        else:  # parabolic-interpolation step
            d = p / q
            u = x + d
            if u - a < t2 or b - u < t2:
                d = tol1 if x < xm else -tol1
        if abs(d) >= tol1:
            u = x + d
        elif d > 0.0:
            u = x + tol1
        else:
            u = x - tol1
        fu = fn(u)
        if fu <= fx:
            if u < x:
                b = x
            else:
                a = x
            v, w, x = w, x, u
            fv, fw, fx = fw, fx, fu
        else:
            if u < x:
                a = u
            else:
                b = u
            if fu <= fw or w == x:
                v, fv = w, fw
                w, fw = u, fu
            elif fu <= fv or v == x or v == w:
                v, fv = u, fu
    return x


def polish_gene(
    y,
    W,
    a0,
    psi0: float,
    pen,
    solver,
    maxit: int = 50,
    tol: float = 1e-8,
    start_cols=None,
    psi_range=(1e-3, 1e3),
    psi_method: str = "profile",
    warm: bool = False,
    offset=None,
) -> dict:
    """Converge one gene to its own penalised NB optimum (R ``.polishGene``).

    Damped Newton at fixed ``psi``, then profile ML for ``psi`` at the converged
    mean and a re-polish, twice. The information matrix is reused for up to three
    consecutive steps. A degenerate start (a fitted log-mean below -10 at a cell
    with a count) restarts from a sane point: the log mean over the cells of each
    ``start_cols`` column. A gene that cannot be polished keeps its input fit.

    Parameters
    ----------
    y
        counts of this gene (length n).
    W
        the design: a dense ``n x p`` tensor or an ``NBBlockDesign``.
    a0, psi0
        starting coefficients (length p) and dispersion.
    pen
        length-p ridge penalty tensor.
    solver
        a ``NewtonSolver`` for this ``W`` and ``pen``.
    start_cols
        boolean vector over the columns of ``W`` marking the cell-type intercepts.
    warm
        ``a0``/``psi0`` are a converged fit at a nearby penalty: a few Newton steps
        at the held dispersion, no dispersion search and no restart check.
    offset
        ``None`` or a per-cell log-scale offset (length n).

    Returns
    -------
    dict
        ``alpha`` (numpy, length p), ``psi``, ``loglik`` (NaN if not polished),
        ``iterations``, ``restarted``, ``capped``, ``singular``, ``psi_bound``,
        ``polished``.
    """
    psi_method = _match_arg(psi_method, ("profile", "fixed"), "psi_method")
    dev, dt = pen.device, pen.dtype
    y = as_tensor(y, dev, dt).reshape(-1)
    n = y.numel()
    if offset is not None and np.asarray(offset).size != n:
        raise ValueError("'offset' must have one value per cell")
    off = (
        torch.zeros(n, dtype=dt, device=dev)
        if offset is None
        else as_tensor(offset, dev, dt).reshape(-1)
    )
    p = int(W.shape[1])
    state = {"singular": False}
    lo, hi = math.log(psi_range[0]), math.log(psi_range[1])

    def eta_mu(a):
        return torch.clamp(torch.exp(_design_eta(W, a) + off), min=MU_FLOOR)

    def sane_start():
        # With an offset each log mean is net of the mean offset over the same cells.
        a = torch.zeros(p, dtype=dt, device=dev)
        ct = [] if start_cols is None else list(np.nonzero(np.asarray(start_cols, bool))[0])
        if ct:
            for j in ct:
                cells = _design_col_nonzero(W, int(j))
                if bool(cells.any()):
                    a[j] = math.log(float(y[cells].mean()) + 1e-3) - float(off[cells].mean())
        else:
            a[0] = math.log(float(y.mean()) + 1e-3) - float(off.mean())
        return a

    def newton(a, psi, maxit):
        mu = eta_mu(a)
        ll = nb_pen_loglik(y, mu, psi, a, pen)
        it = 0
        converged = False
        stale = 0
        fac = None
        have_fac = False
        while it < maxit:
            it += 1
            s = _design_cross(W, (y - mu) / (1 + psi * mu)) - pen * a
            if not have_fac or stale >= 3:
                # Factor here and only here: the weights are what the information
                # depends on, and between refreshes the same factorisation is exact.
                fac = solver.factor(mu / (1 + psi * mu))
                have_fac = True
                stale = 0
            d = solver.solve(fac, s)
            # solve() only errors below rcond ~1e-7; between that and well-conditioned
            # it returns a finite but numerically meaningless answer. A non-finite
            # right-hand side does not error either: treat both as singular.
            if d is None or not bool(torch.isfinite(d).all()):
                state["singular"] = True
                break
            step = 1.0
            ok = False
            halvings = 0
            while step > 1e-6:
                a1 = a + step * d
                mu1 = eta_mu(a1)
                ll1 = nb_pen_loglik(y, mu1, psi, a1, pen)
                if math.isfinite(ll1) and ll1 >= ll - 1e-9 * abs(ll):
                    ok = True
                    break
                step /= 2
                halvings += 1
            if not ok:
                # a stale information matrix can give a bad direction; rebuild it
                # once before giving up
                if stale > 0:
                    fac = solver.factor(mu / (1 + psi * mu))
                    stale = 0
                    continue
                break
            stale = 3 if halvings > 2 else stale + 1
            gain = ll1 - ll
            a, mu, ll = a1, mu1, ll1
            if gain < tol * abs(ll):
                converged = True
                break
        return {"a": a, "mu": mu, "ll": ll, "it": it, "converged": converged}

    def psi_ml(mu):
        # The optimiser needs a bounded interval, so an under-dispersed or near-empty
        # gene lands ON a bound; report it rather than storing it as an estimate.
        def obj(lp):
            return -float(dnbinom(y, mu=mu, size=1.0 / math.exp(lp), log=True).sum())

        m = brent_fmin(obj, lo, hi)
        edge = (m - lo) < 1e-3 * (hi - lo) or (hi - m) < 1e-3 * (hi - lo)
        return math.exp(m), edge

    a0t = as_tensor(a0, dev, dt).reshape(-1)

    def fallback(why):
        # Hand back the input fit, NOT the sane start: the sane start would replace a
        # usable fit with cell-type log means and exact zeros on every tested coefficient.
        return {
            "alpha": a0t.cpu().numpy().copy(),
            "psi": float(psi0),
            "loglik": math.nan,
            "iterations": 0,
            "restarted": False,
            "capped": False,
            "singular": why == "singular",
            "psi_bound": False,
            "polished": False,
        }

    def degenerate(a):
        # A degenerate START is a fitted log-mean below -10 at a cell that has a count;
        # at a zero-count cell it is the likelihood's own direction.
        if not bool(torch.isfinite(a).all()):
            return True
        lp = _design_eta(W, a) + off
        pos = y > 0
        return bool(pos.any()) and float(lp[pos].min()) < -10

    a = a0t
    psi = float(psi0)
    restarted = False
    if warm:
        if not bool(torch.isfinite(a).all()):
            return fallback("nonfinite")
        f = newton(a, psi, maxit)
        if (
            state["singular"]
            or not bool(torch.isfinite(f["a"]).all())
            or not math.isfinite(f["ll"])
        ):
            return fallback("singular" if state["singular"] else "nonfinite")
        return {
            "alpha": f["a"].cpu().numpy(),
            "psi": psi,
            "loglik": f["ll"],
            "iterations": f["it"],
            "restarted": False,
            "capped": not f["converged"],
            "singular": False,
            "psi_bound": False,
            "polished": True,
        }
    if degenerate(a):
        a = sane_start()
        restarted = True
    f = newton(a, psi, maxit)
    if not restarted and (
        state["singular"] or not bool(torch.isfinite(f["a"]).all()) or float(f["mu"].max()) > 1e10
    ):
        state["singular"] = False
        restarted = True
        f = newton(sane_start(), psi, maxit)
    # both starts failed: keep the input fit rather than an unconverged guess
    if state["singular"] or not bool(torch.isfinite(f["a"]).all()) or not math.isfinite(f["ll"]):
        return fallback("singular" if state["singular"] else "nonfinite")
    converged = f["converged"]
    it_total = f["it"]
    psi_bound = False
    # "fixed" keeps the input dispersion and converges only the mean under it;
    # "profile" re-estimates psi by profile ML at the converged mean and re-polishes, twice
    if psi_method == "profile":
        for _ in range(2):
            psi_new, at_bound = psi_ml(f["mu"])
            if at_bound:
                psi_bound = True  # not identified for this gene; keep the input value
                break
            psi = psi_new
            f2 = newton(f["a"], psi, 20)
            it_total += f2["it"]
            converged = converged and f2["converged"]
            f = f2
    return {
        "alpha": f["a"].cpu().numpy(),
        "psi": psi,
        "loglik": f["ll"],
        "iterations": it_total,
        "restarted": restarted,
        "capped": not converged,
        "singular": state["singular"],
        "psi_bound": psi_bound,
        "polished": True,
    }
