"""Negative-binomial d/p/q (mean/size parameterisation) on torch tensors.

Matches ``stats::dnbinom/pnbinom/qnbinom(mu=, size=)`` on any device, including
``size = Inf`` (Poisson), ``size = 0`` and ``mu = 0`` (point mass at 0). ``size``
may be a per-gene vector with ``len == nrow(mu)``; as in R it is broadcast
row-wise against genes x cells matrices.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import torch

from ._backend import DEFAULT_WINSOR, check_winsor

_EPS = float(np.finfo(np.float64).eps)
_QFUZZ = 1e-10
_BETAINC_MAX_SIZE = 1000.0
_SUM_MAX_K = 5000
# Above this size the lgamma difference is evaluated through Stirling's series
# (direct lgamma(x+size) - lgamma(size) loses ~1e-9 relative accuracy at size ~ 1e6).
_STIRLING_MIN = 100.0
_QNBINOM_CAP = 1_000_000  # guards against runaway search for p extremely close to 1


def _stirlerr(z: torch.Tensor) -> torch.Tensor:
    """lgamma(z) - ((z - 0.5) log z - z + 0.5 log(2 pi)) for z >= 100."""
    z2 = z * z
    return (1.0 / 12 - (1.0 / 360 - (1.0 / 1260 - 1.0 / (1680 * z2)) / z2) / z2) / z


def _coerce(x, ref: torch.Tensor | None) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x if ref is None else x.to(device=ref.device, dtype=ref.dtype)
    if ref is None:
        return torch.as_tensor(np.asarray(x, dtype=float))
    return torch.as_tensor(np.asarray(x, dtype=float), dtype=ref.dtype, device=ref.device)


def _prepare(first, mu, size):
    """Coerce to a common device/dtype; a per-row ``size`` vector becomes (nr, 1)."""
    ref = next((t for t in (mu, first, size) if isinstance(t, torch.Tensor)), None)
    if ref is not None and not ref.dtype.is_floating_point:
        ref = ref.to(torch.float64)
    f, m, s = _coerce(first, ref), _coerce(mu, ref), _coerce(size, ref)
    if not f.dtype.is_floating_point:
        f = f.to(torch.float64)
    if not m.dtype.is_floating_point:
        m = m.to(torch.float64)
    if not s.dtype.is_floating_point:
        s = s.to(torch.float64)
    shape_ref = m if m.dim() >= f.dim() else f
    if s.dim() == 1 and shape_ref.dim() == 2 and s.shape[0] == shape_ref.shape[0]:
        s = s.view(-1, 1)
    return f, m, s


def _logpmf(x: torch.Tensor, mu: torch.Tensor, size: torch.Tensor) -> torch.Tensor:
    x, mu, r = torch.broadcast_tensors(x, mu, size)
    inf_r = torch.isinf(r)
    big = (r >= _STIRLING_MIN) & ~inf_r
    point_mass = (mu == 0) | (r == 0)
    mu_s = torch.where(point_mass, torch.ones_like(mu), mu)
    lg_x1 = torch.lgamma(x + 1)
    log_mu = torch.log(mu_s)

    r_d = torch.where(big | inf_r | (r == 0), torch.ones_like(r), r)
    direct = (
        torch.lgamma(x + r_d)
        - torch.lgamma(r_d)
        - lg_x1
        - r_d * torch.log1p(mu_s / r_d)
        + x * (log_mu - torch.log(r_d + mu_s))
    )
    r_b = torch.where(big, r, torch.full_like(r, _STIRLING_MIN))
    z = x + r_b
    stable = (
        (z - 0.5) * torch.log1p(x / r_b)
        - x
        + _stirlerr(z)
        - _stirlerr(r_b)
        - lg_x1
        - z * torch.log1p(mu_s / r_b)
        + x * log_mu
    )
    pois = x * log_mu - mu_s - lg_x1

    lp = torch.where(inf_r, pois, torch.where(big, stable, direct))
    zero = torch.zeros_like(lp)
    lp = torch.where(point_mass, torch.where(x == 0, zero, torch.full_like(lp, -math.inf)), lp)
    invalid = (x < 0) | (x != torch.floor(x))
    return torch.where(invalid & ~torch.isnan(x), torch.full_like(lp, -math.inf), lp)


def dnbinom(x, mu, size, log: bool = False) -> torch.Tensor:
    """Density of the negative binomial with mean ``mu`` and size ``size``."""
    xt, mut, st = _prepare(x, mu, size)
    lp = _logpmf(xt, mut, st)
    return lp if log else torch.exp(lp)


def _np_inputs(first, mu, size):
    """Broadcast float64 numpy views of the inputs plus the common dtype/device."""
    ft, mt, st = _prepare(first, mu, size)
    ft, mt, st = torch.broadcast_tensors(ft, mt, st)
    return (
        ft.detach().cpu().numpy().astype(float),
        mt.detach().cpu().numpy().astype(float),
        st.detach().cpu().numpy().astype(float),
        mt,
    )


def _nb_params(mu, size):
    """Masks and safe scipy parameters: point mass at 0 (mu == 0 or size == 0), Poisson (size = Inf)."""
    point = (mu == 0) | (size == 0)
    pois = np.isinf(size) & ~point
    ok = ~point & ~pois
    n = np.where(ok, size, 1.0)
    n_safe = np.where(ok, size, 1.0)
    pr = np.where(ok, n_safe / (n_safe + np.where(ok, mu, 1.0)), 0.5)
    lam = np.where(pois, mu, 1.0)
    return point, pois, n, pr, lam


def _cdf_sf_np(k, mu, size):
    """Exact CDF and survival function at integer ``k >= 0`` (numpy, broadcast inputs).

    Both tails are evaluated directly with the incomplete beta: forming 1 - p for
    p = size/(size+mu) would lose all precision when mu << size.
    """
    from scipy import special as sp

    point, pois, n, pr, lam = _nb_params(mu, size)
    nopar = pois | point
    qp = np.where(nopar, 0.5, mu) / (n + np.where(nopar, 1.0, mu))
    with np.errstate(all="ignore"):
        cdf = np.where(pois, sp.pdtr(k, lam), sp.betainc(n, k + 1, pr))
        sf = np.where(pois, sp.pdtrc(k, lam), sp.betainc(k + 1, n, qp))
    # scipy's betainc loses accuracy for large size (5e-11 relative at 1e6); there the
    # CDF is summed from the stable pmf instead (bounded number of terms).
    ok_sum = (
        (~point) & (~pois) & (size >= _BETAINC_MAX_SIZE) & np.isfinite(size) & (k <= _SUM_MAX_K)
    )
    if ok_sum.any():
        idx = np.nonzero(ok_sum)
        kk = k[idx].astype(np.longdouble)
        mm = mu[idx].astype(np.longdouble)
        rr = size[idx].astype(np.longdouble)
        # pmf(0) and the ratio pmf(j+1)/pmf(j) = (j + r)/(j + 1) * mu/(r + mu), evaluated in
        # extended precision so the far lower tail is exact to ~1e-15 (x86; plain double elsewhere)
        pmf = np.exp(-rr * np.log1p(mm / rr))
        ratio = mm / (rr + mm)
        total = pmf.copy()
        for j in range(int(kk.max())):
            pmf = pmf * ((j + rr) / (j + 1)) * ratio
            total = total + np.where(kk >= j + 1, pmf, 0)
        cdf = cdf.copy()
        cdf[idx] = total.astype(float)
    return np.where(point, 1.0, cdf), np.where(point, 0.0, sf)


def _pnbinom_cpu(q, mu, size, log):
    """Exact CDF (regularised incomplete beta), like R's stats::pnbinom."""
    qf, mu, size, ref = _np_inputs(q, mu, size)
    k = np.where(np.isfinite(qf), np.floor(np.maximum(qf, 0.0)), 0.0)
    cdf, sf = _cdf_sf_np(k, mu, size)
    with np.errstate(all="ignore"):
        neg, top = qf < 0, qf == np.inf
        if log:
            res = np.where(cdf > 0.5, np.log1p(-sf), np.log(cdf))
            res = np.where(neg, -np.inf, np.where(top, 0.0, res))
        else:
            res = np.where(neg, 0.0, np.where(top, 1.0, cdf))
        res = np.where(np.isnan(qf) | np.isnan(mu) | np.isnan(size), np.nan, res)
    return torch.as_tensor(res, dtype=ref.dtype, device=ref.device)


def _qnbinom_cpu(p, mu, size):
    """Exact quantile: smallest k with CDF(k) >= p, as R, with R's left-continuity fuzz.

    scipy's ppf gives the starting point; it is then corrected against the exact
    CDF (scipy's own CDF loses precision for mu << size). The fuzz is
    ``p * (1 - 64 eps)`` on the CDF, except in the far upper tail (1 - p < 1e-6),
    where the search runs on the survival function, ``sf(k) <= (1 - p) + 8 eps p``.
    R >= 4.2 fuzzes the upper tail much less than 64 eps; a 64 eps fuzz is a large
    fraction of a ~1e-10 tail and moves the quantile by several counts. (Further
    from 1 the CDF R feeds back through p carries errors up to ~1e-14 that only a
    64 eps fuzz absorbs, so that regime is unchanged.)
    """
    import scipy.stats as ss

    pv, mu, size, ref = _np_inputs(p, mu, size)
    point, pois, n, pr, lam = _nb_params(mu, size)
    bad = np.isnan(pv) | np.isnan(mu) | np.isnan(size) | (pv < 0) | (pv > 1)
    target = np.where(bad, 0.5, pv * (1 - 64 * _EPS))
    upper = ~bad & (1.0 - pv < 1e-6)
    sf_target = np.where(upper, (1.0 - pv) + 8 * _EPS * pv, 0.0)

    def reached(k):
        cdf, sf = _cdf_sf_np(k, mu, size)
        return np.where(upper, sf <= sf_target, cdf >= target)

    with np.errstate(all="ignore"):
        k = np.where(pois, ss.poisson.ppf(target, lam), ss.nbinom.ppf(target, n, pr))
        qu = np.where(upper, 1.0 - pv, 0.5)
        k_up = np.where(pois, ss.poisson.isf(qu, lam), ss.nbinom.isf(qu, n, pr))
        k = np.where(upper & np.isfinite(k_up), k_up, k)
        k = np.where(np.isfinite(k), np.maximum(k, 0.0), 0.0)
        todo = ~(bad | point | (pv == 1))
        for _ in range(10_000):
            up = todo & ~reached(k)
            down = todo & (k > 0) & reached(np.maximum(k - 1, 0.0))
            if not (up | down).any():
                break
            k = k + up - down
        res = np.where(pv == 1, np.inf, k)
        res = np.where(point, 0.0, res)
        res = np.where(bad, np.nan, res)
    return torch.as_tensor(res, dtype=ref.dtype, device=ref.device)


def _pnbinom_loop(q, mu, size, log=False):
    """Accelerator path: cumulative sum of the pmf up to ``floor(q)`` (as R's GPU code)."""
    qt, mut, st = _prepare(q, mu, size)
    qt, mut, st = torch.broadcast_tensors(qt, mut, st)
    if mut.numel() == 0:
        return torch.zeros_like(mut)
    qf = torch.floor(qt)
    k_max = int(torch.where(torch.isfinite(qf) & (qf >= 0), qf, torch.zeros_like(qf)).max().item())
    res = torch.zeros_like(mut)
    run = torch.zeros_like(mut)
    for k in range(k_max + 1):
        run = run + torch.exp(_logpmf(torch.full_like(mut, k), mut, st))
        res = torch.where(qf == k, run, res)
    res = torch.where(qf == math.inf, torch.ones_like(res), res)
    res = torch.where(
        torch.isnan(qt) | torch.isnan(mut) | torch.isnan(st), torch.full_like(res, math.nan), res
    )
    res = torch.clamp(res, max=1.0)
    return torch.log(res) if log else res


def _qnbinom_loop(p, mu, size):
    """Accelerator path: accumulate the pmf until every element crosses its probability."""
    pt, mut, st = _prepare(p, mu, size)
    pt, mut, st = torch.broadcast_tensors(pt, mut, st)
    if mut.numel() == 0:
        return torch.zeros_like(mut)
    # R fuzzes the target by 64 eps against rounding in its (exact) CDF. The summed CDF is
    # accurate to ~1e-11 relative in far tails, so the fuzz is widened to 1e-10 here
    # (docs/r-discrepancies.md).
    target = pt * (1 - _QFUZZ)
    res = torch.zeros_like(mut)
    run = torch.zeros_like(mut)
    bad = torch.isnan(pt) | torch.isnan(mut) | torch.isnan(st) | (pt < 0) | (pt > 1)
    inf_q = (pt == 1) & (mut != 0) & (st != 0)  # R returns 0 for a point mass even at p == 1
    done = bad | inf_q
    k = 0
    while not bool(done.all()):
        run = run + torch.exp(_logpmf(torch.full_like(mut, k), mut, st))
        newly = (run >= target) & ~done
        res = torch.where(newly, torch.full_like(res, k), res)
        done = done | newly
        k += 1
        if k > _QNBINOM_CAP:
            warnings.warn(
                f"qnbinom search reached the cap of {_QNBINOM_CAP} without converging; "
                "unfinished elements are NaN",
                stacklevel=2,
            )
            res = torch.where(done, res, torch.full_like(res, math.nan))
            break
    res = torch.where(inf_q, torch.full_like(res, math.inf), res)
    return torch.where(bad, torch.full_like(res, math.nan), res)


def _on_cpu(*xs) -> bool:
    return all(not isinstance(x, torch.Tensor) or x.device.type == "cpu" for x in xs)


def pnbinom(q, mu, size, log: bool = False) -> torch.Tensor:
    """Lower-tail CDF. Exact (scipy) on CPU; pmf cumulative sum on accelerators."""
    if _on_cpu(q, mu, size):
        return _pnbinom_cpu(q, mu, size, log)
    return _pnbinom_loop(q, mu, size, log)


def qnbinom(p, mu, size) -> torch.Tensor:
    """Smallest integer ``k`` with CDF(k) >= p (``Inf`` for p = 1, NaN outside [0, 1])."""
    if _on_cpu(p, mu, size):
        return _qnbinom_cpu(p, mu, size)
    return _qnbinom_loop(p, mu, size)


def winsorise_psi(psi, winsor: float = DEFAULT_WINSOR, psi_max: float | None = None) -> np.ndarray:
    """Cap dispersions at ``exp(median(log psi) + winsor * MAD(log psi))``.

    ``psi_max`` supplies a precomputed global threshold (R's ``psi.max``
    attribute, used when normalising in gene blocks); ``winsor=inf`` disables it.
    """
    check_winsor(winsor)
    psi = np.asarray(psi, dtype=float)
    if math.isinf(winsor):
        return psi
    if psi_max is None:
        with np.errstate(all="ignore"):  # psi = 0 gives log = -Inf, silently as in R
            lp = np.log(psi)
            med = np.median(lp)
            psi_max = np.exp(med + winsor * 1.4826 * np.median(np.abs(lp - med)))
    return np.minimum(psi, psi_max)
