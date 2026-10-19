"""Negative-binomial d/p/q (mean/size parameterisation) on torch tensors.

Matches ``stats::dnbinom/pnbinom/qnbinom(mu=, size=)`` on any device, including
``size = Inf`` (Poisson), ``size = 0`` and ``mu = 0`` (point mass at 0). ``size``
may be a per-gene vector with ``len == nrow(mu)``; as in R it is broadcast
row-wise against genes x cells matrices.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from ._backend import DEFAULT_WINSOR, check_winsor

_QFUZZ = 1e-10
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
        torch.lgamma(x + r_d) - torch.lgamma(r_d) - lg_x1
        - r_d * torch.log1p(mu_s / r_d)
        + x * (log_mu - torch.log(r_d + mu_s))
    )
    r_b = torch.where(big, r, torch.full_like(r, _STIRLING_MIN))
    z = x + r_b
    stable = (
        (z - 0.5) * torch.log1p(x / r_b) - x + _stirlerr(z) - _stirlerr(r_b) - lg_x1
        - z * torch.log1p(mu_s / r_b) + x * log_mu
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


def pnbinom(q, mu, size, log: bool = False) -> torch.Tensor:
    """Lower-tail CDF, accumulated as the cumulative sum of the pmf up to ``floor(q)``."""
    qt, mut, st = _prepare(q, mu, size)
    qt, mut, st = torch.broadcast_tensors(qt, mut, st)
    qf = torch.floor(qt)
    finite = torch.isfinite(qf)
    k_max = int(torch.where(finite & (qf >= 0), qf, torch.zeros_like(qf)).max().item())
    res = torch.zeros_like(mut)
    run = torch.zeros_like(mut)
    for k in range(k_max + 1):
        run = run + torch.exp(_logpmf(torch.full_like(mut, k), mut, st))
        res = torch.where(qf == k, run, res)
    res = torch.where(qf == math.inf, torch.ones_like(res), res)
    res = torch.where(torch.isnan(qt) | torch.isnan(mut) | torch.isnan(st), torch.full_like(res, math.nan), res)
    res = torch.clamp(res, max=1.0)
    return torch.log(res) if log else res


def qnbinom(p, mu, size) -> torch.Tensor:
    """Smallest integer ``k`` with CDF(k) >= p (``Inf`` for p = 1, NaN outside [0, 1])."""
    pt, mut, st = _prepare(p, mu, size)
    pt, mut, st = torch.broadcast_tensors(pt, mut, st)
    # R fuzzes the target by 64 eps against rounding in its (exact) CDF. Our CDF is a
    # running sum of pmf terms accurate to ~1e-11 relative (far tails), so the fuzz is
    # widened to 1e-10 to keep qnbinom(pnbinom(k)) == k.
    target = pt * (1 - _QFUZZ)
    res = torch.zeros_like(mut)
    run = torch.zeros_like(mut)
    bad = torch.isnan(pt) | torch.isnan(mut) | torch.isnan(st) | (pt < 0) | (pt > 1)
    done = bad | ((pt == 1) & (mut != 0) & (st != 0))
    k = 0
    while not bool(done.all()):
        run = run + torch.exp(_logpmf(torch.full_like(mut, k), mut, st))
        newly = (run >= target) & ~done
        res = torch.where(newly, torch.full_like(res, k), res)
        done = done | newly
        k += 1
        if k > _QNBINOM_CAP:
            break
    # R returns 0 for a point mass at 0 even when p == 1
    res = torch.where((pt == 1) & (mut != 0) & (st != 0), torch.full_like(res, math.inf), res)
    return torch.where(bad, torch.full_like(res, math.nan), res)


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
        lp = np.log(psi)
        med = np.median(lp)
        psi_max = np.exp(med + winsor * 1.4826 * np.median(np.abs(lp - med)))
    return np.minimum(psi, psi_max)
