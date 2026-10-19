"""Quasi-likelihood dispersion for per-gene NB GLMs (port of ``R/quasilik.R``).

The edgeR v4 adjusted deviance and effective df are obtained from the first two
moments of the NB unit deviance, matched to a scaled chi-square: with
``d ~ c * chisq_nu``, ``w1 = nu = 2 E[d]^2 / Var[d]`` is the observation's
effective df and ``w0 = 1/c = 2 E[d] / Var[d]`` rescales its unit deviance.
The moments depend only on ``(mu, phi)``, so :func:`ql_dispersion` evaluates
them once on a shared ``(log mu, log phi)`` table and bilinearly interpolates.

Everything is genes x cells (R orientation). ``numpy`` inputs give ``numpy``
outputs. ``torch`` tensors (any device) are supported for
``leverage="trace"`` / ``moments="table"``, as in R; the moment table itself is
always built on the CPU.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from ._nbdist import qnbinom

_DOUBLE_XMIN = float(np.finfo(np.float64).tiny)
_STIRLING_MIN = 100.0  # as _nbdist: Stirling form of the pmf for size >= 100
_CACHE_MAX = 4_000_000  # max (terms x elements) kept between the two summation passes


def _is_tensor(*xs) -> bool:
    return any(isinstance(x, torch.Tensor) for x in xs)


def _np(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _num(x) -> np.ndarray:
    return np.asarray(_np(x), dtype=float)


def _to_tensor(x, ref: torch.Tensor) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=ref.device, dtype=ref.dtype)
    return torch.as_tensor(np.asarray(x, dtype=float), dtype=ref.dtype, device=ref.device)


def _rep_len(x, n: int) -> np.ndarray:
    """R's ``rep_len(as.numeric(x), n)``."""
    x = np.asarray(_np(x), dtype=float).ravel()
    return np.resize(x, n)


def nb_unit_deviance(y, mu, phi):
    """NB unit deviance ``2{y log(y/mu) - (y + 1/phi) log((y + 1/phi)/(mu + 1/phi))}``.

    Elementwise over a genes x cells matrix; ``phi`` is a scalar or one value per
    gene (row). 1-D ``y``/``mu`` are treated as one gene. A torch tensor input
    gives a tensor on its device.
    """
    if _is_tensor(y, mu):
        ref = mu if isinstance(mu, torch.Tensor) else y
        yt, mut = _to_tensor(y, ref), _to_tensor(mu, ref)
        phi_a = _num(phi)
        size = torch.as_tensor(1.0 / phi_a, dtype=ref.dtype, device=ref.device)
        if size.dim() > 0 and size.numel() > 1 and yt.dim() == 2:
            size = size.view(-1, 1)
        pos = yt > 0
        ysafe = torch.where(pos, yt, torch.ones_like(yt))
        a = torch.where(pos, yt * (torch.log(ysafe) - torch.log(mut)), torch.zeros_like(yt))
        return 2 * (a - (yt + size) * (torch.log(yt + size) - torch.log(mut + size)))
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    phi_a = np.asarray(phi, dtype=float).ravel()
    if y.ndim == 2:
        size = (1.0 / np.resize(phi_a, y.shape[0]))[:, None]
    else:
        size = 1.0 / phi_a[0]
    with np.errstate(all="ignore"):
        a = np.where(y > 0, y * np.log(np.maximum(y, _DOUBLE_XMIN) / mu), 0.0)
        return 2 * (a - (y + size) * np.log((y + size) / (mu + size)))


def _unit_integer(i: np.ndarray, m: np.ndarray, size: np.ndarray) -> np.ndarray:
    """Unit deviance at integer support points ``i`` (R's inner ``unit`` closure)."""
    with np.errstate(all="ignore"):
        a = np.where(i > 0, i * np.log(np.maximum(i, 1.0) / m), 0.0)
        return 2 * (a - (i + size) * np.log((i + size) / (m + size)))


def _stirlerr(z: np.ndarray) -> np.ndarray:
    z2 = z * z
    return (1.0 / 12 - (1.0 / 360 - (1.0 / 1260 - 1.0 / (1680 * z2)) / z2) / z2) / z


def _dnb(i: np.ndarray, size: np.ndarray, m: np.ndarray) -> np.ndarray:
    """``dnbinom(i, size, mu=m)`` for integer ``i >= 0``, ``m > 0``, finite ``size > 0``.

    The same formulas as :func:`spanormpy._nbdist.dnbinom` (direct ``lgamma`` form,
    Stirling form once ``size >= 100``) in plain numpy, which is several times
    faster for the many small calls the moment summation makes.
    """
    from scipy.special import gammaln

    lg_x1 = gammaln(i + 1.0)
    log_m = np.log(m)
    lp = np.empty_like(m)
    big = size >= _STIRLING_MIN
    sm = ~big
    if sm.any():
        x, r, mm = i[sm], size[sm], m[sm]
        lp[sm] = (
            gammaln(x + r) - gammaln(r) - lg_x1[sm] - r * np.log1p(mm / r) + x * (log_m[sm] - np.log(r + mm))
        )
    if big.any():
        x, r, mm = i[big], size[big], m[big]
        z = x + r
        lp[big] = (
            (z - 0.5) * np.log1p(x / r)
            - x
            + _stirlerr(z)
            - _stirlerr(r)
            - lg_x1[big]
            - z * np.log1p(mm / r)
            + x * log_m[big]
        )
    return np.exp(lp)


def nb_deviance_moments(mu, phi, eps: float = 1e-10, maxterms: int = 20000) -> dict:
    """First two moments of the NB unit deviance, by direct summation over the pmf.

    Returns ``{"w0", "w1"}`` (deviance rescaling and effective df), each the length
    of ``mu``; invalid entries (``mu <= 1e-32``, non-finite or non-positive
    ``phi``) get 0. ``phi`` is a scalar or the same length as ``mu``.
    """
    mu = _num(mu).ravel()
    phi = _rep_len(phi, mu.size)
    w0 = np.zeros(mu.size)
    w1 = np.zeros(mu.size)
    with np.errstate(invalid="ignore"):
        ok = np.nonzero(np.isfinite(mu) & (mu > 1e-32) & np.isfinite(phi) & (phi > 0))[0]
    if ok.size == 0:
        return {"w0": w0, "w1": w1}

    mu_ok, phi_ok = mu[ok], phi[ok]
    sz = 1.0 / phi_ok
    t_mu, t_sz = torch.as_tensor(mu_ok), torch.as_tensor(sz)
    lo = qnbinom(torch.full_like(t_mu, eps), t_mu, t_sz).numpy()
    hi = qnbinom(torch.full_like(t_mu, 1 - eps), t_mu, t_sz).numpy()
    wid = np.minimum(maxterms, np.maximum(10.0, hi - lo))
    key = np.minimum(30, np.ceil(np.log2(wid))).astype(int)

    for k in np.unique(key):
        ix = np.nonzero(key == k)[0]
        m, size = mu_ok[ix], 1.0 / phi_ok[ix]
        lg = lo[ix]
        kmax = int(wid[ix].max())
        ed = np.zeros(ix.size)
        vd = np.zeros(ix.size)
        cache = (kmax + 1) * ix.size <= _CACHE_MAX
        kept = []
        for j in range(kmax + 1):  # pass 1: E[d]
            i = lg + j
            pj, dj = _dnb(i, size, m), _unit_integer(i, m, size)
            ed = ed + pj * dj
            if cache:
                kept.append((pj, dj))
        for j in range(kmax + 1):  # pass 2: Var[d]
            if cache:
                pj, dj = kept[j]
            else:
                i = lg + j
                pj, dj = _dnb(i, size, m), _unit_integer(i, m, size)
            vd = vd + pj * (dj - ed) ** 2
        with np.errstate(all="ignore"):
            good = np.isfinite(ed) & np.isfinite(vd) & (vd > 0)
            tgt = ok[ix][good]
            w0[tgt] = (2 * ed / vd)[good]
            w1[tgt] = (2 * ed * ed / vd)[good]
    return {"w0": w0, "w1": w1}


def ql_moment_table(
    lmu_range, lphi_range, step_mu: float = 0.08, step_phi: float = 0.12, ngrid=None, nphi=None
) -> dict:
    """Deviance moments ``w0``/``w1`` on a uniform (log phi, log mu) grid.

    ``w0``/``w1`` are ``J x K`` (log-phi rows, log-mu columns); ``lo``, ``h``,
    ``K`` describe the log-mu axis and ``plo``, ``ph``, ``J`` the log-phi axis.
    """
    lmu_range = _num(lmu_range).ravel()
    lo = float(lmu_range.min())
    hi = max(float(lmu_range.max()), lo + 1e-8)
    K = max(2, math.ceil((hi - lo) / step_mu) + 1) if ngrid is None else int(ngrid)
    kn = np.linspace(lo, hi, K)
    lphi_range = _num(lphi_range).ravel()
    p0, p1 = float(lphi_range.min()), float(lphi_range.max())
    if p1 - p0 < 1e-12:
        J = 1
    elif nphi is None:
        J = max(2, math.ceil((p1 - p0) / step_phi) + 1)
    else:
        J = int(nphi)
    lp = np.array([p0]) if J == 1 else np.linspace(p0, p1, J)
    mk = nb_deviance_moments(np.tile(np.exp(kn), J), np.repeat(np.exp(lp), K))
    return {
        "w0": mk["w0"].reshape(J, K),
        "w1": mk["w1"].reshape(J, K),
        "lo": lo,
        "h": (hi - lo) / (K - 1),
        "K": K,
        "plo": float(lp[0]),
        "ph": float((lp[-1] - lp[0]) / (J - 1)) if J > 1 else 1.0,
        "J": J,
    }


def _ql_moment_table(lmu_range, lphi, ngrid, nphi, step_phi: float = 0.1) -> dict:
    """The per-call table :func:`ql_dispersion` builds when none is supplied."""
    lphi = _num(lphi)
    pr = (float(lphi.min()), float(lphi.max()))
    if pr[1] - pr[0] < 1e-12:
        J = 1
    else:
        J = min(nphi, max(2, math.ceil((pr[1] - pr[0]) / step_phi) + 1))
    return ql_moment_table(lmu_range, pr, ngrid=ngrid, nphi=J)


def _ql_interp(tab: dict, lmu, lphi) -> dict:
    """Row-wise-phi bilinear lookup of ``w0``/``w1`` with edge clamping (R's rule 2)."""
    K, J = tab["K"], tab["J"]
    lphi = _num(lphi).ravel()
    if J > 1:
        xp = np.minimum(np.maximum((lphi - tab["plo"]) / tab["ph"], 0), J - 1)
        jp = np.minimum(np.floor(xp), max(J - 2, 0))
        fp = xp - jp
    else:
        jp = np.zeros(lphi.size)
        fp = np.zeros(lphi.size)

    if isinstance(lmu, torch.Tensor):
        x = torch.clamp((lmu - tab["lo"]) / tab["h"], min=0, max=K - 1)
        i = torch.clamp(torch.floor(x), min=0, max=K - 2)
        fx = x - i
        i = i.long()
        jt = torch.as_tensor(jp, device=lmu.device).long().view(-1, 1)
        j1 = torch.clamp(jt + 1, max=J - 1)
        fpt = torch.as_tensor(fp, dtype=lmu.dtype, device=lmu.device).view(-1, 1)

        def lerp(M):
            M = torch.as_tensor(M, dtype=lmu.dtype, device=lmu.device)
            g = lambda jj, ii: M[jj.expand_as(ii), ii]  # noqa: E731
            return (1 - fpt) * ((1 - fx) * g(jt, i) + fx * g(jt, i + 1)) + fpt * (
                (1 - fx) * g(j1, i) + fx * g(j1, i + 1)
            )

        return {"w0": lerp(tab["w0"]), "w1": lerp(tab["w1"])}

    lmu = np.asarray(lmu, dtype=float)
    x = np.minimum(np.maximum((lmu - tab["lo"]) / tab["h"], 0), K - 1)
    i = np.minimum(np.floor(x), K - 2)
    fx = x - i
    ii = i.astype(int)
    jm = jp.astype(int)[:, None]
    j1 = np.minimum(jm + 1, J - 1)
    fpm = fp[:, None]

    def lerp(M):
        return (1 - fpm) * ((1 - fx) * M[jm, ii] + fx * M[jm, ii + 1]) + fpm * (
            (1 - fx) * M[j1, ii] + fx * M[j1, ii + 1]
        )

    return {"w0": lerp(tab["w0"]), "w1": lerp(tab["w1"])}


def _s2(dev, df):
    with np.errstate(all="ignore"):
        return np.where(df < 1e-4, 0.0, dev / df)


def ql_dispersion(
    y,
    mu,
    phi,
    design=None,
    p=None,
    prior: float = 1.0,
    leverage: str = "trace",
    moments: str = "table",
    ngrid: int = 256,
    nphi: int = 64,
    table=None,
) -> dict:
    """Per-gene quasi-likelihood dispersion (edgeR v4 style).

    Returns ``{"deviance", "df", "s2"}`` per gene (``s2 = deviance / df``, 0 where
    ``df < 1e-4``). ``leverage`` is ``"trace"`` or ``"exact"`` (needs ``design``,
    CPU only); ``moments`` is ``"table"`` (shared interpolation table, both
    backends), ``"grid"`` (per-gene log-mu grid) or ``"cell"`` (every cell).
    """
    if leverage not in ("trace", "exact"):
        raise ValueError("'arg' should be one of 'trace', 'exact'")
    if moments not in ("table", "grid", "cell"):
        raise ValueError("'arg' should be one of 'table', 'grid', 'cell'")
    on_device = _is_tensor(y, mu)
    if p is None:
        if design is None:
            raise ValueError("supply `design` or its column count `p`")
        p = np.asarray(_np(design)).shape[1]
    if on_device and (leverage != "trace" or moments != "table"):
        raise ValueError(
            'on torch tensors ql_dispersion() supports leverage = "trace" and moments = "table" only'
        )
    shape = (y if isinstance(y, torch.Tensor) else mu).shape if on_device else np.asarray(y).shape
    ng, n = int(shape[0]), int(shape[1])
    phi = _rep_len(phi, ng)
    phi_eff = phi / prior

    if moments == "table":
        hdp = 1 - p / n
        if not math.isfinite(hdp) or hdp < 1e-4:
            return {"deviance": np.zeros(ng), "df": np.zeros(ng), "s2": np.zeros(ng)}
        if on_device:
            ref = mu if isinstance(mu, torch.Tensor) else y
            mu_t = _to_tensor(mu, ref)
            lmu = torch.log(torch.clamp(mu_t / prior, min=1e-32))
            if table is None:
                rng = (float(lmu.min().cpu()), float(lmu.max().cpu()))
                table = _ql_moment_table(rng, np.log(phi), ngrid, nphi)
            w = _ql_interp(table, lmu, np.log(phi))
            udp = nb_unit_deviance(y, mu_t, phi_eff)
            udp = torch.where(torch.isfinite(udp), udp, torch.zeros_like(udp))
            dev = torch.sum(udp * w["w0"], dim=1).cpu().numpy().astype(float)
            df = hdp * torch.sum(w["w1"], dim=1).cpu().numpy().astype(float)
        else:
            y = np.asarray(y, dtype=float)
            mu = np.asarray(mu, dtype=float)
            with np.errstate(all="ignore"):
                lmu = np.log(np.maximum(mu / prior, 1e-32))
            if table is None:
                table = _ql_moment_table((lmu.min(), lmu.max()), np.log(phi), ngrid, nphi)
            w = _ql_interp(table, lmu, np.log(phi))
            udp = nb_unit_deviance(y, mu, phi_eff)
            udp[~np.isfinite(udp)] = 0.0
            dev = (udp * w["w0"]).sum(axis=1)
            df = hdp * w["w1"].sum(axis=1)
        return {"deviance": dev, "df": df, "s2": _s2(dev, df)}

    # per-gene paths: a log-mu grid per gene, or the moments at every cell
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    des = None if design is None else np.asarray(design, dtype=float)
    dev = np.zeros(ng)
    df = np.zeros(ng)
    for g in range(ng):
        mg, yg = mu[g], y[g]
        if leverage == "exact":
            if des is None:
                raise ValueError('leverage = "exact" needs `design`')
            zw = np.sqrt(mg / (1 + mg * phi_eff[g]))
            q = np.linalg.qr(des * zw[:, None])[0]
            h = (q**2).sum(axis=1)
        else:
            h = np.full(n, p / n)
        # moments are taken at mu/prior with the UNSCALED phi (edgeR's compute_weight convention)
        ms = mg / prior
        if moments == "cell" or np.unique(ms).size <= ngrid:
            m = nb_deviance_moments(ms, phi[g])
        else:
            lg = np.log(np.maximum(ms, 1e-32))
            kn = np.linspace(lg.min(), lg.max(), ngrid)
            mk = nb_deviance_moments(np.exp(kn), phi[g])
            m = {"w0": np.interp(lg, kn, mk["w0"]), "w1": np.interp(lg, kn, mk["w1"])}
        udp = nb_unit_deviance(yg, mg, phi_eff[g])
        hdp = 1 - h
        drop = ~np.isfinite(hdp) | (hdp < 1e-4)
        udp = np.where(drop, 0.0, udp)
        hdp = np.where(drop, 0.0, hdp)
        dev[g] = np.nansum(udp * m["w0"])
        df[g] = np.nansum(hdp * m["w1"])
    return {"deviance": dev, "df": df, "s2": _s2(dev, df)}
