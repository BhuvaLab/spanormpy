"""Port of the ``edgeR::estimateDisp`` code path used by SpaNorm.

Covers exactly ``estimateDisp(y, design = matrix(1, n), offset = offs,
tagwise = TRUE, robust = TRUE)`` on a numeric matrix (edgeR 4.8, limma 3.66):

* edgeR: ``estimateDisp.default``, ``adjustedProfileLik`` (``compute_apl``),
  ``maximizeInterpolant`` (``fmm_spline`` + ``find_max``), ``WLEB``,
  ``locfitByCol``, ``aveLogCPM`` (``ave_log_cpm``), ``glmFit`` ->
  ``mglmOneWay`` -> ``mglmOneGroup`` (``glm_one_group_vec``),
  ``nbinomDeviance`` (``compute_unit_nb_deviance``), ``.residDF``,
  ``.comboGroups``.
* limma: ``squeezeVar`` -> ``fitFDistRobustly`` -> ``fitFDist``, ``loessFit``
  (``stats::lowess``), ``trigammaInverse``; statmod ``logmdigamma`` and
  ``gauss.quad.prob``; ``splines::ns``; ``stats::uniroot`` (``R_zeroin2``).
* locfit: ``locfit(y ~ x, alpha = span, deg = 0)`` with the default adaptive
  tree evaluation structure, ``fitted()`` (a local-constant fit with tricube
  weights and nearest-neighbour bandwidth at tree vertices, interpolated
  linearly because a degree-0 fit carries no derivatives).

Arrays are genes x cells (R orientation). Everything is float64 numpy on CPU.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
from scipy import special, stats


class EdgeRError(RuntimeError):
    """Raised where edgeR / limma / locfit raise an R error."""


_EPS = np.finfo(float).eps


# ----------------------------------------------------------------------------
# small special functions
# ----------------------------------------------------------------------------
def _logmdigamma(x):
    """statmod::logmdigamma = log(x) - digamma(x), evaluated as statmod does."""
    z = np.asarray(x, dtype=float)
    out = np.full(z.shape, np.nan)
    ok = ~np.isnan(z) & (z > 0)
    small = ok & (z < 5)
    big = ok & ~small
    if np.any(small):
        s = z[small]
        out[small] = (
            np.log(s / (s + 5))
            + _logmdigamma(s + 5)
            + 1 / s
            + 1 / (s + 1)
            + 1 / (s + 2)
            + 1 / (s + 3)
            + 1 / (s + 4)
        )
    if np.any(big):
        zz = z[big]
        x2 = 1 / zz**2
        # Bernoulli-number asymptotic tail, Horner form (statmod::logmdigamma)
        tail = 3617 * x2 / 8160
        for coef in (-1 / 12, 691 / 32760, -1 / 132, 1 / 240, -1 / 252, 1 / 120, -1 / 12):
            tail = x2 * (coef + tail)
        out[big] = 1 / (2 * zz) - tail
    return out


def _trigamma_inverse(x: float) -> float:
    """limma::trigammaInverse for a scalar."""
    if not np.isfinite(x) and not np.isinf(x):
        raise EdgeRError("missing value where TRUE/FALSE needed")
    if x < 0:
        return float("nan")
    if x > 1e7:
        return 1 / math.sqrt(x)
    if x < 1e-6:
        return 1 / x
    y = 0.5 + 1 / x
    it = 0
    while True:
        it += 1
        tri = special.polygamma(1, y)
        dif = tri * (1 - tri / x) / special.polygamma(2, y)
        y = y + dif
        if -dif / y < 1e-8:
            break
        if it > 50:
            warnings.warn("Iteration limit exceeded", stacklevel=2)
            break
    return float(y)


def _zeroin2(f, ax, bx, fa, fb, tol, maxit=1000):
    """R_zeroin2 (Brent) as used by stats::uniroot; returns the root."""
    a, b = ax, bx
    c, fc = a, fa
    maxit = maxit + 1
    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    while maxit:
        maxit -= 1
        prev_step = b - a
        if abs(fc) < abs(fb):
            a, b, c = b, c, b
            fa, fb, fc = fb, fc, fb
        tol_act = 2 * _EPS * abs(b) + tol / 2
        new_step = (c - b) / 2
        if abs(new_step) <= tol_act or fb == 0.0:
            return b
        if abs(prev_step) >= tol_act and abs(fa) > abs(fb):
            cb = c - b
            if a == c:
                t1 = fb / fa
                p = cb * t1
                q = 1.0 - t1
            else:
                q = fa / fc
                t1 = fb / fc
                t2 = fb / fa
                p = t2 * (cb * q * (q - t1) - (b - a) * (t1 - 1.0))
                q = (q - 1.0) * (t1 - 1.0) * (t2 - 1.0)
            if p > 0:
                q = -q
            else:
                p = -p
            if p < (0.75 * cb * q - abs(tol_act * q) / 2) and p < abs(prev_step * q / 2):
                new_step = p / q
        if abs(new_step) < tol_act:
            new_step = tol_act if new_step > 0 else -tol_act
        a, fa = b, fb
        b += new_step
        fb = f(b)
        if (fb > 0 and fc > 0) or (fb < 0 and fc < 0):
            c, fc = a, fa
    raise EdgeRError("uniroot: failed to converge")


# ----------------------------------------------------------------------------
# F distribution helpers (R semantics incl. df2 = Inf and log.p tails)
# ----------------------------------------------------------------------------
def _qf(p, df1, df2):
    if np.isinf(df2):
        return stats.chi2.ppf(p, df1) / df1
    return stats.f.ppf(p, df1, df2)


def _df_dens(x, df1, df2):
    if np.isinf(df2):
        return stats.chi2.pdf(x * df1, df1) * df1
    return stats.f.pdf(x, df1, df2)


def _log_betainc_small_x(x, a, b):
    """log I_x(a, b) for small x (continued fraction, Numerical Recipes betacf)."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    tiny = 1e-300
    c = np.ones_like(x)
    d = 1 - qab * x / qap
    d = np.where(np.abs(d) < tiny, tiny, d)
    d = 1 / d
    h = d.copy()
    for m in range(1, 1000):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1 + aa * d
        d = np.where(np.abs(d) < tiny, tiny, d)
        c = 1 + aa / c
        c = np.where(np.abs(c) < tiny, tiny, c)
        d = 1 / d
        h = h * d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1 + aa * d
        d = np.where(np.abs(d) < tiny, tiny, d)
        c = 1 + aa / c
        c = np.where(np.abs(c) < tiny, tiny, c)
        d = 1 / d
        de = d * c
        h = h * de
        if np.all(np.abs(de - 1) < 1e-15):
            break
    return a * np.log(x) + b * np.log1p(-x) - np.log(a) - special.betaln(a, b) + np.log(h)


def _pf_upper_log(f, df1, df2):
    """pf(f, df1, df2, lower.tail = FALSE, log.p = TRUE), accurate in the far tail."""
    f = np.asarray(f, dtype=float)
    if np.isinf(df2):
        return stats.chi2.logsf(f * df1, df1)
    out = np.atleast_1d(stats.f.logsf(f, df1, df2))
    bad = ~np.isfinite(out) | (out < -700)
    if np.any(bad):
        fb = np.atleast_1d(f)[bad]
        d1 = np.broadcast_to(np.asarray(df1, dtype=float), out.shape)[bad]
        x = df2 / (df2 + d1 * fb)  # sf = I_x(df2/2, df1/2)
        out = out.copy()
        for k, idx in enumerate(np.where(bad)[0]):
            out[idx] = _log_betainc_small_x(x[k], df2 / 2, d1[k] / 2)[0]
    return out


# ----------------------------------------------------------------------------
# stats::lowess (clowess)
# ----------------------------------------------------------------------------
def _lowest(x, y, n, xs, nleft, nright, w, rw, iterpass):
    """Port of clowess' lowest(); 0-based arrays; returns (ys, notok)."""
    rng = x[n - 1] - x[0]
    h = max(xs - x[nleft], x[nright] - xs)
    h9 = 0.999 * h
    h1 = 0.001 * h
    a = 0.0
    j = nleft
    while j < n:
        w[j] = 0.0
        r = abs(x[j] - xs)
        if r <= h9:
            if r <= h1:
                w[j] = 1.0
            else:
                w[j] = (1.0 - (r / h) ** 3) ** 3
            if iterpass > 1:
                w[j] *= rw[j]
            a += w[j]
        elif x[j] > xs:
            break
        j += 1
    nrt = j - 1
    if a <= 0.0:
        return 0.0, True
    w[nleft : nrt + 1] /= a
    if h > 0.0:
        a = float(np.sum(w[nleft : nrt + 1] * x[nleft : nrt + 1]))
        b = xs - a
        c = float(np.sum(w[nleft : nrt + 1] * (x[nleft : nrt + 1] - a) ** 2))
        if math.sqrt(c) > 0.001 * rng:
            b /= c
            w[nleft : nrt + 1] *= b * (x[nleft : nrt + 1] - a) + 1.0
    ys = float(np.sum(w[nleft : nrt + 1] * y[nleft : nrt + 1]))
    return ys, False


def _lowess(x, y, f=2.0 / 3.0, iters=3):
    """``stats::lowess(x, y, f, iter)`` fitted values in the order of sorted x."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    o = np.argsort(x, kind="stable")
    xs_, ys_ = x[o], y[o]
    n = len(xs_)
    delta = 0.01 * (xs_[-1] - xs_[0])
    if n < 2:
        return ys_.copy(), o
    ys = np.zeros(n)
    rw = np.zeros(n)
    res = np.zeros(n)
    ns = max(2, min(n, int(f * n + 1e-7)))
    it = 1
    while it <= iters + 1:
        nleft, nright, last, i = 0, ns - 1, -1, 0
        while True:
            if nright < n - 1:
                d1 = xs_[i] - xs_[nleft]
                d2 = xs_[nright + 1] - xs_[i]
                if d1 > d2:
                    nleft += 1
                    nright += 1
                    continue
            val, notok = _lowest(xs_, ys_, n, xs_[i], nleft, nright, res, rw, it)
            ys[i] = ys_[i] if notok else val
            if last < i - 1:
                denom = xs_[i] - xs_[last]
                for j in range(last + 1, i):
                    alpha = (xs_[j] - xs_[last]) / denom
                    ys[j] = alpha * ys[i] + (1.0 - alpha) * ys[last]
            last = i
            cut = xs_[last] + delta
            i = last + 1
            while i < n:
                if xs_[i] > cut:
                    break
                if xs_[i] == xs_[last]:
                    ys[i] = ys[last]
                    last = i
                i += 1
            i = max(last + 1, i - 1)
            if last >= n - 1:
                break
        res = ys_ - ys
        sc = float(np.sum(np.abs(res))) / n
        if it > iters:
            break
        rw = np.abs(res)
        m1 = n // 2
        srt = np.sort(rw)
        if n % 2 == 0:
            m2 = n - m1 - 1
            cmad = 3.0 * (srt[m1] + srt[m2])
        else:
            cmad = 6.0 * srt[m1]
        if cmad < 1e-7 * sc:
            break
        c9, c1 = 0.999 * cmad, 0.001 * cmad
        r = np.abs(res)
        rw = np.where(r <= c1, 1.0, np.where(r <= c9, (1.0 - (r / cmad) ** 2) ** 2, 0.0))
        it += 1
    return ys, o


def _loess_fit_unweighted(y, x, span):
    """limma::loessFit(y, x, span) without weights -> lowess(x, y, f = span, iter = 3)."""
    n = len(y)
    if span < 1.0 / n:
        return y.copy(), np.zeros(n)
    fit_sorted, o = _lowess(x, y, f=span, iters=3)
    fitted = np.empty(n)
    fitted[o] = fit_sorted
    return fitted, y - fitted


# ----------------------------------------------------------------------------
# locfit(y ~ x, alpha = span, deg = 0), fitted()
# ----------------------------------------------------------------------------
def _locfit0_by_col(Y, x, span):
    """edgeR::locfitByCol(Y, x, span = span, degree = 0)."""
    Y = np.asarray(Y, dtype=float)
    x = np.asarray(x, dtype=float)
    n = Y.shape[0]
    if span * n < 2 or n <= 1:
        return Y.copy()

    nn = int(n * span + 1e-12)
    cut = 0.8
    a0 = 1.0 if span > 1 else 1.0 / span
    cu = min(1.0, cut)
    nvm = int((5 * a0 / cu + 1) * 2)  # atree_guessnv (mk/100 == 1)
    fl_lo, fl_hi = float(np.min(x)), float(np.max(x))

    vx: list[float] = []
    vh: list[float] = []
    vv: list[np.ndarray] = []

    def fit_vertex(xev):
        di = np.abs(x - xev)
        if nn < n:
            h = float(np.partition(di, nn - 1)[nn - 1])
        else:
            h = float(np.max(di)) * math.exp(math.log(nn / n))
        if h == 0:
            w = (di == 0).astype(float)
        else:
            u = di / h
            w = np.where(u > 1, 0.0, (1 - u**3) ** 3)
        keep = w > 0
        s0 = float(np.sum(w[keep]))
        return h, (w[keep] @ Y[keep]) / s0

    def add_vertex(xev):
        h, val = fit_vertex(xev)
        vx.append(xev)
        vh.append(h)
        vv.append(val)

    add_vertex(fl_lo)
    add_vertex(fl_hi)

    # tree nodes: (v0, v1, left, right)
    nodes: list[list[int]] = []

    def grow(v0, v1):
        node = len(nodes)
        nodes.append([v0, v1, -1, -1])
        ll, ur = vx[v0], vx[v1]
        hs = [h for h in (vh[v0], vh[v1]) if h > 0]
        hmin = min(hs) if hs else 0.0
        le = ur - ll
        if hmin == 0:
            score = 2 * (ur - ll) / (fl_hi - fl_lo) if fl_hi != fl_lo else float("nan")
        else:
            score = le / hmin
        if not (cut < score):
            return node
        pv = le < cut * min(vh[v0], vh[v1])
        if len(vx) == nvm:
            raise EdgeRError("newsplit: out of vertex space")
        xm = (ll + ur) / 2
        if pv:
            vx.append(xm)
            vh.append((vh[v0] + vh[v1]) / 2)
            vv.append((vv[v0] + vv[v1]) / 2)
        else:
            add_vertex(xm)
        nv = len(vx) - 1
        nodes[node][2] = grow(v0, nv)
        nodes[node][3] = grow(nv, v1)
        return node

    grow(0, 1)

    nd = np.array(nodes)
    vxa = np.array(vx)
    vva = np.vstack(vv)
    cur = np.zeros(n, dtype=int)
    while True:
        leaf = nd[cur, 2] < 0
        if np.all(leaf):
            break
        act = ~leaf
        c = cur[act]
        ll = vxa[nd[c, 0]]
        ur = vxa[nd[c, 1]]
        lo = 2 * (x[act] - ll) < (ur - ll)
        cur[act] = np.where(lo, nd[c, 2], nd[c, 3])
    v0, v1 = nd[cur, 0], nd[cur, 1]
    ll, ur = vxa[v0], vxa[v1]
    d = ur - ll
    hh = x - ll
    with np.errstate(divide="ignore", invalid="ignore"):
        out = ((d - hh)[:, None] * vva[v0] + hh[:, None] * vva[v1]) / d[:, None]
    zero = d == 0
    if np.any(zero):
        out[zero] = vva[v0[zero]]
    return out


# ----------------------------------------------------------------------------
# maximizeInterpolant
# ----------------------------------------------------------------------------
def _fmm_spline(x, y):
    """Row-vectorised stats fmm_spline; y is (R, n). Returns b, c, d (R, n)."""
    n = len(x)
    R = y.shape[0]
    b = np.zeros((R, n))
    c = np.zeros((R, n))
    d = np.zeros((R, n))
    if n < 2:
        return b, c, d
    if n < 3:
        t = y[:, 1] - y[:, 0]
        b[:, 0] = t / (x[1] - x[0])
        b[:, 1] = b[:, 0]
        return b, c, d
    nm1 = n - 1
    d[:, 0] = x[1] - x[0]
    c[:, 1] = (y[:, 1] - y[:, 0]) / d[:, 0]
    for i in range(1, nm1):
        d[:, i] = x[i + 1] - x[i]
        b[:, i] = 2.0 * (d[:, i - 1] + d[:, i])
        c[:, i + 1] = (y[:, i + 1] - y[:, i]) / d[:, i]
        c[:, i] = c[:, i + 1] - c[:, i]
    b[:, 0] = -d[:, 0]
    b[:, n - 1] = -d[:, nm1 - 1]
    c[:, 0] = 0.0
    c[:, n - 1] = 0.0
    if n > 3:
        c[:, 0] = c[:, 2] / (x[3] - x[1]) - c[:, 1] / (x[2] - x[0])
        c[:, n - 1] = c[:, nm1 - 1] / (x[n - 1] - x[n - 3]) - c[:, n - 3] / (x[nm1 - 1] - x[n - 4])
        c[:, 0] = c[:, 0] * d[:, 0] * d[:, 0] / (x[3] - x[0])
        c[:, n - 1] = -c[:, n - 1] * d[:, nm1 - 1] * d[:, nm1 - 1] / (x[n - 1] - x[n - 4])
    for i in range(1, n):
        t = d[:, i - 1] / b[:, i - 1]
        b[:, i] = b[:, i] - t * d[:, i - 1]
        c[:, i] = c[:, i] - t * c[:, i - 1]
    c[:, n - 1] = c[:, n - 1] / b[:, n - 1]
    for i in range(nm1 - 1, -1, -1):
        c[:, i] = (c[:, i] - d[:, i] * c[:, i + 1]) / b[:, i]
    b[:, n - 1] = (y[:, n - 1] - y[:, n - 2]) / d[:, n - 2] + d[:, n - 2] * (
        c[:, n - 2] + 2.0 * c[:, n - 1]
    )
    for i in range(nm1):
        b[:, i] = (y[:, i + 1] - y[:, i]) / d[:, i] - d[:, i] * (c[:, i + 1] + 2.0 * c[:, i])
        d[:, i] = (c[:, i + 1] - c[:, i]) / d[:, i]
        c[:, i] = 3.0 * c[:, i]
    c[:, n - 1] = 3.0 * c[:, n - 1]
    d[:, n - 1] = d[:, nm1 - 1]
    return b, c, d


def _maximize_interpolant(x, y):
    """edgeR::maximizeInterpolant(x, y): per-row argmax of the FMM spline through (x, y[r])."""
    x = np.asarray(x, dtype=float)
    y = np.atleast_2d(np.asarray(y, dtype=float))
    if len(x) != y.shape[1]:
        raise EdgeRError("number of columns must equal number of spline points")
    if np.any(np.diff(x) <= 0):
        raise EdgeRError("spline points must be unique and sorted")
    R, npts = y.shape
    if R == 0:
        return np.zeros(0)
    ar = np.arange(R)
    at = np.argmax(y, axis=1)
    maxed = y[ar, at]
    xmax = x[at]
    b, c, d = _fmm_spline(x, y)

    def segment(idx, valid, dx):
        nonlocal maxed, xmax
        i = np.where(valid, idx, 0)
        ld, lc, lb = d[ar, i], c[ar, i], b[ar, i]
        with np.errstate(invalid="ignore", divide="ignore"):
            delta = lc * lc - 3 * ld * lb
            solvable = valid & ~(delta < 0)
            sol = (-lc - np.sqrt(delta)) / (3 * ld)
            cond = solvable & (sol > 0) & (sol < dx)
            temp = ((ld * sol + lc) * sol + lb) * sol + y[ar, i]
        upd = cond & (temp > maxed)
        maxed = np.where(upd, temp, maxed)
        xmax = np.where(upd, sol + x[i], xmax)

    left_valid = at > 0
    li = np.maximum(at - 1, 0)
    segment(li, left_valid, x[at] - x[li])
    right_valid = at < npts - 1
    ri = np.minimum(at, npts - 2)
    segment(ri, right_valid, x[np.minimum(at + 1, npts - 1)] - x[at])
    return xmax


# ----------------------------------------------------------------------------
# glm: one group, deviance, adjusted profile likelihood
# ----------------------------------------------------------------------------
def _glm_one_group(y, offset, disp, start=None, maxit=50, tol=1e-10, return_conv=False):
    """edgeR glm_one_group_vec / fit_one_group_mat for unit weights; rows are genes.

    y, offset: (G, n); disp: (G,) or (G, n); start: (G,) or None.
    Returns beta (G,) (and a convergence flag per gene if ``return_conv``).
    """
    G, n = y.shape
    disp = np.broadcast_to(
        np.asarray(disp, dtype=float).reshape(G, -1)
        if np.ndim(disp)
        else np.full((G, 1), float(disp)),
        (G, n),
    )
    # Poisson closed form (all-zero dispersion, unit weights)
    pois = np.all(disp == 0, axis=1)
    beta = np.empty(G)
    conv = np.ones(G, dtype=bool)
    with np.errstate(all="ignore"):
        if np.any(pois):
            sc = y[pois].sum(axis=1)
            sl = np.exp(offset[pois]).sum(axis=1)
            beta[pois] = np.where(sc == 0, -np.inf, np.log(sc / sl))
        nr = ~pois
        if np.any(nr):
            yy, oo, dd = y[nr], offset[nr], disp[nr]
            nz = yy > 1e-10
            allzero = ~nz.any(axis=1)
            if start is None:
                cur = np.log(np.where(nz, yy / np.exp(oo), 0.0).sum(axis=1) / n)
            else:
                cur = np.asarray(start, dtype=float)[nr].copy()
            res = cur.copy()
            active = ~allzero
            res[allzero] = -np.inf
            for _ in range(maxit):
                if not active.any():
                    break
                idx = np.where(active)[0]
                mu = np.exp(cur[idx, None] + oo[idx])
                den = 1 + mu * dd[idx]
                dl = ((yy[idx] - mu) / den).sum(axis=1)
                info = (mu / den).sum(axis=1)
                step = dl / info
                cur[idx] += step
                done = np.abs(step) < tol
                res[idx[done]] = cur[idx[done]]
                active[idx[done]] = False
            # Not converged: edgeR's C code never assigns `beta` (`ocoef` is an uninitialised
            # local in fit_one_group_mat), and R observes ~4.6e-310, i.e. effectively 0.
            # The uninitialised local keeps the value of the previous Newton-fitted row
            # (same stack slot, reused across the row loop); the first row reads ~0.
            conv_nr = ~active
            for i in np.where(active)[0]:
                res[i] = res[i - 1] if i > 0 else 0.0
            beta[nr] = res
            conv[nr] = conv_nr | allzero
    return (beta, conv) if return_conv else beta


def _unit_nb_deviance(y, mu, phi):
    """edgeR compute_unit_nb_deviance (note: C integer division makes 2/3 == 0)."""
    y = y + 1e-8
    mu = mu + 1e-8
    phi = np.broadcast_to(phi, y.shape)
    with np.errstate(all="ignore"):
        resid = y - mu
        poi = 2 * (
            y * np.log(y / mu) - resid - 0.5 * resid * resid * phi * (1 + phi * (0.0 * resid - y))
        )
        prod = mu * phi
        gam = 2 * ((y - mu) / mu - np.log(y / mu)) * mu / (1 + prod)
        invphi = 1 / phi
        nb = 2 * (y * np.log(y / mu) + (y + invphi) * np.log((mu + invphi) / (y + invphi)))
        out = np.where(phi < 1e-4, poi, np.where(prod > 1e6, gam, nb))
    return np.maximum(out, 0)


def _glm_fit_one_group(y, offset, disp, start=None):
    """glmFit for the intercept-only design: returns (beta, mu, deviance)."""
    beta = _glm_one_group(y, offset, disp, start)
    beta = np.maximum(beta, -1e8)
    with np.errstate(all="ignore"):
        mu = np.exp(offset + beta[:, None])
    d = np.asarray(disp, dtype=float)
    dmat = d[:, None] if d.ndim == 1 else d
    dev = _unit_nb_deviance(y, mu, dmat).sum(axis=1)
    return beta, mu, dev


def _compute_apl_one_coef(y, mu, disp):
    """edgeR compute_apl (unit weights, 1-column design, Cox-Reid adjust)."""
    with np.errstate(all="ignore"):
        nz = mu != 0
        u = np.where(nz, mu, 1.0)
        yy = np.where(nz, y, 0.0)
        if disp > 0:
            r = 1.0 / disp
            logur = np.log(u + r)
            ll = (
                yy * np.log(u)
                - yy * logur
                + r * np.log(r)
                - r * logur
                + special.gammaln(yy + r)
                - special.gammaln(yy + 1)
                - special.gammaln(r)
            )
        else:
            ll = yy * np.log(u) - u - special.gammaln(yy + 1)
        ll = np.where(nz, ll, 0.0)
        zw = np.where(nz, u / (1 + disp * u), 0.0)
        adj = np.log(np.abs(zw.sum(axis=1))) / 2
    return ll.sum(axis=1) - adj


def _ave_log_cpm(y, lib_size, dispersion, prior_count=2.0, maxit=50, tol=1e-10):
    """edgeR::aveLogCPM(y, lib.size, dispersion) with a scalar dispersion."""
    G, n = y.shape
    off = np.exp(np.log(lib_size))  # edgeR passes log(lib.size) as the offset
    ave = off.sum() / n
    prior = prior_count * off / ave
    offset = np.log(off + 2 * prior)
    beta = _glm_one_group(y + prior[None, :], np.broadcast_to(offset, (G, n)), dispersion)
    return (beta + math.log(1e6)) / math.log(2)


# ----------------------------------------------------------------------------
# natural splines (splines::ns with intercept = TRUE) and lm.fit
# ----------------------------------------------------------------------------
def _bspline_design(knots, x, derivs=0):
    from scipy.interpolate import BSpline

    nb = len(knots) - 4
    spl = BSpline(knots, np.eye(nb), 3, extrapolate=False)
    if derivs:
        spl = spl.derivative(derivs)
    x = np.asarray(x, dtype=float)
    out = spl(x)
    # BSpline is left-continuous at the right end only through extrapolate; patch endpoint.
    right = x == knots[-1]
    if np.any(right):
        out[right] = spl(np.nextafter(knots[-1], -np.inf))
    return out


def _ns_basis(x, df=None, knots=None, boundary=None):
    """splines::ns(x, df, intercept = TRUE); returns (basis, knots, boundary)."""
    x = np.asarray(x, dtype=float)
    given_boundary = boundary is not None
    bk = (
        np.sort(np.asarray(boundary, dtype=float))
        if given_boundary
        else np.array([x.min(), x.max()])
    )
    outside = (x < bk[0]) | (x > bk[1]) if given_boundary else np.zeros(len(x), bool)
    if knots is None and df is not None:
        nik = df - 2
        if nik > 0:
            probs = np.linspace(0, 1, nik + 2)[1:-1]
            knots = np.quantile(x[~outside], probs)
            if np.any(np.isin([knots.min(), knots.max()], bk)):
                if knots.min() == bk[0]:
                    i = knots == bk[0]
                    if np.all(i):
                        raise EdgeRError("all interior knots match left boundary knot")
                    knots = knots.copy()
                    knots[i] = knots[i] + (knots[knots > bk[0]].min() - bk[0]) / 8
                if knots.max() == bk[1]:
                    i = knots == bk[1]
                    if np.all(i):
                        raise EdgeRError("all interior knots match right boundary knot")
                    knots = knots.copy()
                    knots[i] = knots[i] - (bk[1] - knots[knots < bk[1]].max()) / 8
        else:
            knots = np.zeros(0)
    elif knots is None:
        knots = np.zeros(0)
    knots = np.asarray(knots, dtype=float)
    A = np.sort(np.concatenate([np.repeat(bk, 4), knots]))
    nb = len(A) - 4
    basis = np.zeros((len(x), nb))
    ins = ~outside
    if np.any(ins):
        basis[ins] = _bspline_design(A, x[ins])
    if np.any(outside):
        for pivot, sel in ((bk[0], x < bk[0]), (bk[1], x > bk[1])):
            if np.any(sel):
                tt = np.vstack([_bspline_design(A, [pivot]), _bspline_design(A, [pivot], 1)])
                basis[sel] = np.column_stack([np.ones(sel.sum()), x[sel] - pivot]) @ tt
    const = _bspline_design(A, bk, 2)
    q, _ = np.linalg.qr(const.T, mode="complete")
    return basis @ q[:, 2:], knots, bk


def _lm_fit(design, e):
    """lm.fit: fitted values, coefficients, rank, residual sum of squares."""
    # R's dqrdc2 uses a 1e-7 relative pivoting tolerance
    sv = np.linalg.svd(design, compute_uv=False)
    rank = int(np.sum(sv > 1e-7 * sv[0])) if sv.size else 0
    coef = np.linalg.lstsq(design, e, rcond=1e-7)[0]
    fitted = design @ coef
    rss = float(np.sum((e - fitted) ** 2))
    return fitted, coef, rank, rss


# ----------------------------------------------------------------------------
# limma fitFDist / fitFDistRobustly / squeezeVar
# ----------------------------------------------------------------------------
def _fit_fdist(x, df1, covariate=None):
    """limma::fitFDist for a vector df1; returns dict(scale, df2)."""
    x = np.asarray(x, dtype=float).copy()
    df1 = np.asarray(df1, dtype=float).copy()
    n = len(x)
    if n == 0:
        return {"scale": np.nan, "df2": np.nan}
    if n == 1:
        return {"scale": x, "df2": 0.0}
    ok = np.isfinite(df1) & (df1 > 1e-15)
    if covariate is not None:
        covariate = np.asarray(covariate, dtype=float).copy()
        if len(covariate) != n:
            raise EdgeRError("x and covariate must be of same length")
        if np.any(np.isnan(covariate)):
            raise EdgeRError("NA covariate values not allowed")
        isfin = np.isfinite(covariate)
        if not np.all(isfin):
            if np.any(isfin):
                lo, hi = covariate[isfin].min(), covariate[isfin].max()
                covariate[covariate == -np.inf] = lo - 1
                covariate[covariate == np.inf] = hi + 1
            else:
                covariate = np.sign(covariate)
    ok = ok & np.isfinite(x) & (x > -1e-15)
    nok = int(ok.sum())
    if nok == 1:
        return {"scale": x[ok], "df2": 0.0}
    notallok = nok < n
    cov_notok = None
    if notallok:
        x = x[ok]
        df1 = df1[ok]
        if covariate is not None:
            cov_notok = covariate[~ok]
            covariate = covariate[ok]
    if covariate is not None:
        splinedf = 1 + (nok >= 3) + (nok >= 6) + (nok >= 30)
        splinedf = min(splinedf, len(np.unique(covariate)))
        if splinedf < 2:
            out = _fit_fdist(x, df1)
            out["scale"] = np.resize(np.atleast_1d(out["scale"]), n)  # rep_len(scale, n)
            return out
    x = np.maximum(x, 0)
    m = float(np.median(x))
    if m == 0:
        warnings.warn(
            "More than half of residual variances are exactly zero: eBayes unreliable", stacklevel=2
        )
        m = 1.0
    elif np.any(x == 0):
        warnings.warn(
            "Zero sample variances detected, have been offset away from zero", stacklevel=2
        )
    x = np.maximum(x, 1e-5 * m)
    z = np.log(x)
    e = z + _logmdigamma(df1 / 2)
    if covariate is None:
        emean = float(np.mean(e))
        evar = float(np.sum((e - emean) ** 2)) / (nok - 1)
    else:
        design, knots, bk = _ns_basis(covariate, df=splinedf)
        fitted, coef, rank, rss = _lm_fit(design, e)
        if notallok:
            design2, _, _ = _ns_basis(cov_notok, knots=knots, boundary=bk)
            emean = np.zeros(n)
            emean[ok] = fitted
            emean[~ok] = design2 @ coef
        else:
            emean = fitted
        if nok - rank <= 0:
            raise EdgeRError("missing value where TRUE/FALSE needed")
        evar = rss / (nok - rank)
    evar = evar - float(np.mean(special.polygamma(1, df1 / 2)))
    if evar > 0:
        df2 = 2 * _trigamma_inverse(evar)
        s20 = np.exp(emean - _logmdigamma(np.array([df2 / 2]))[0])
    else:
        df2 = np.inf
        s20 = float(np.mean(x)) if covariate is None else np.exp(emean)
    return {"scale": s20, "df2": df2}


_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(128)
_GQ_NODES = (_GL_NODES + 1) / 2  # statmod::gauss.quad.prob(128, "uniform")
_GQ_WEIGHTS = _GL_WEIGHTS / 2


def _winsorized_moments(df1, df2, wtp):
    fq = np.array([_qf(wtp[0], df1, df2), _qf(1 - wtp[1], df1, df2)])
    zq = np.log(fq)
    q = fq / (1 + fq)
    nodes = q[0] + (q[1] - q[0]) * _GQ_NODES
    fnodes = nodes / (1 - nodes)
    znodes = np.log(fnodes)
    f = _df_dens(fnodes, df1, df2) / (1 - nodes) ** 2
    q21 = q[1] - q[0]
    m = q21 * np.sum(_GQ_WEIGHTS * f * znodes) + np.sum(zq * wtp)
    v = q21 * np.sum(_GQ_WEIGHTS * f * (znodes - m) ** 2) + np.sum((zq - m) ** 2 * wtp)
    return m, v


def _fit_fdist_robustly(x, df1, covariate, winsor_tail_p=(0.05, 0.1)):
    """limma::fitFDistRobustly with a covariate; returns dict(df2, df2_shrunk)."""
    x = np.asarray(x, dtype=float).copy()
    df1 = np.asarray(df1, dtype=float).copy()
    covariate = np.asarray(covariate, dtype=float)
    n = len(x)
    if n < 2:
        return {"df2": np.nan, "df2_shrunk": np.full(max(n, 1), np.nan)}
    if n == 2:
        raise EdgeRError("fitFDistRobustly with two variances is not supported")
    if len(covariate) != n:
        raise EdgeRError("x and covariate are different lengths")
    if not np.all(np.isfinite(covariate)):
        raise EdgeRError("covariate contains NA or infinite values")
    ok = ~np.isnan(x) & np.isfinite(df1) & (df1 > 1e-6)
    if not np.all(ok):
        shrunk = x.copy()
        fit = _fit_fdist_robustly(x[ok], df1[ok], covariate[ok], winsor_tail_p)
        shrunk[ok] = fit["df2_shrunk"]
        shrunk[~ok] = fit["df2"]
        return {"df2": fit["df2"], "df2_shrunk": shrunk}
    m = float(np.median(x))
    if m <= 0:
        raise EdgeRError("Variances are mostly <= 0")
    x[x < m * 1e-12] = m * 1e-12
    non_robust = _fit_fdist(x, df1, covariate)
    wtp = np.array(winsor_tail_p, dtype=float)
    prob = np.array([wtp[0], 1 - wtp[1]])
    if np.all(wtp < 1.0 / n):
        return {"df2": non_robust["df2"], "df2_shrunk": np.full(n, non_robust["df2"])}
    df1max = df1.max()
    if np.any(df1 < df1max - 1e-14):
        raise EdgeRError("fitFDistRobustly with unequal df1 is not supported")
    df1 = float(df1[0])
    z = np.log(x)
    ztrend, _ = _loess_fit_unweighted(z, covariate, 0.4)
    zresid = z - ztrend
    zrq = np.quantile(zresid, prob)
    zwins = np.minimum(np.maximum(zresid, zrq[0]), zrq[1])
    zwmean = float(np.mean(zwins))
    zwvar = float(np.mean((zwins - zwmean) ** 2)) * n / (n - 1)

    mom = _winsorized_moments(df1, np.inf, wtp)
    funval_inf = math.log(zwvar / mom[1])
    if funval_inf <= 0:
        df2 = np.inf
        ztc = ztrend + zwmean - mom[0]
        fstat = np.exp(z - ztc)
        tailp = stats.chi2.sf(fstat * df1, df1)
        r = stats.rankdata(fstat)
        emp = (n - r + 0.5) / n
        pno = np.minimum(tailp / emp, 1)
        df_pooled = n * df1
        shrunk = np.full(n, df2)
        o_mask = pno < 1
        if np.any(o_mask):
            shrunk[o_mask] = pno[o_mask] * df_pooled
            o = np.argsort(tailp, kind="stable")
            shrunk[o] = np.maximum.accumulate(shrunk[o])
        return {"df2": df2, "df2_shrunk": shrunk}

    def fun(xx):
        d2 = xx / (1 - xx)
        mm = _winsorized_moments(df1, d2, wtp)
        return math.log(zwvar / mm[1])

    if non_robust["df2"] == np.inf:
        return {"df2": np.inf, "df2_shrunk": np.full(n, np.inf)}
    rbx = non_robust["df2"] / (1 + non_robust["df2"])
    funval_low = fun(rbx)
    if funval_low >= 0:
        df2 = non_robust["df2"]
    else:
        root = _zeroin2(fun, rbx, 1.0, funval_low, funval_inf, 1e-8)
        df2 = root / (1 - root)
    mom = _winsorized_moments(df1, df2, wtp)
    ztc = ztrend + zwmean - mom[0]
    zres = z - ztc
    fstat = np.exp(zres)
    log_tailp = _pf_upper_log(fstat, df1, df2)
    r = stats.rankdata(fstat)
    log_emp = np.log(n - r + 0.5) - math.log(n)
    log_pno = np.minimum(log_tailp - log_emp, 0)
    pno = np.exp(log_pno)
    pout = -np.expm1(log_pno)
    if np.any(log_pno < 0):
        min_ltp = float(np.min(log_tailp))
        if min_ltp == -np.inf:
            df2_out = 0.0
            shrunk = pno * df2
        else:
            df2_out = math.log(0.5) / min_ltp * df2
            new_ltp = float(_pf_upper_log(np.array([fstat.max()]), df1, df2_out)[0])
            df2_out = math.log(0.5) / new_ltp * df2_out
            shrunk = pno * df2 + pout * df2_out
        o = np.argsort(log_tailp, kind="stable")
        ordered = shrunk[o]
        mm = np.cumsum(ordered) / np.arange(1, n + 1)
        imin = int(np.argmin(mm))
        ordered[: imin + 1] = mm[imin]
        shrunk[o] = np.maximum.accumulate(ordered)
    else:
        shrunk = np.full(n, df2)
    return {"df2": df2, "df2_shrunk": shrunk}


def _weighted_lowess_one_pass(x, y, w, span, npts=200):
    """limma::weightedLowess(x, y, weights = w, span, iterations = 1) fitted values (x sorted)."""
    n = len(x)
    thr = 1e-7
    # delta
    if npts >= n:
        delta = 0.0
    else:
        dx = np.sort(np.diff(x))
        cumrange = np.cumsum(dx)
        nc = np.arange(npts)
        delta = float(np.min(cumrange[len(dx) - 1 - nc] / (npts - nc)))
    spanweight = float(np.sum(w)) * span
    # find_seeds
    seeds = [0]
    last = 0
    for pt in range(1, n - 1):
        if x[pt] - x[last] > delta:
            seeds.append(pt)
            last = pt
    seeds.append(n - 1)
    fit = np.zeros(n)

    def lowess_fit(curpt, left, right, dist):
        sl = slice(left, right + 1)
        xs, ys, ws = x[sl], y[sl], w[sl]
        if dist < thr:
            return float(np.sum(ys * ws) / np.sum(ws))
        work = (1 - (np.abs(x[curpt] - xs) / dist) ** 3) ** 3 * ws
        allw = work.sum()
        xmean = np.sum(work * xs) / allw
        ymean = np.sum(work * ys) / allw
        t = xs - xmean
        var = np.sum(t * t * work)
        covar = np.sum(t * (ys - ymean) * work)
        if var < thr:
            return float(ymean)
        slope = covar / var
        return float(slope * x[curpt] + (ymean - slope * xmean))

    def limits(curpt):
        left = right = curpt
        curw = w[curpt]
        ende = curpt == n - 1
        ends = curpt == 0
        mdist = 0.0
        while curw < spanweight and (not ende or not ends):
            if ende:
                left -= 1
                curw += w[left]
                if left == 0:
                    ends = True
                mdist = max(mdist, x[curpt] - x[left])
            elif ends:
                right += 1
                curw += w[right]
                if right == n - 1:
                    ende = True
                mdist = max(mdist, x[right] - x[curpt])
            else:
                ld = x[curpt] - x[left - 1]
                rd = x[right + 1] - x[curpt]
                if ld < rd:
                    left -= 1
                    curw += w[left]
                    if left == 0:
                        ends = True
                    mdist = max(mdist, ld)
                else:
                    right += 1
                    curw += w[right]
                    if right == n - 1:
                        ende = True
                    mdist = max(mdist, rd)
        while left > 0 and x[left] == x[left - 1]:
            left -= 1
        while right < n - 1 and x[right] == x[right + 1]:
            right += 1
        return left, right, mdist

    subrange = (x[-1] - x[0]) / n
    l0, r0, d0 = limits(0)
    fit[0] = lowess_fit(0, l0, r0, d0)
    last_pt = 0
    for pt in seeds[1:]:
        lft, rgt, md = limits(pt)
        fit[pt] = lowess_fit(pt, lft, rgt, md)
        if pt - last_pt > 1:
            cur = x[pt] - x[last_pt]
            if cur > thr * subrange:
                slope = (fit[pt] - fit[last_pt]) / cur
                intercept = fit[pt] - slope * x[pt]
                fit[last_pt + 1 : pt] = slope * x[last_pt + 1 : pt] + intercept
            else:
                fit[last_pt + 1 : pt] = 0.5 * (fit[pt] + fit[last_pt])
        last_pt = pt
    return fit


def _loess_fit_weighted(y, x, weights, span, iterations, min_weight=1e-5, max_weight=1e5):
    """limma::loessFit(y, x, weights, ...) for the default weightedLowess method; fitted values."""
    n = len(y)
    fitted = np.full(n, np.nan)
    obs = np.isfinite(y) & np.isfinite(x)
    xobs, yobs = x[obs], y[obs]
    nobs = len(yobs)
    if nobs == 0:
        return fitted
    if span < 1.0 / nobs:
        fitted[obs] = y[obs]
        return fitted
    min_weight = max(min_weight, 0)
    wobs = np.asarray(weights, dtype=float)[obs].copy()
    wobs[np.isnan(wobs)] = 0
    wobs = np.minimum(np.maximum(wobs, min_weight), max_weight)
    if wobs.max() - wobs.min() < 1e-15:
        f, o = _lowess(xobs, yobs, f=span, iters=iterations - 1)
        tmp = np.empty(nobs)
        tmp[o] = f
        fitted[obs] = tmp
        return fitted
    nwobs = nobs if min_weight > 0 else int(np.sum(wobs > 0))
    if nwobs < 4 + 1 / span:
        if nwobs == 1:
            fitted[obs] = yobs[wobs > 0][0]
        else:
            X = np.column_stack([np.ones(nobs), xobs])
            sw = np.sqrt(wobs)
            coef = np.linalg.lstsq(X * sw[:, None], yobs * sw, rcond=None)[0]
            fitted[obs] = X @ coef
        return fitted
    if iterations != 1:
        raise EdgeRError("weighted loessFit with several iterations is not ported")
    o = np.argsort(xobs, kind="stable")
    fo = _weighted_lowess_one_pass(xobs[o], yobs[o], wobs[o], span)
    tmp = np.empty(nobs)
    tmp[o] = fo
    fitted[obs] = tmp
    return fitted


def _brent_fmin(f, ax, bx, tol):
    """R's Brent_fmin (stats::optimize)."""
    c = (3.0 - math.sqrt(5.0)) * 0.5
    eps = math.sqrt(_EPS)
    a, b = ax, bx
    v = a + c * (b - a)
    w = x = v
    d = e = 0.0
    fx = f(x)
    fv = fw = fx
    tol3 = tol / 3.0
    while True:
        xm = (a + b) * 0.5
        tol1 = eps * abs(x) + tol3
        t2 = tol1 * 2.0
        if abs(x - xm) <= t2 - (b - a) * 0.5:
            break
        p = q = r = 0.0
        if abs(e) > tol1:
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
            e = (b - x) if x < xm else (a - x)
            d = c * e
        else:
            d = p / q
            u = x + d
            if u - a < t2 or b - u < t2:
                d = tol1
                if x >= xm:
                    d = -d
        if abs(d) >= tol1:
            u = x + d
        elif d > 0.0:
            u = x + tol1
        else:
            u = x - tol1
        fu = f(u)
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


def _p_adjust_bh(p):
    n = len(p)
    o = np.argsort(-p, kind="stable")
    ro = np.argsort(o, kind="stable")
    i = np.arange(n, 0, -1)
    return np.minimum(1, np.minimum.accumulate(n / i * p[o]))[ro]


def _fit_fdist_unequal_df1(x, df1, covariate=None, span=None, robust=False, prior_weights=None):
    """limma::fitFDistUnequalDF1 (vector df1)."""
    x = np.asarray(x, dtype=float).copy()
    df1 = np.asarray(df1, dtype=float).copy()
    n = len(x)
    if len(df1) != n:
        raise EdgeRError("x and df1 are different lengths")
    if np.any(np.isnan(df1)):
        raise EdgeRError("NA df1 values")
    if covariate is not None:
        covariate = np.asarray(covariate, dtype=float)
        if len(covariate) != n:
            raise EdgeRError("x and covariate are different lengths")
        if np.any(np.isnan(covariate)):
            raise EdgeRError("prior.weights contain NA values")
    if prior_weights is not None:
        prior_weights = np.asarray(prior_weights, dtype=float).copy()
        if len(prior_weights) != n:
            raise EdgeRError("x and covariate are different lengths")
        if np.min(prior_weights) < 0:
            raise EdgeRError("prior.weights are negative")
    if np.any(np.isnan(x)):
        i = np.isnan(x)
        if prior_weights is None:
            prior_weights = (~i).astype(float)
        else:
            prior_weights[i] = 0
        x[i] = 0
    if df1.min() < 0.01:
        i = df1 < 0.01
        if prior_weights is None:
            prior_weights = (~i).astype(float)
        else:
            prior_weights[i] = 0
        df1[i] = 1
    has_pw = prior_weights is not None
    inf_mask = x > 0
    if has_pw:
        inf_mask[prior_weights == 0] = False
    n_inf = int(inf_mask.sum())
    if n_inf < 2:
        return {"scale": np.nan, "df2": np.nan}
    if n_inf == 2:
        covariate = None
        robust = False
        prior_weights = None
    m = float(np.median(x[inf_mask]))
    xpos = np.maximum(x, 1e-12 * m)
    z = np.log(xpos)
    d1 = df1 / 2
    e = z + _logmdigamma(d1)
    w = 1 / special.polygamma(1, d1)
    if has_pw:
        if prior_weights is None:
            raise EdgeRError("non-finite value supplied by optimize")
        w = w * prior_weights
    if covariate is None:
        emean = float(np.sum(w * e) / np.sum(w))
    else:
        if span is None:
            span = _choose_lowess_span(n, small_n=500)
        emean = _loess_fit_weighted(
            e, covariate, w / np.quantile(w, 0.75), span, 1, min_weight=1e-8, max_weight=100
        )
    d1x = d1 * xpos

    def minus_twice_loglik(par):
        d2 = par / (1 - par)
        d2s20 = d2 * np.exp(emean - _logmdigamma(np.array([d2]))[0])
        terms = (
            -(d1 + d2) * np.log1p(d1x / d2s20)
            - d1 * np.log(d2s20)
            + special.gammaln(d1 + d2)
            - special.gammaln(d2)
        )
        if has_pw:
            terms = prior_weights * terms
        val = -2 * float(np.sum(terms))
        if not np.isfinite(val):
            raise EdgeRError("non-finite value supplied by optimize")
        return val

    mn = _brent_fmin(minus_twice_loglik, 0.5, 0.9998, _EPS**0.25)
    d2 = mn / (1 - mn)
    s20 = np.exp(emean - _logmdigamma(np.array([d2]))[0])
    if not robust:
        return {"scale": s20, "df2": 2 * d2}
    df2 = 2 * d2
    fstat = x / s20
    right_p = stats.f.sf(fstat, df1, df2)
    left_p = 1 - right_p
    if left_p.min() < 0.001:
        i = left_p < 0.001
        left_p[i] = stats.f.cdf(fstat[i], df1[i], df2)
    two_sided = 2 * np.minimum(left_p, right_p)
    fdr = _p_adjust_bh(two_sided)
    fdr[fdr > 0.3] = 1
    if fdr.min() == 1.0:
        return {"scale": s20, "df2": df2}
    outpw = _fit_fdist_unequal_df1(x, df1, covariate, span=span, robust=False, prior_weights=fdr)
    s20 = outpw["scale"]
    df2 = outpw["df2"]
    r = stats.rankdata(fstat)
    uniform_p = (n - r + 0.5) / n
    pno = np.minimum(right_p / uniform_p, 1)
    if pno.min() == 1.0:
        return outpw
    i = int(np.argmin(right_p))
    min_right_p = right_p[i]
    if min_right_p == 0:
        df2_out = 0.0
        shrunk = pno * df2
    else:
        df2_out = math.log(0.5) / math.log(min_right_p) * df2
        new_lrp = float(_pf_upper_log(np.array([fstat[i]]), np.array([df1[i]]), df2_out)[0])
        df2_out = math.log(0.5) / new_lrp * df2_out
        shrunk = pno * df2 + (1 - pno) * df2_out
    o = np.argsort(right_p, kind="stable")
    ordered = shrunk[o]
    mm = np.cumsum(ordered) / np.arange(1, n + 1)
    imin = int(np.argmin(mm))
    ordered[: imin + 1] = mm[imin]
    shrunk[o] = np.maximum.accumulate(ordered)
    return {"scale": s20, "df2": df2, "df2_outlier": df2_out, "df2_shrunk": shrunk}


def _squeeze_var_prior_df(var, df, covariate, robust, winsor_tail_p=(0.05, 0.1)):
    """limma::squeezeVar(...)$df.prior."""
    var = np.asarray(var, dtype=float).copy()
    df = np.asarray(df, dtype=float)
    n = len(var)
    if n == 0:
        raise EdgeRError("var is empty")
    if n < 3:
        return 0.0
    var[df == 0] = 0
    dfp = df[df > 0]
    # identical(min(dfp), max(dfp)); an empty dfp gives min = Inf, max = -Inf (not identical)
    legacy = dfp.size > 0 and dfp.min() == dfp.max()
    if legacy:
        if robust:
            prior_df = _fit_fdist_robustly(var, df, covariate, winsor_tail_p)["df2_shrunk"]
        else:
            prior_df = np.full(n, _fit_fdist(var, df, covariate)["df2"])
    else:
        fit = _fit_fdist_unequal_df1(var, df, covariate, span=None, robust=robust)
        prior_df = fit.get("df2_shrunk")
        if prior_df is None:
            prior_df = np.full(n, fit["df2"])
    if np.any(np.isnan(prior_df)):
        raise EdgeRError("Could not estimate prior df")
    return prior_df


# ----------------------------------------------------------------------------
# estimateDisp
# ----------------------------------------------------------------------------
def _choose_lowess_span(n, small_n=50, min_span=0.3, power=1 / 3):
    if n == 0:
        return 1.0
    return min(min_span + (1 - min_span) * (small_n / n) ** power, 1.0)


def _wleb(
    theta,
    loglik,
    covariate,
    prior_n=None,
    span=None,
    m0=None,
    trend=True,
    individual=True,
    m0_out=False,
):
    ntags = loglik.shape[0]
    if span is None:
        span = _choose_lowess_span(ntags)
    out = {"span": span}
    if m0 is None:
        m0 = _locfit0_by_col(loglik, covariate, span)
    if trend:
        out["trend"] = _maximize_interpolant(theta, m0)
    if individual:
        pn = np.asarray(prior_n, dtype=float)
        pn = pn[:, None] if pn.ndim == 1 else pn
        out["individual"] = _maximize_interpolant(theta, loglik + pn * m0)
    if m0_out:
        out["shared_loglik"] = m0
    return out


def _comb_groups(zero):
    """Index groups of rows of a boolean matrix sharing the same pattern."""
    _, inv = np.unique(zero, axis=0, return_inverse=True)
    inv = np.asarray(inv).ravel()
    return [np.where(inv == g)[0] for g in range(inv.max() + 1)]


def estimate_disp(
    Y,
    offset,
    robust: bool = True,
    *,
    min_row_sum: float = 5,
    grid_length: int = 21,
    grid_range=(-10, 10),
    winsor_tail_p=(0.05, 0.1),
):
    """``edgeR::estimateDisp(Y, matrix(1, n), offset = offset, tagwise = TRUE, robust = robust)``.

    Returns a dict with ``common_dispersion``, ``trended_dispersion``,
    ``tagwise_dispersion``, ``span``, ``prior_df``, ``prior_n`` and the
    adjusted-profile-likelihood grid ``apl_grid`` (selected genes x grid points)
    with ``spline_pts`` and ``sel``.
    """
    y = np.asarray(Y, dtype=float)
    if y.ndim != 2:
        raise EdgeRError("y must be a matrix")
    ntags, nlibs = y.shape
    if ntags == 0:
        raise EdgeRError("no genes")
    offs = np.asarray(offset, dtype=float)
    if offs.ndim <= 1:
        if offs.size not in (1, nlibs):
            raise EdgeRError("offset has wrong length")
        offs = np.broadcast_to(np.broadcast_to(offs.ravel(), (nlibs,)), y.shape)
    elif offs.shape != y.shape:
        raise EdgeRError("Dimensions of offset don't agree with dimensions of y")
    if not np.all(np.isfinite(offs)):
        raise EdgeRError("offsets must be finite values")
    lib_size = y.sum(axis=0)

    ncoef = 1
    if ncoef >= nlibs:
        warnings.warn("No residual df: setting dispersion to NA", stacklevel=2)
        nan = np.full(ntags, np.nan)
        return {
            "common_dispersion": np.nan,
            "trended_dispersion": nan,
            "tagwise_dispersion": nan,
        }

    sel = y.sum(axis=1) >= min_row_sum
    nsel = int(sel.sum())
    if nsel == 0:
        raise EdgeRError("no genes with sufficient counts")
    sely = y[sel]
    seloff = np.ascontiguousarray(offs[sel])
    spline_pts = np.linspace(grid_range[0], grid_range[1], grid_length)
    spline_disp = 0.1 * 2.0**spline_pts
    l0 = np.zeros((nsel, grid_length))

    _, fitmu, _ = _glm_fit_one_group(sely, seloff, 0.05)
    zerofit = (sely < 1e-4) & (fitmu < 1e-4)
    for subg in _comb_groups(zerofit):
        cur_nzero = ~zerofit[subg[0]]
        if not cur_nzero.any():
            continue
        if not cur_nzero.all() and cur_nzero.sum() == 1:
            continue  # nrow(redesign) == ncol(redesign)
        cury = sely[np.ix_(subg, cur_nzero)]
        curo = seloff[np.ix_(subg, cur_nzero)]
        last_beta = None
        for i in range(grid_length):
            beta = _glm_one_group(cury, curo, spline_disp[i], last_beta)
            beta = np.maximum(beta, -1e8)
            with np.errstate(all="ignore"):
                mu = np.exp(curo + beta[:, None])
            l0[subg, i] = _compute_apl_one_coef(cury, mu, spline_disp[i])
            last_beta = beta

    overall = float(_maximize_interpolant(spline_pts, l0.sum(axis=0, keepdims=True))[0])
    common = 0.1 * 2.0**overall

    if np.any(lib_size <= 0):
        # aveLogCPM passes offset = log(lib.size); a zero-count cell gives -Inf
        raise EdgeRError("offsets must be finite values")
    ave = _ave_log_cpm(y, lib_size, common)
    out1 = _wleb(spline_pts, l0, ave[sel], span=None, trend=True, individual=False, m0_out=True)
    span = out1["span"]
    m0 = out1["shared_loglik"]
    disp_trend = 0.1 * 2.0 ** out1["trend"]
    trended = np.full(ntags, disp_trend[int(np.argmin(ave[sel]))])
    trended[sel] = disp_trend

    _, fitmu, dev = _glm_fit_one_group(sely, seloff, disp_trend)
    zerofit = (sely < 1e-4) & (fitmu < 1e-4)
    nzero = zerofit.sum(axis=1)
    df_resid = np.full(nsel, nlibs - ncoef, dtype=float)
    df_resid[nzero == nlibs] = 0
    somezero = (nzero > 0) & (nzero < nlibs)
    if np.any(somezero):
        # rank of the intercept design on the non-zero cells is 1
        df_resid[somezero] = np.maximum((nlibs - nzero[somezero]) - 1, 0)
    s2 = dev / df_resid
    s2[df_resid == 0] = 0
    s2 = np.maximum(s2, 0)
    dfp = df_resid[df_resid > 0]
    unequal_df = not (dfp.size > 0 and dfp.min() == dfp.max())
    prior_df = _squeeze_var_prior_df(s2, df_resid, ave[sel], robust, winsor_tail_p)
    prior_df = np.broadcast_to(np.asarray(prior_df, dtype=float), (nsel,)).copy()
    prior_n = prior_df / (nlibs - ncoef)

    tagwise = trended.copy()
    too_large = prior_n > 1e6
    if not np.all(too_large):
        temp_n = np.where(too_large, 1e6, prior_n)
        out2 = _wleb(
            spline_pts, l0, ave[sel], prior_n=temp_n, span=span, m0=m0, trend=False, individual=True
        )
        indiv = 0.1 * 2.0 ** out2["individual"]
        tag_sel = tagwise[sel]
        if not robust:
            tag_sel = indiv
        else:
            tag_sel[~too_large] = indiv[~too_large]
        tagwise[sel] = tag_sel
    if robust:
        full_df = np.full(ntags, np.inf)
        full_n = np.full(ntags, np.inf)
        full_df[sel] = prior_df
        full_n[sel] = prior_n
        prior_df, prior_n = full_df, full_n
    return {
        "common_dispersion": common,
        "trended_dispersion": trended,
        "tagwise_dispersion": tagwise,
        "span": span,
        "prior_df": prior_df,
        "prior_n": prior_n,
        "apl_grid": l0,
        "spline_pts": spline_pts,
        "sel": sel,
        "ave_log_cpm": ave,
        "unequal_df": unequal_df,
    }


def estimate_disp_tagwise(Y: np.ndarray, offset: np.ndarray, robust: bool = True) -> np.ndarray:
    """Tagwise dispersions from ``edgeR::estimateDisp`` as SpaNorm calls it.

    ``Y`` is genes x cells; ``offset`` is genes x cells or length cells.
    Raises :class:`EdgeRError` where edgeR errors (SpaNorm maps this to NA).
    """
    return estimate_disp(Y, offset, robust=robust)["tagwise_dispersion"]
