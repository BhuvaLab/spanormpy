"""Size factors: scran-style pooled sum factors, quickCluster and fastSizeFactors.

Port of the code reached by ``scran::quickCluster(Y)`` and
``scran::calculateSumFactors(Y, clusters = cl)`` (scran 1.38.0 / scuttle 1.20.0 /
bluster 1.20.0 defaults), plus ``R/fastSizeFactors.R``.  Everything works on a
genes x cells matrix (dense or ``scipy.sparse``); only the AnnData wrappers transpose.
"""

from __future__ import annotations

import warnings

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
from anndata import AnnData
from scipy.spatial import cKDTree

from ._filter import _check_counts, get_counts_genes_by_cells

_LOWWEIGHT = 1e-6  # scuttle:::LOWWEIGHT
_EPS = np.finfo(float).eps


# --------------------------------------------------------------------------------------
# small numeric helpers
# --------------------------------------------------------------------------------------
def _col_sums(Y) -> np.ndarray:
    return np.asarray(Y.sum(axis=0), dtype=float).ravel()


def _dense_block(Y, rows=None, cols=None) -> np.ndarray:
    """Return ``Y[rows, cols]`` as a dense float64 array."""
    if sp.issparse(Y):
        Y = sp.csr_matrix(Y) if rows is not None else sp.csc_matrix(Y)
        if rows is not None:
            Y = Y[rows]
        if cols is not None:
            Y = sp.csc_matrix(Y)[:, cols]
        return np.asarray(Y.toarray(), dtype=float)
    out = np.asarray(Y)
    if rows is not None:
        out = out[rows]
    if cols is not None:
        out = out[:, cols]
    return np.asarray(out, dtype=float)


def _weighted_median(x: np.ndarray, w: np.ndarray) -> float:
    """``limma::weighted.median(x, w, na.rm = TRUE)``."""
    ok = ~np.isnan(x)
    x, w = x[ok], w[ok]
    w_pos = w != 0
    x, w = x[w_pos], w[w_pos]
    if x.size == 0:
        return np.nan
    o = np.argsort(x, kind="stable")
    x, w = x[o], w[o]
    p = np.cumsum(w) / w.sum()
    n = int((p < 0.5).sum())
    if p[n] > 0.5:
        return float(x[n])
    return float((x[n] + x[n + 1]) / 2)


# --------------------------------------------------------------------------------------
# nls (Gauss-Newton, as R's default algorithm)
# --------------------------------------------------------------------------------------
class _NlsError(RuntimeError):
    pass


def _nls(fun, start, y, weights, maxiter=50, tol=1e-5, min_factor=1 / 1024, warn_only=False):
    """Weighted Gauss-Newton least squares mirroring ``stats::nls`` defaults.

    ``fun(theta)`` returns the model prediction. Returns ``(theta, converged)``. With
    ``warn_only`` failures inside the iteration return the current estimate (as
    ``nls.control(warnOnly = TRUE)``); a singular gradient at the start always raises.
    """
    sw = np.sqrt(weights)
    theta = np.array(start, dtype=float)
    p = theta.size

    def jac(th, f0):
        J = np.empty((f0.size, p))
        for i in range(p):
            d = np.sqrt(_EPS) * abs(th[i]) if th[i] != 0 else np.sqrt(_EPS)
            t2 = th.copy()
            t2[i] += d
            J[:, i] = (fun(t2) - f0) / d
        return J

    def decompose(th):
        f0 = fun(th)
        if not np.all(np.isfinite(f0)):
            raise _NlsError("non-finite model values")
        J = jac(th, f0) * sw[:, None]
        r = (y - f0) * sw
        if not np.all(np.isfinite(J)):
            raise _NlsError("non-finite gradient")
        Q, R = np.linalg.qr(J, mode="complete")
        return r, Q, R

    r, Q, R = decompose(theta)
    if np.linalg.matrix_rank(R[:p]) < p:
        raise _NlsError("singular gradient matrix at initial parameter estimates")
    rss = float(r @ r)
    for _ in range(maxiter):
        qty = Q.T @ r
        num, den = float(qty[:p] @ qty[:p]), float(qty[p:] @ qty[p:])
        conv = np.sqrt(num / den) if den > 0 else 0.0
        if conv < tol:
            return theta, True
        try:
            inc = sla.solve_triangular(R[:p], qty[:p])
        except Exception as e:  # singular R
            if warn_only:
                return theta, False
            raise _NlsError("singular gradient") from e
        factor, accepted = 1.0, False
        while factor >= min_factor:
            cand = theta + factor * inc
            try:
                fc = fun(cand)
                rc = (y - fc) * sw
                ok = np.all(np.isfinite(rc)) and float(rc @ rc) <= rss
            except FloatingPointError:
                ok = False
            if ok:
                accepted = True
                break
            factor /= 2
        if not accepted:
            if warn_only:
                return theta, False
            raise _NlsError("step factor reduced below minFactor")
        theta = cand
        try:
            r, Q, R = decompose(theta)
        except _NlsError:
            if warn_only:
                return theta, False
            raise
        if np.linalg.matrix_rank(R[:p]) < p:
            if warn_only:
                return theta, False
            raise _NlsError("singular gradient")
        rss = float(r @ r)
    if warn_only:
        return theta, False
    raise _NlsError("number of iterations exceeded maximum")


# --------------------------------------------------------------------------------------
# scuttle::cleanSizeFactors
# --------------------------------------------------------------------------------------
def _clean_size_factors(sf: np.ndarray, num_detected: np.ndarray, iterations=3, nmads=3):
    keep = sf > 0
    if keep.all():
        return sf
    sf = sf.copy()
    X, Y = sf[keep], num_detected[keep].astype(float)
    if X.size < 3:
        raise ValueError("need at least three positive values for trend fitting")
    lower = np.median(X) > X
    A = float(np.sum(X[lower] * Y[lower]) / np.sum(X[lower] ** 2))
    below = np.flatnonzero(Y < A * X)
    top = below[np.argmax(Y[below])]
    B = A / Y[top] - 1 / X[top]
    with np.errstate(all="ignore"):
        init = np.log(np.array([A, B]))
        lY, lX = np.log(Y), np.log(X)
        weights = np.ones(Y.size)

        def model(th):
            return th[0] + lX - np.log(X * np.exp(th[1]) + 1)

        coefs = init
        for _ in range(iterations + 1):
            try:
                coefs, _ok = _nls(model, init, lY, weights, maxiter=50, warn_only=True)
            except _NlsError:
                sf[~keep] = X.min()
                return sf
            init = coefs
            resids = np.abs(lY - model(coefs))
            bandwidth = max(1e-8, np.median(resids) * nmads)
            weights = (1 - np.minimum(resids / bandwidth, 1) ** 3) ** 3
        failed = num_detected[~keep].astype(float)
        new_sf = failed / (np.exp(coefs[0]) - np.exp(coefs[1]) * failed)
    new_sf[new_sf < 0] = X.max()
    sf[~keep] = new_sf
    return sf


# --------------------------------------------------------------------------------------
# scuttle pooled size factors (calculateSumFactors)
# --------------------------------------------------------------------------------------
def _guess_min_mean(Y, min_mean):
    if min_mean is None:
        mid_lib = np.median(_col_sums(Y))
        if np.isnan(mid_lib):
            return 1.0
        if mid_lib <= 50000:
            return 0.1
        if mid_lib >= 1e5:
            return 1.0
        warnings.warn("assuming UMI data when setting 'min.mean'", stacklevel=3)
        return 0.1
    return max(min_mean, 1e-8)


def _generate_sphere(lib_sizes: np.ndarray) -> np.ndarray:
    """scuttle:::.generateSphere (0-based). Length ``2 * ncells``."""
    o = np.argsort(lib_sizes, kind="stable")
    odd = o[0::2]  # R 1-based odd positions
    even = o[1::2]
    out = np.concatenate([odd, even[::-1]])
    return np.concatenate([out, out])


def _pool_factors(exprs: np.ndarray, pseudo: np.ndarray, order: np.ndarray, sizes, block=256):
    """Port of the C++ ``pool_size_factors``: median-of-ratios for every sliding pool.

    Returns an ``(nsizes, ncells)`` array; entry ``[si, win]`` is the pool made of the cells
    ``order[win : win + sizes[si]]`` (cumulative sums over the ordering).
    """
    ngenes, n = exprs.shape
    out = np.empty((len(sizes), n))
    smax = sizes[-1]
    for w0 in range(0, n, block):
        w1 = min(n, w0 + block)
        cells = order[w0 : w1 + smax - 1]
        cs = np.zeros((ngenes, cells.size + 1))
        np.cumsum(exprs[:, cells], axis=1, out=cs[:, 1:])
        wins = np.arange(w1 - w0)
        for si, s in enumerate(sizes):
            pool = cs[:, wins + s] - cs[:, wins]
            out[si, w0:w1] = np.median(pool / pseudo[:, None], axis=0)
    return out


def _solve_least_squares(A: sp.csr_matrix, b: np.ndarray) -> np.ndarray:
    """Least squares via the normal equations with iterative refinement."""
    AtA = (A.T @ A).toarray()
    Atb = A.T @ b
    try:
        cf = sla.cho_factor(AtA, check_finite=False)

        def solve(v):
            return sla.cho_solve(cf, v, check_finite=False)
    except sla.LinAlgError:
        pinv = np.linalg.pinv(AtA)

        def solve(v):
            return pinv @ v

    x = solve(Atb)
    for _ in range(3):
        x = x + solve(A.T @ (b - A @ x))
    return x


def _per_cluster_normalize(x: np.ndarray, sizes, min_mean, positive, scaling=None):
    if scaling is None:
        scaling = x.sum(axis=0)
    if np.any(scaling == 0):
        raise ValueError("cells should have non-zero library sizes or 'scaling' values")
    exprs = x / scaling
    ave_cell = exprs.mean(axis=1) * scaling.mean()
    high = ave_cell >= min_mean
    use_ave = ave_cell
    if not high.all():
        exprs = exprs[high]
        use_ave = ave_cell[high]
    if exprs.shape[0] == 0:
        raise ValueError("insufficient features for median calculations")
    n = exprs.shape[1]
    sphere = _generate_sphere(scaling)
    sizes = [s for s in sizes if s <= n]

    # linear system (scuttle:::.create_linear_system)
    ns = len(sizes)
    if ns:
        pf = _pool_factors(exprs, use_ave, sphere, sizes)
        rows, cols = [], []
        win = np.arange(n)
        for si, s in enumerate(sizes):
            rows.append(np.repeat(win + si * n, s))
            cols.append(sphere[win[:, None] + np.arange(s)[None, :]].ravel())
        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        resp = [pf.ravel()]
    else:
        rows = cols = np.empty(0, dtype=int)
        resp = []
    sq = np.sqrt(_LOWWEIGHT)
    rows = np.concatenate([rows, np.arange(n) + n * ns])
    cols = np.concatenate([cols, np.arange(n)])
    vals = np.concatenate([np.ones(rows.size - n), np.full(n, sq)])
    resp.append(np.full(n, sq / use_ave.sum()))
    b = np.concatenate(resp)
    A = sp.csr_matrix((vals, (rows, cols)), shape=(n * (ns + 1), n))

    final_nf = _solve_least_squares(A, b) * scaling
    if np.any(final_nf <= 0):
        warnings.warn("encountered non-positive size factor estimates", stacklevel=3)
        if positive:
            num_detected = (exprs > 0).sum(axis=0)
            final_nf = _clean_size_factors(final_nf, num_detected)
    return final_nf, ave_cell


def _limit_cluster_size(clusters: np.ndarray, max_size):
    """scuttle:::.limit_cluster_size (labels become strings only if a split is needed)."""
    _, counts = np.unique(clusters, return_counts=True)
    if max_size is None or not np.any(counts > max_size):
        return clusters
    clusters = clusters.astype(str).astype(object)
    out = clusters.copy()
    for cid in np.unique(clusters):
        idx = np.flatnonzero(clusters == cid)
        mult = int(np.ceil(idx.size / max_size))
        realloc = np.tile(np.arange(1, mult + 1), idx.size // mult + 1)[: idx.size]
        out[idx] = [f"{cid}-{r}" for r in realloc]
    return out.astype(str)


def _rescale_clusters(profiles, ref_idx, min_mean):
    ref = profiles[ref_idx]
    out = np.empty(len(profiles))
    for i, cur in enumerate(profiles):
        cl, rl = cur.sum(), ref.sum()
        to_use = (cur / cl + ref / rl) / 2 * (cl + rl) / 2 >= min_mean
        c, r = cur[to_use], ref[to_use]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = c / r
        ratio = ratio[~np.isnan(ratio)]
        sf = np.median(ratio) if ratio.size else np.nan
        if not np.isfinite(sf) or sf <= 0:
            warnings.warn(
                f"inter-cluster rescaling factor for cluster {i + 1} is not strictly "
                "positive, reverting to the ratio of average library sizes",
                stacklevel=3,
            )
            sf = c.sum() / r.sum()
        out[i] = sf
    return out


def sum_factors(
    Y,
    clusters=None,
    sizes=None,
    ref_clust=None,
    max_cluster_size: int | None = 3000,
    positive: bool = True,
    scaling=None,
    min_mean: float | None = None,
) -> np.ndarray:
    """Pooled size factors, an exact port of ``scran::calculateSumFactors``.

    ``Y`` is a genes x cells count matrix (dense or sparse). ``clusters`` gives a label per
    cell (``None`` means one cluster, split into chunks of ``max_cluster_size`` cells).
    Defaults are those of ``scuttle::pooledSizeFactors``: ``sizes = seq(21, 101, 5)``,
    ``max.cluster.size = 3000``, ``positive = TRUE``, ``min.mean`` guessed from the
    median library size.
    """
    ncells = Y.shape[1]
    sizes = np.arange(21, 102, 5) if sizes is None else np.asarray(sizes)
    clusters = np.zeros(ncells, dtype=int) if clusters is None else np.asarray(clusters)
    if clusters.ndim != 1 or clusters.size != ncells:
        raise ValueError("'ncol(x)' is not equal to 'length(clusters)'")
    clusters = _limit_cluster_size(clusters, max_cluster_size)
    labels = np.unique(clusters)  # sorted, like levels of factor()
    if labels.size == 0:
        raise ValueError("zero cells in one of the clusters")
    if scaling is not None:
        scaling = np.asarray(scaling, dtype=float)
        if scaling.size != ncells:
            raise ValueError("'length(scaling)' should be equal to 'ncol(x)'")
    min_mean = _guess_min_mean(Y, min_mean)
    sizes = np.sort(sizes.astype(int))
    if np.unique(sizes).size != sizes.size:
        raise ValueError("'sizes' are not unique")

    idxs = [np.flatnonzero(clusters == lab) for lab in labels]
    nfs, profiles = [], []
    for idx in idxs:
        x = _dense_block(Y, cols=idx)
        nf, ave = _per_cluster_normalize(
            x, sizes, min_mean, positive, None if scaling is None else scaling[idx]
        )
        nfs.append(nf)
        profiles.append(ave)
    if ref_clust is None:
        ref_idx = int(np.argmax([(p > 0).sum() for p in profiles]))
    else:
        hits = np.flatnonzero(labels.astype(str) == str(ref_clust))
        if hits.size == 0:
            raise ValueError("'ref.clust' not in 'clusters'")
        ref_idx = int(hits[0])
    resc = _rescale_clusters(profiles, ref_idx, min_mean)
    final = np.full(ncells, np.nan)
    for idx, nf, r in zip(idxs, nfs, resc, strict=True):
        final[idx] = nf * r
    is_pos = (final > 0) & ~np.isnan(final)
    return final / final[is_pos].mean()


# --------------------------------------------------------------------------------------
# quickCluster
# --------------------------------------------------------------------------------------
def _r_density_inverse_weights(x: np.ndarray) -> np.ndarray:
    """scran:::.inverse_density_weights: ``1 / density(x, from = min, to = max)`` at ``x``."""
    n_user, n = 512, 512
    lo_x, hi_x = x.min(), x.max()
    nx = x.size
    sd = x.std(ddof=1)
    iqr = np.subtract(*np.percentile(x, [75, 25])) / 1.34
    lo_bw = min(sd, iqr)
    if not lo_bw:
        lo_bw = sd or abs(x[0]) or 1.0
    bw = 0.9 * lo_bw * nx ** (-0.2)
    lo, up = lo_x - 4 * bw, hi_x + 4 * bw
    # linear binning (R's BinDist)
    y = np.zeros(2 * n)
    xdelta = (up - lo) / (n - 1)
    xpos = (x - lo) / xdelta
    ix = np.floor(xpos).astype(int)
    fx = xpos - ix
    wi = 1.0 / nx
    m = (ix >= 0) & (ix <= n - 2)
    np.add.at(y, ix[m], wi * (1 - fx[m]))
    np.add.at(y, ix[m] + 1, wi * fx[m])
    m = ix == -1
    np.add.at(y, ix[m] + 1, wi * fx[m])
    m = ix == n - 1
    np.add.at(y, ix[m], wi * (1 - fx[m]))
    kords = np.linspace(0, 2 * (up - lo), 2 * n)
    kords[n + 1 :] = -kords[1:n][::-1]
    kords = np.exp(-0.5 * (kords / bw) ** 2) / (bw * np.sqrt(2 * np.pi))
    conv = np.fft.ifft(np.fft.fft(y) * np.conj(np.fft.fft(kords)))
    dens = np.maximum(0, conv.real[:n])
    xords = np.linspace(lo, up, n)
    xg = np.linspace(lo_x, hi_x, n_user)
    dg = np.interp(xg, xords, dens)
    w = 1 / np.interp(x, xg, dg)
    return w / w.mean()


def _weighted_lowess(x, y, weights, span=0.3, iterations=4, npts=200):
    """Port of ``limma::weightedLowess`` (C code ``weighted_lowess``); returns fitted values."""
    thr = 1e-7
    x = np.asarray(x, float)
    o = np.argsort(x, kind="stable")
    xs, ys, ws = x[o], np.asarray(y, float)[o], np.asarray(weights, float)[o]
    n = xs.size
    if npts >= n:
        delta = 0.0
    else:
        dx = np.sort(np.diff(xs))
        cum = np.cumsum(dx)
        nc = np.arange(npts)
        delta = float(np.min(cum[dx.size - 1 - nc] / (npts - nc)))
    # seeds
    seeds, last = [0], 0
    for pt in range(1, n - 1):
        if xs[pt] - xs[last] > delta:
            seeds.append(pt)
            last = pt
    seeds.append(n - 1)
    seeds = np.array(seeds)
    spanweight = ws.sum() * span
    subrange = (xs[-1] - xs[0]) / n
    starts, ends, dists = [], [], []
    for curpt in seeds:
        left = right = curpt
        curw = ws[curpt]
        ende, ends_ = curpt == n - 1, curpt == 0
        md = 0.0
        while curw < spanweight and (not ende or not ends_):
            if ende:
                left -= 1
                curw += ws[left]
                ends_ = left == 0
                md = max(md, xs[curpt] - xs[left])
            elif ends_:
                right += 1
                curw += ws[right]
                ende = right == n - 1
                md = max(md, xs[right] - xs[curpt])
            else:
                ld = xs[curpt] - xs[left - 1]
                rd = xs[right + 1] - xs[curpt]
                if ld < rd:
                    left -= 1
                    curw += ws[left]
                    ends_ = left == 0
                    md = max(md, ld)
                else:
                    right += 1
                    curw += ws[right]
                    ende = right == n - 1
                    md = max(md, rd)
        while left > 0 and xs[left] == xs[left - 1]:
            left -= 1
        while right < n - 1 and xs[right] == xs[right + 1]:
            right += 1
        starts.append(left)
        ends.append(right)
        dists.append(md)

    def fit_at(curpt, left, right, dist, rw):
        sl = slice(left, right + 1)
        xx, yy = xs[sl], ys[sl]
        if dist < thr:
            wk = ws[sl] * rw[sl]
            return float((yy * wk).sum() / wk.sum())
        wk = (1 - (np.abs(xs[curpt] - xx) / dist) ** 3) ** 3 * ws[sl] * rw[sl]
        aw = wk.sum()
        xm, ym = (wk * xx).sum() / aw, (wk * yy).sum() / aw
        var = (wk * (xx - xm) ** 2).sum()
        covar = (wk * (xx - xm) * (yy - ym)).sum()
        if var < thr:
            return float(ym)
        slope = covar / var
        return float(slope * xs[curpt] + (ym - slope * xm))

    rob = np.ones(n)
    fit = np.zeros(n)
    totalw = ws.sum()
    for _ in range(iterations):
        fit[0] = fit_at(0, starts[0], ends[0], dists[0], rob)
        last_pt = 0
        for s in range(1, len(seeds)):
            pt = seeds[s]
            fit[pt] = fit_at(pt, starts[s], ends[s], dists[s], rob)
            if pt - last_pt > 1:
                cur = xs[pt] - xs[last_pt]
                if cur > thr * subrange:
                    slope = (fit[pt] - fit[last_pt]) / cur
                    icpt = fit[pt] - slope * xs[pt]
                    fit[last_pt + 1 : pt] = slope * xs[last_pt + 1 : pt] + icpt
                else:
                    fit[last_pt + 1 : pt] = 0.5 * (fit[pt] + fit[last_pt])
            last_pt = pt
        rsd = np.abs(ys - fit)
        scale = rsd.mean()
        ro = np.argsort(rsd, kind="stable")
        rs = rsd[ro]
        cw = np.cumsum(ws[ro])
        half = totalw / 2
        cmad = 0.0
        hit = np.flatnonzero(cw >= half)
        if hit.size:
            pt = hit[0]
            if cw[pt] == half:
                cmad = 3 * (rs[pt] + rs[min(pt + 1, n - 1)])
            else:
                cmad = 6 * rs[pt]
        if cmad <= thr * scale:
            break
        r = np.where(rs < cmad, (1 - (rs / cmad) ** 2) ** 2, 0.0)
        rob[ro] = r
    out = np.empty(n)
    out[o] = fit
    return out


def _fit_trend_var(means, variances, min_mean=0.1):
    """scran::fitTrendVar with defaults; returns ``trend(m)``."""
    ok = ~np.isnan(variances) & (variances > 1e-8) & (means >= min_mean)
    v, m = variances[ok], means[ok]
    w = _r_density_inverse_weights(m)
    if v.size < 2:
        raise ValueError("need at least 2 points for non-parametric curve fitting")
    if v.size <= 3:
        raise ValueError("need at least 4 points for non-linear curve fitting")
    to_fit = np.log(v)

    # starting values (scran:::.get_nls_starts)
    o = np.argsort(means[ok], kind="stable")
    left_n = int(min(100, v.size * 0.1))
    keep = o[: max(1, left_n)]
    yk, xk = v[keep], m[keep]
    grad = float((xk * yk).sum() / (xk * xk).sum())
    bpts = 2.0 ** np.linspace(-5, 5, 10)
    npts_ = 2.0 ** np.linspace(0, 10, 10)
    best, hitB, hitN = np.inf, None, None
    with np.errstate(over="ignore"):
        for nn in npts_:  # expand.grid: B varies fastest
            for B in bpts:
                ss = float(((v - grad * B * m / (m**nn + B)) ** 2).sum())
                if ss < best:
                    best, hitB, hitN = ss, B, nn
    A0, B0, N0 = np.log(hitB * grad), np.log(hitB), np.log(max(1e-8, hitN - 1))

    def param(th, x):
        return np.exp(th[0]) * x / (x ** (1 + np.exp(th[2])) + np.exp(th[1]))

    with np.errstate(all="ignore"):
        try:
            th, _ = _nls(lambda t: param(t, m), [A0, B0, N0], v, w, maxiter=500)
            fitted = param(th, m)
            paramfun = lambda x: param(th, x)
        except _NlsError:
            th = np.array([A0, B0, N0])
            paramfun = lambda x: param(th, x)
            fitted = paramfun(m)
        to_fit = to_fit - np.log(fitted)
        lfit = _weighted_lowess(m, to_fit, w)
        # approxfun(m, fitted, rule = 2): ties averaged
        um, inv = np.unique(m, return_inverse=True)
        uy = np.bincount(inv, weights=lfit) / np.bincount(inv)

        def unscaled(x):
            return np.exp(np.interp(x, um, uy)) * paramfun(x)

        leftovers = v / unscaled(m)
        med = _weighted_median(leftovers, w)
    return lambda x: unscaled(np.asarray(x, float)) * med


def _gene_mean_var_log(Y, sf, rows=None, block=2000):
    """Mean and variance (n - 1) of ``log2(Y / sf + 1)`` for each gene."""
    idx = np.arange(Y.shape[0]) if rows is None else np.asarray(rows)
    n = Y.shape[1]
    means = np.empty(idx.size)
    vars_ = np.empty(idx.size)
    for s in range(0, idx.size, block):
        part = idx[s : s + block]
        z = np.log2(_dense_block(Y, rows=part) / sf + 1)
        means[s : s + block] = z.mean(axis=1)
        vars_[s : s + block] = z.var(axis=1, ddof=1) if n > 1 else np.nan
    return means, vars_


def _denoise_pc_number(var_exp, var_tech, var_total):
    flipped = var_exp[::-1]
    est = np.cumsum(flipped) + (var_total - flipped.sum())
    above = np.flatnonzero(est > var_tech)
    if above.size:
        return len(var_exp) - int(above[0])  # npcs - min(which) + 1 with 1-based which
    return 1


def _centered_svd(y: np.ndarray, k: int, random_state: int):
    yc = y - y.mean(axis=0)
    k = min(k, min(yc.shape))
    if min(yc.shape) <= 1000:
        u, d, _ = np.linalg.svd(yc, full_matrices=False)
        return u[:, :k], d[:k]
    from scipy.sparse.linalg import svds

    rng = np.random.default_rng(random_state)
    u, d, _ = svds(yc, k=k, v0=rng.standard_normal(min(yc.shape)))
    o = np.argsort(-d)
    return u[:, o], d[o]


def _denoised_pcs(Y, sf, random_state: int) -> np.ndarray:
    """``modelGeneVar`` -> ``getTopHVGs(n=500, prop=0.1)`` -> ``getDenoisedPCs`` (cells x PCs)."""
    ncells = Y.shape[1]
    means, total = _gene_mean_var_log(Y, sf)
    trend = _fit_trend_var(means, total)
    tech = trend(means)
    bio = total - tech
    # getTopHVGs: var.threshold = 0 on "bio", then head(order(decreasing), max(500, round(0.1 n)))
    survivors = np.flatnonzero(~np.isnan(bio) & (bio > 0))
    o = np.argsort(-bio[survivors], kind="stable")
    n_top = max(500, int(np.round(0.1 * survivors.size)))
    chosen = survivors[o[:n_top]]
    # getDenoisedPCs with a DataFrame of technical statistics
    all_var = total[chosen]
    tech_var = tech[chosen]  # scale = all.var / total.var = 1
    keep = all_var > tech_var
    use = chosen[keep]
    tech_var, all_var = tech_var[keep], all_var[keep]
    if use.size == 0:
        raise ValueError("no genes with biological variance exceeding the technical trend")
    y = np.log2(_dense_block(Y, rows=use) / sf + 1).T
    u, d = _centered_svd(y, 50, random_state)
    var_exp = d**2 / (ncells - 1)
    npcs = _denoise_pc_number(var_exp, tech_var.sum(), all_var.sum())
    npcs = max(npcs, min(5, var_exp.size))
    return u[:, :npcs] * d[:npcs]


def _snn_graph_rank(pcs: np.ndarray, k: int = 10):
    """bluster::makeSNNGraph(type = "rank"): returns ``(edges, weights)`` (0-based)."""
    n = pcs.shape[0]
    tree = cKDTree(pcs)
    _, idx = tree.query(pcs, k=min(k + 1, n))
    nbrs = np.empty((n, idx.shape[1] - 1), dtype=int)
    for i in range(n):
        row = idx[i][idx[i] != i]
        nbrs[i] = row[: nbrs.shape[1]]
    kk = nbrs.shape[1]
    hosts: list[list[tuple[int, int]]] = [[(i, 0)] for i in range(n)]
    for i in range(n):
        for r, x in enumerate(nbrs[i], start=1):
            hosts[x].append((i, r))
    edges, weights = [], []
    for j in range(n):
        score: dict[int, int] = {}
        for i in range(kk + 1):
            cur = j if i == 0 else nbrs[j, i - 1]
            for other, r in hosts[cur]:
                if other < j:
                    cr = r + i
                    old = score.get(other)
                    if old is None or old > cr:
                        score[other] = cr
        for other, sc in score.items():
            edges.append((j, other))
            weights.append(max(kk - 0.5 * sc, 1e-6))
    return edges, np.asarray(weights)


def _merge_closest_graph(g, clusters: np.ndarray, min_size: int) -> np.ndarray:
    """scran:::.merge_closest_graph."""

    def modularity(cl):
        _, mem = np.unique(cl, return_inverse=True)
        return g.modularity(mem.tolist(), weights="weight")

    clusters = clusters.copy()
    while True:
        names, sizes = np.unique(clusters, return_counts=True)
        if np.all(sizes >= min_size):
            break
        if names.size == 2:
            clusters[:] = 1
            break
        failed = names[np.argmin(sizes)]
        to_merge = clusters == failed
        max_m, max_clust = 0.0, clusters
        for other in names:
            if other == failed:
                continue
            nxt = clusters.copy()
            nxt[to_merge] = other
            m = modularity(nxt)
            if max_m < m:
                max_m, max_clust = m, nxt
        clusters = max_clust
    _, codes = np.unique(clusters, return_inverse=True)
    return codes + 1


def quick_cluster(Y, random_state: int = 0, min_size: int = 100) -> np.ndarray:
    """Port of ``scran::quickCluster(Y)`` with defaults (labels are 1-based like R's factor).

    Call chain (scran 1.38.0): ``min.size = 100``, ``method = "igraph"``, ``use.ranks = FALSE``,
    ``d = NULL``, ``block = NULL``, ``graph.fun = "walktrap"``;
    ``librarySizeFactors`` -> ``normalizeCounts`` (log2, pseudo-count 1) -> ``modelGeneVar`` ->
    ``getTopHVGs(n = 500, prop = 0.1)`` -> ``getDenoisedPCs`` -> ``NNGraphParam`` (``k = 10``,
    ``type = "rank"``, exact KNN) -> igraph walktrap (``steps = 4``) -> ``.merge_closest_graph``.
    ``random_state`` only seeds the sparse SVD fallback for very wide gene sets.
    """
    import igraph as ig

    ncells = Y.shape[1]
    ls = _col_sums(Y)
    sf = ls / ls.mean()
    if np.any(~(sf > 0)):
        raise ValueError("size factors should be positive")
    pcs = _denoised_pcs(Y, sf, random_state)
    if ncells < min_size:
        raise ValueError("fewer cells than the minimum cluster size")
    edges, weights = _snn_graph_rank(pcs, k=10)
    g = ig.Graph(n=ncells, edges=edges, directed=False)
    g.es["weight"] = weights.tolist()
    comm = g.community_walktrap(weights="weight", steps=4).as_clustering()
    membership = np.asarray(comm.membership, dtype=int) + 1
    return _merge_closest_graph(g, membership, min_size)


# --------------------------------------------------------------------------------------
# AnnData wrappers
# --------------------------------------------------------------------------------------
def _write_obs(adata: AnnData, key: str, values: np.ndarray) -> None:
    if adata.is_view:
        adata._init_as_actual(adata.copy())
    adata.obs[key] = values


def compute_size_factors(
    adata: AnnData,
    clusters=None,
    layer: str | None = None,
    key_added: str = "spanorm_size_factors",
    copy: bool = False,
    random_state: int = 0,
):
    """Scran-style pooled size factors (the default for SpaNorm).

    Equivalent to ``scran::calculateSumFactors(Y, clusters = scran::quickCluster(Y))``. If
    ``clusters`` is ``None`` the cells are clustered with :func:`quick_cluster`; otherwise it
    is an array of labels or the name of an ``obs`` column. The result is written to
    ``adata.obs[key_added]`` (and returned if ``copy=True``). ``obs["size_factors"]`` is never read.
    """
    Y = get_counts_genes_by_cells(adata, layer)
    _check_counts(Y)
    if clusters is None:
        cl = quick_cluster(Y, random_state=random_state)
    elif isinstance(clusters, str):
        cl = np.asarray(adata.obs[clusters])
    else:
        cl = np.asarray(clusters)
    sf = sum_factors(Y, cl)
    _write_obs(adata, key_added, sf)
    return sf if copy else None


def _fast_size_factors(Y) -> np.ndarray:
    """``fastSizeFactors_intl``: library sizes scaled to mean one."""
    ls = _col_sums(Y)
    return ls / ls.mean()


def fast_size_factors(adata: AnnData, layer: str | None = None) -> None:
    """Library-size factors (R ``fastSizeFactors``), written to ``obs["spanorm_size_factors"]``."""
    Y = get_counts_genes_by_cells(adata, layer)
    _check_counts(Y)
    _write_obs(adata, "spanorm_size_factors", _fast_size_factors(Y))
